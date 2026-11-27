"""Pruebas 4 y 5: despiece para lijar y pintar y orientación de cada pieza.

El método está en pintura.py (la app lo usa desde la pestaña Dividir). Este
script lo ejecuta sobre un .glb a 100 mm y deja junto a la salida:
  _pintura.3mf / .zip   piezas orientadas para imprimir (3MF pintado con 4 colores)
  _pintura_vista.glb    vista separada (piezas en su sitio)
  _pintura_color.glb    vista separada con los colores, para revisar
y una tabla por pieza. Con CACHE_CORTES=archivo.pkl guarda la búsqueda de zonas
(la parte lenta) para repetir.

Uso:  python pruebas/cortes_pintura.py modelo.glb [max_piezas]
"""
import os
import pickle
import sys
import time

import numpy as np
import trimesh

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import acabados  # noqa: E402
import multicolor  # noqa: E402
import pintura  # noqa: E402
import segmentar  # noqa: E402

SALIDA = os.path.join(RAIZ, "pruebas", "salida")


def main():
    glb = sys.argv[1]
    max_piezas = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    textura = trimesh.load(glb, force="mesh")
    malla = acabados.aplicar(textura, "ninguno", 100)
    tx = textura.copy()
    tx.apply_transform(malla.metadata["transformacion"])
    centros = malla.metadata["vertices_base"][malla.faces].mean(axis=1)
    et, pal = multicolor.cuantizar(multicolor.colores_en_puntos(tx, centros), 4)
    et = multicolor.absorber_manchas(multicolor.suavizar(et, malla), malla)

    cache = os.environ.get("CACHE_CORTES")
    cands = pickle.load(open(cache, "rb")) if cache and os.path.exists(cache) else None
    res, informe, datos = pintura.dividir_para_pintar(
        malla, et, max_piezas, candidatas=cands,
        avisar=lambda f, txt: print(f"{txt}  ({time.time() - t0:.0f} s)"))
    if cache and cands is None:
        pickle.dump(datos["candidatas"], open(cache, "wb"))
    for linea in informe:
        print(linea)

    montadas = datos["partes"]
    etiquetas = segmentar.colores_de_partes(segmentar.Resultado(partes=montadas), malla, et)
    base = os.path.join(SALIDA, os.path.splitext(os.path.basename(glb))[0] + "_pintura")
    segmentar.exportar_3mf(base + ".3mf", res, etiquetas, pal)
    segmentar.exportar_zip(base + ".zip", res)
    segmentar.vista_explosionada(montadas, base + "_vista.glb")
    c0 = malla.bounds.mean(0)
    vista = []
    for p, e in zip(montadas, etiquetas):
        v = p.copy()
        v.visual.face_colors = np.column_stack([pal[e], np.full(len(e), 255)])
        v.apply_translation((p.bounds.mean(0) - c0) * 0.4)
        vista.append(v)
    v = trimesh.util.concatenate(vista)
    v.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    v.export(base + "_color.glb")

    print(f"\n{'pieza':>5} {'medidas mm':>20} {'color ppal':>10} {'girada':>20} {'apoyo mm2':>9} "
          f"{'sop.vis':>7} {'sop.cm3':>7}")
    for i, (p, e, g, f) in enumerate(zip(montadas, etiquetas, res.partes, datos["filas"]), 1):
        a = p.area_faces
        dom = np.bincount(e, a, len(pal))
        print(f"{i:>5} {str(p.extents.round(0).tolist()):>20} {dom.max() / a.sum():10.0%} "
              f"{str(g.extents.round(0).tolist()):>20} {f['apoyo_mm2']:9.0f} {f['soporte_vis']:7.0f} "
              f"{f['soporte_mm3'] / 1000:7.1f}")
    print(f"{len(res.pasadores)} pasadores.  {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
