"""Prueba 1b: limpiar manchas de color y medir cuántos cambios de filamento ahorra.

Uso:  python pruebas/limpiar_color.py modelo.glb [area_minima_mm2 ...]   (por defecto 3)
Escribe <modelo>_limpio<area>.3mf y _vista.glb en pruebas/salida/ e imprime una
estimación de cambios de filamento (capas de 0.2 mm). La estimación sirve para
comparar variantes; la cifra real la da Bambu Studio al laminar.

Probado y descartado: k-means en Lab. Con 4 colores juntó el blanco de piel y
pelaje con el gris y partió el naranja en dos.
"""
import os
import sys

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import acabados  # noqa: E402
import multicolor  # noqa: E402

SALIDA = os.path.join(RAIZ, "pruebas", "salida")
CAPA = 0.2


def estimar_cambios(malla, et):
    """Por capa: (colores presentes - 1), más uno si no comparte color con la anterior."""
    z = malla.vertices[malla.faces][:, :, 2]
    capas = int(np.ceil(malla.bounds[1, 2] / CAPA))
    a = np.clip(np.ceil((z.min(1) - CAPA / 2) / CAPA), 0, capas).astype(int)
    b = np.clip(np.floor((z.max(1) - CAPA / 2) / CAPA), -1, capas - 1).astype(int) + 1
    dif = np.zeros((capas + 1, et.max() + 1), np.int64)
    ok = b > a
    np.add.at(dif, (a[ok], et[ok]), 1)
    np.add.at(dif, (b[ok], et[ok]), -1)
    cambios, previo = 0, None
    for fila in np.cumsum(dif, 0)[:capas] > 0:
        if fila.any():
            cambios += int(fila.sum()) - 1 + (previo is not None and not (fila & previo).any())
            previo = fila
    return cambios


def regiones(malla, et):
    p = malla.face_adjacency
    p = p[et[p[:, 0]] == et[p[:, 1]]]
    n = len(et)
    return connected_components(sparse.coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n)),
                                directed=False)[0]


def main():
    glb = sys.argv[1]
    textura = trimesh.load(glb, force="mesh")
    pieza = acabados.aplicar(textura, "ninguno", 100)
    tx = textura.copy()
    tx.apply_transform(pieza.metadata["transformacion"])
    centros = pieza.metadata["vertices_base"][pieza.faces].mean(axis=1)
    et, pal = multicolor.cuantizar(multicolor.colores_en_puntos(tx, centros), 4)
    base = os.path.join(SALIDA, os.path.splitext(os.path.basename(glb))[0])
    print(f"{'sin limpiar':>12}: cambios~{estimar_cambios(pieza, et):5d}  regiones={regiones(pieza, et):5d}")
    suave = multicolor.suavizar(et, pieza)
    for amin in [float(x) for x in sys.argv[2:]] or [multicolor.AREA_MANCHA_MM2]:
        et_c = multicolor.absorber_manchas(suave, pieza, amin)
        ruta = f"{base}_limpio{amin:g}.3mf"
        multicolor.exportar_3mf(ruta, pieza, et_c, pal)
        vista = trimesh.Trimesh(pieza.vertices, pieza.faces, process=False)
        vista.visual.face_colors = np.column_stack([pal[et_c], np.full(len(et_c), 255)])
        vista.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        vista.export(ruta[:-4] + "_vista.glb")
        print(f"{f'{amin:g} mm2':>12}: cambios~{estimar_cambios(pieza, et_c):5d}  "
              f"regiones={regiones(pieza, et_c):5d}  -> {ruta}")


if __name__ == "__main__":
    main()
