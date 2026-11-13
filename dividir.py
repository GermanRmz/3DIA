"""Divide un modelo ya generado en partes para imprimirlo por piezas.

Uso:
    python dividir.py modelo.glb --plantilla 6 --union bola --altura-mm 120
    python dividir.py modelo_impresion.stl --plantilla 4 --union espiga
    python dividir.py modelo.glb --cortes "z=40" --union imanes

Entradas: .glb/.gltf/.obj (salida de generar3d o app; se escala a --altura-mm) o
.stl/.3mf/.ply (malla de impresión, ya en mm con Z arriba; se respeta su tamaño
salvo que se indique --altura-mm). Con un .glb con textura y --colores N, el 3MF
sale pintado por filamentos.

Salidas junto al modelo: <nombre>_partes.zip (un STL por parte), <nombre>_partes.3mf
(todas colocadas en la cama) y <nombre>_partes.glb (vista separada).
"""
import argparse
import os
import time

import numpy as np
import trimesh

import acabados
import formas
import personaje
import segmentar

FORMATOS_MODELO = (".glb", ".gltf", ".obj")


def cargar(ruta: str, altura_mm: float | None):
    """Devuelve (malla de impresión en mm Z arriba, malla con textura o None).

    Un *_impresion.stl de esta app se acompaña de su .glb con color y de la
    transformación usada (*_impresion.txt): con ellos se recupera el color."""
    malla = trimesh.load(ruta, force="mesh")
    textura = None
    if ruta.lower().endswith(FORMATOS_MODELO):
        if getattr(malla.visual, "kind", None) == "texture":
            textura = malla
        return acabados.a_impresion(malla, altura_mm or 60), textura

    impresion = trimesh.Trimesh(malla.vertices, malla.faces, process=True)
    transformacion = None
    base = os.path.splitext(ruta)[0]
    if base.endswith("_impresion") and os.path.exists(base + ".txt") and \
            os.path.exists(base[:-len("_impresion")] + ".glb"):
        con_color = trimesh.load(base[:-len("_impresion")] + ".glb", force="mesh")
        if getattr(con_color.visual, "kind", None) == "texture":
            textura, transformacion = con_color, np.loadtxt(base + ".txt")
    escala = (altura_mm / impresion.extents[2]) if altura_mm else 1.0
    impresion.apply_scale(escala)
    dz = -impresion.bounds[0][2]
    impresion.apply_translation([0, 0, dz])
    if transformacion is not None:
        transformacion = trimesh.transformations.translation_matrix([0, 0, dz]) @ \
            np.diag([escala, escala, escala, 1.0]) @ transformacion
    impresion.metadata["transformacion"] = transformacion
    return impresion, textura


def colores_por_parte(partes, impresion, textura, n_colores):
    """Etiquetas de filamento por cara de cada parte y paleta (desde la textura)."""
    import multicolor

    tex = textura.copy()
    tex.apply_transform(impresion.metadata["transformacion"])
    centros = [p.triangles_center for p in partes]
    colores = multicolor.colores_en_puntos(tex, np.vstack(centros))
    etiquetas, paleta = multicolor.cuantizar(colores, n_colores)
    cortes = np.cumsum([len(c) for c in centros])[:-1]
    return np.split(etiquetas, cortes), paleta


def dividir_modelo(ruta: str, plantilla: str | None = "6", cortes_manual: str = "",
                   union: str = "espiga", diametro: float = 0.0, holgura: float = 0.2,
                   altura_mm: float | None = None, colores: int = 1,
                   iman=(6.0, 3.0), salida_base: str | None = None, max_partes: int = 8):
    """Divide el modelo. Devuelve (rutas de salida, texto de informe)."""
    t = time.time()
    impresion, textura = cargar(ruta, altura_mm)
    informe = [f"Modelo: {impresion.extents.round(1)} mm, {len(impresion.faces)} caras."]
    if plantilla == "auto":
        if union not in personaje.UNIONES:
            union = "espiga"
        d = formas.dividir(impresion, max_partes, personaje.OpcionesUnion(union, diametro, holgura,
                                                                          iman[1]))
        res = segmentar.Resultado(partes=[m for _, m in d.partes], avisos=d.avisos,
                                  nombres=[n for n, _ in d.partes])
    elif plantilla:
        if union not in personaje.UNIONES:
            informe.append(f"La unión '{union}' es para cortes planos; en personajes se usa espiga.")
            union = "espiga"
        d = personaje.dividir(impresion, plantilla, personaje.OpcionesUnion(union, diametro, holgura))
        p = d.puntos
        detectado = [f"cuello z={p.cuello:.1f}" if p.cuello else "cuello: no encontrado",
                     f"entrepierna z={p.entrepierna:.1f}" if p.entrepierna else "entrepierna: no encontrada",
                     f"hombros x={p.hombro_izq:.1f}/{p.hombro_der:.1f}" if p.hombro_izq is not None
                     else "hombros: brazos pegados al cuerpo"]
        informe.append("Detectado: " + ", ".join(detectado) + " (mm).")
        res = segmentar.Resultado(partes=[m for _, m in d.partes], avisos=d.avisos,
                                  nombres=[n for n, _ in d.partes])
    else:
        tipo = union if union in segmentar.UNIONES else "espiga"
        opciones = segmentar.OpcionesUnion(tipo=tipo, diametro=diametro or 5.0, holgura=holgura,
                                           iman_diametro=iman[0], iman_alto=iman[1])
        res = segmentar.segmentar(impresion, segmentar.planificar(impresion, "manual", manual=cortes_manual),
                                  opciones)
    informe += res.avisos
    etiquetas, paleta = None, None
    if colores > 1:
        if textura is None or impresion.metadata.get("transformacion") is None:
            informe.append("Colores: el modelo no tiene textura (usa el .glb con color); se exporta sin pintar.")
        else:
            etiquetas, paleta = colores_por_parte(res.partes, impresion, textura, colores)
            informe.append("Filamentos: " + ", ".join(f"{i + 1}: #{r:02X}{g:02X}{b:02X}"
                                                      for i, (r, g, b) in enumerate(paleta)))
    base = salida_base or os.path.splitext(ruta)[0]
    rutas = {"zip": base + "_partes.zip", "3mf": base + "_partes.3mf", "vista": base + "_partes.glb"}
    segmentar.exportar_zip(rutas["zip"], res)
    segmentar.exportar_3mf(rutas["3mf"], res, etiquetas, paleta)
    segmentar.vista_explosionada(res.partes, rutas["vista"])
    informe.append(f"{len(res.partes)} partes: " + ", ".join(
        f"{n} ({'x'.join(f'{d:.0f}' for d in m.extents)} mm)"
        for n, m in zip(res.nombres or [f"parte {i + 1}" for i in range(len(res.partes))], res.partes))
        + (f" + {len(res.pasadores)} pasadores" if res.pasadores else "") + f". ({time.time() - t:.1f} s)")
    return rutas, "\n".join(informe)


def main():
    p = argparse.ArgumentParser(description="Divide un modelo 3D en partes con uniones")
    p.add_argument("modelo", help=".glb/.obj (de la app) o .stl/.3mf (malla de impresión)")
    p.add_argument("--plantilla", default="6", choices=list(personaje.PLANTILLAS) + ["auto"],
                   help="; ".join(f"{k}: {v[0]}" for k, v in personaje.PLANTILLAS.items())
                   + "; auto: cualquier forma (separa lo que sobresale: patas, cola, cabeza...)")
    p.add_argument("--max-partes", type=int, default=8, help="Máximo de partes en modo auto")
    p.add_argument("--cortes", default="", help='En vez de plantilla, cortes planos: "z=40, x=0" (mm)')
    p.add_argument("--union", default="espiga",
                   choices=sorted(set(personaje.UNIONES) | set(segmentar.UNIONES)),
                   help="espiga o bola para personajes; con --cortes también pasadores e imanes")
    p.add_argument("--diametro", type=float, default=0.0, help="Diámetro de la unión en mm (0 = automático)")
    p.add_argument("--holgura", type=float, default=0.2, help="Juego en mm (súbelo si no encaja)")
    p.add_argument("--altura-mm", type=float, help="Altura final de la figura (mm)")
    p.add_argument("--colores", type=int, default=1, help="Filamentos para el 3MF pintado (.glb con textura)")
    args = p.parse_args()
    rutas, informe = dividir_modelo(args.modelo, None if args.cortes else args.plantilla, args.cortes,
                                    args.union, args.diametro, args.holgura, args.altura_mm, args.colores,
                                    max_partes=args.max_partes)
    print(informe)
    print("Archivos:", ", ".join(rutas.values()))


if __name__ == "__main__":
    main()
