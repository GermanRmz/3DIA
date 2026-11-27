"""Prueba 3: medidor de orientación de impresión (FDM, PETG, A1 mini).

Para ~200 direcciones "arriba" mide, sobre la malla con sus colores:
  escalones_frente  área con escalón de capa visible en el frente (mm², ponderada por
                    la altura del escalón: capa * |cos| del ángulo con la vertical)
  soporte_frente    área del frente que necesita soporte (deja marcas al quitarlo)
  soporte_mm3       volumen aproximado de soporte (área colgante x altura a la cama)
  cambios           cambios de filamento estimados (como en limpiar_color.py)
  capas, apoyo_mm2  número de capas y área apoyada en la cama si se recortan 0.6 mm
Además de la esfera de direcciones se prueban las caras planas grandes del casco
convexo (p. ej. la base de la figura, aunque el modelo de IA la deje inclinada).
El "frente" es lo que mira a la cámara en la imagen de origen (en la malla de
impresión, -Y) y la cabeza (40 % superior) cuenta el doble.

Puntaje (menor = mejor): primero calidad, luego costo. Cada medida se divide por
su mediana entre todas las orientaciones para poder sumarlas:
  2*escalones_frente + 2*soporte_frente + 1*soporte_mm3 + 0.5*cambios
  + hasta 2 si el apoyo es menor de 150 mm²
Uso:  python pruebas/orientacion.py modelo.glb
"""
import os
import sys
import time

import numpy as np
import trimesh

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "pruebas"))
import acabados  # noqa: E402
import multicolor  # noqa: E402
from corte_color import a_manifold, cambios, de_manifold, presencia  # noqa: E402

SALIDA = os.path.join(RAIZ, "pruebas", "salida")
CAPA = 0.2
ANGULO_SOPORTE = 40      # grados desde la horizontal; más plano que esto necesita soporte
RECORTE_BASE = 0.6       # mm que se pueden recortar abajo para dejar la base plana
APOYO_MIN = 150          # mm²; con menos, la pieza necesita soporte o balsa para no moverse
FRENTE = np.array([0.0, -1.0, 0.0])
PESOS = {"escalones_frente": 2.0, "soporte_frente": 2.0, "soporte_mm3": 1.0, "cambios": 0.5}
PESO_SIN_APOYO = 2.0     # se suma (0..2) si el apoyo es menor que APOYO_MIN


def caras_planas(malla, n=8, tolerancia=5):
    """Direcciones "arriba" que apoyan en la cama las caras planas más grandes del
    casco convexo (agrupadas si sus normales difieren < `tolerancia` grados)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    casco = malla.convex_hull
    idx = np.argsort(-casco.area_faces)[:2000]
    nor, area = casco.face_normals[idx], casco.area_faces[idx]
    g = fcluster(linkage(nor, "average", metric="cosine"), 1 - np.cos(np.radians(tolerancia)), "distance")
    grupos = sorted(((area[g == k].sum(), (nor[g == k] * area[g == k, None]).sum(0)) for k in np.unique(g)),
                    key=lambda x: -x[0])[:n]
    return np.array([-v / np.linalg.norm(v) for _, v in grupos])


def direcciones(n=200):
    """Direcciones casi uniformes en la esfera + los 6 ejes."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    tita = np.pi * (1 + 5 ** 0.5) * i
    d = np.c_[np.cos(tita) * np.sin(phi), np.sin(tita) * np.sin(phi), np.cos(phi)]
    return np.vstack([np.eye(3), -np.eye(3), d])


def medir(malla, et, k, arriba, importancia):
    tri = malla.triangles
    area = malla.area_faces
    nd = malla.face_normals @ arriba
    h = tri @ arriba                         # (caras, 3) alturas de los vértices
    piso = h.min()
    hmin, hmax = h.min(1) - piso, h.max(1) - piso
    hcen = (hmin + hmax) / 2
    # Escalón visible: |cos| alto pero no superficie plana horizontal (esa sale lisa).
    escalon = np.where(np.abs(nd) < 0.985, CAPA * np.abs(nd), 0.0)
    colgante = (nd < -np.sin(np.radians(ANGULO_SOPORTE))) & (hmin > 0.3)
    apoyo = (nd < -0.94) & (hmax < RECORTE_BASE)
    return {
        "escalones_frente": float((escalon * area * importancia).sum()),
        "soporte_frente": float((area * importancia)[colgante].sum()),
        "soporte_mm3": float((area * -nd * hcen)[colgante].sum()),
        "cambios": cambios(presencia(hmin, hmax, et, k))[0],
        "capas": int(np.ceil(hmax.max() / CAPA)),
        "apoyo_mm2": float(area[apoyo].sum()),
    }


def girar_a(arriba):
    """Matriz 4x4 que lleva `arriba` a +Z."""
    z = np.array([0.0, 0.0, 1.0])
    eje = np.cross(arriba, z)
    s, c = np.linalg.norm(eje), float(arriba @ z)
    if s < 1e-9:
        return np.eye(4) if c > 0 else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    return trimesh.transformations.rotation_matrix(np.arctan2(s, c), eje / s)


def colocar(malla, et, arriba):
    """Gira para que `arriba` sea +Z, apoya en z = 0 y recorta RECORTE_BASE mm de
    abajo para que la base quede plana. Devuelve (malla, etiquetas)."""
    g = malla.copy()
    g.apply_transform(girar_a(arriba))
    g.apply_translation([-g.bounds.mean(0)[0], -g.bounds.mean(0)[1], -g.bounds[0, 2]])
    man = a_manifold(g.vertices, g.faces)
    oid = man.to_mesh().run_original_id[0]
    recortada = man.trim_by_plane((0.0, 0.0, 1.0), RECORTE_BASE)
    if recortada.is_empty():
        return g, et
    g2, et2 = de_manifold(recortada, et, oid, -1)
    et2[et2 < 0] = np.bincount(et).argmax()   # cara de apoyo: el color principal
    g2.apply_translation([0, 0, -g2.bounds[0, 2]])
    return g2, et2


def main():
    glb = sys.argv[1]
    t0 = time.time()
    textura = trimesh.load(glb, force="mesh")
    malla = acabados.aplicar(textura, "ninguno", 100, base_plana=False)  # sin enderezar: se mide todo
    tx = textura.copy()
    tx.apply_transform(malla.metadata["transformacion"])
    centros = malla.metadata["vertices_base"][malla.faces].mean(axis=1)
    et, pal = multicolor.cuantizar(multicolor.colores_en_puntos(tx, centros), 4)
    et = multicolor.absorber_manchas(multicolor.suavizar(et, malla), malla)

    z = malla.triangles_center[:, 2]
    cabeza = z > malla.bounds[0, 2] + 0.6 * malla.extents[2]
    importancia = np.clip(malla.face_normals @ FRENTE, 0, None) * (1 + cabeza)

    planas = caras_planas(malla)
    dirs = np.vstack([direcciones(), planas])
    filas = [medir(malla, et, len(pal), d, importancia) for d in dirs]
    claves = list(PESOS)
    m = np.array([[f[c] for c in claves] for f in filas], float)
    norm = m / np.maximum(np.median(m, 0), 1e-9)
    apoyo = np.array([f["apoyo_mm2"] for f in filas])
    puntaje = norm @ np.array([PESOS[c] for c in claves]) + PESO_SIN_APOYO * np.clip(1 - apoyo / APOYO_MIN, 0, 1)
    orden = np.argsort(puntaje)
    print(f"{len(dirs)} orientaciones en {time.time() - t0:.0f} s\n")
    print(f"{'#':>3} {'arriba (x,y,z)':>20} {'puntaje':>7} {'escal.frente':>12} {'sop.frente':>10} "
          f"{'sop. cm3':>8} {'cambios':>7} {'capas':>5} {'apoyo mm2':>9}")

    def linea(etq, i):
        f = filas[i]
        print(f"{etq:>3} {str(np.round(dirs[i], 2).tolist()):>20} {puntaje[i]:7.2f} "
              f"{f['escalones_frente']:12.0f} {f['soporte_frente']:10.0f} {f['soporte_mm3'] / 1000:8.1f} "
              f"{f['cambios']:7d} {f['capas']:5d} {f['apoyo_mm2']:9.0f}")

    for r, i in enumerate(orden[:5], 1):
        linea(str(r), i)
    print("referencias:")
    for etq, d in [("Z", 2), ("Y", 1), ("-Y", 4), ("B", len(dirs) - len(planas))]:
        linea(etq, d)
    print("(B = base plana en la cama)")

    base = os.path.join(SALIDA, os.path.splitext(os.path.basename(glb))[0])
    for nombre, d in [("1", dirs[orden[0]]), ("2", dirs[orden[1]]), ("B", planas[0]), ("Z", dirs[2]), ("mY", dirs[4])]:
        g, e = colocar(malla, et, d)
        multicolor.exportar_3mf(f"{base}_orient{nombre}.3mf", g, e, pal)
        apoyo = g.area_faces[(g.face_normals[:, 2] < -0.99) & (g.triangles[:, :, 2].max(1) < 0.01)].sum()
        print(f"orient{nombre}: {g.extents.round(1).tolist()} mm, apoyo plano {apoyo:.0f} mm²")
    print(f"guardado en {SALIDA}  {time.time() - t0:.0f} s")

if __name__ == "__main__":
    main()
