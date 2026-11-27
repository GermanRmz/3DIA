"""Prueba 1: placa 3MF de 40x40x2 mm con los 4 filamentos e islas de 0.2, 0.4, 0.8 y 1.2 mm.

Uso:  python pruebas/cubo_color.py            -> pruebas/salida/1_placa_4_colores.3mf
      python pruebas/cubo_color.py modelo.glb -> además <modelo>_4colores.3mf (10 cm)
"""
import os
import sys

import numpy as np
import trimesh

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import multicolor  # noqa: E402

SALIDA = os.path.join(RAIZ, "pruebas", "salida")
PALETA = np.array([[255, 255, 255], [220, 40, 40], [40, 90, 220], [30, 30, 30]], np.uint8)
ISLAS = [0.2, 0.4, 0.8, 1.2]
LADO, ALTO = 40.0, 2.0   # bajo: cada capa con 4 colores cuesta varios cambios


def cubo():
    """Placa de 40x40x2: lados = filamentos 1-4; arriba, islas sobre filamento 1."""
    xs, ys, islas = {0.0, LADO}, {0.0, LADO}, []
    for c, lado in enumerate(ISLAS):
        xs |= {8.0 + 8 * c - lado / 2, 8.0 + 8 * c + lado / 2}
        for f in range(3):
            ys |= {10.0 + 10 * f - lado / 2, 10.0 + 10 * f + lado / 2}
            islas.append((8.0 + 8 * c, 10.0 + 10 * f, lado / 2, f + 1))
    xs, ys = sorted(xs), sorted(ys)
    tri, color = [], []

    def cuad(a, b, c, d, k):
        tri.extend([[a, b, c], [a, c, d]])
        color.extend([k, k])

    for x0, x1 in zip(xs, xs[1:]):
        cuad((x0, 0, 0), (x1, 0, 0), (x1, 0, ALTO), (x0, 0, ALTO), 0)
        cuad((x1, LADO, 0), (x0, LADO, 0), (x0, LADO, ALTO), (x1, LADO, ALTO), 1)
        for y0, y1 in zip(ys, ys[1:]):
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            k = next((i[3] for i in islas if abs(mx - i[0]) < i[2] and abs(my - i[1]) < i[2]), 0)
            cuad((x0, y0, ALTO), (x1, y0, ALTO), (x1, y1, ALTO), (x0, y1, ALTO), k)
            cuad((x0, y1, 0), (x1, y1, 0), (x1, y0, 0), (x0, y0, 0), 0)
    for y0, y1 in zip(ys, ys[1:]):
        cuad((0, y1, 0), (0, y0, 0), (0, y0, ALTO), (0, y1, ALTO), 2)
        cuad((LADO, y0, 0), (LADO, y1, 0), (LADO, y1, ALTO), (LADO, y0, ALTO), 3)
    pts, idx = np.unique(np.array(tri, float).reshape(-1, 3), axis=0, return_inverse=True)
    return trimesh.Trimesh(pts, idx.reshape(-1, 3), process=False), np.array(color)


def main():
    os.makedirs(SALIDA, exist_ok=True)
    malla, color = cubo()
    assert malla.is_watertight and np.isclose(malla.volume, LADO * LADO * ALTO)
    ruta = os.path.join(SALIDA, "1_placa_4_colores.3mf")
    multicolor.exportar_3mf(ruta, malla, color, PALETA)
    print(ruta, "triangulos por filamento:", np.bincount(color))
    for glb in sys.argv[1:]:
        import acabados
        textura = trimesh.load(glb, force="mesh")
        pieza = acabados.aplicar(textura, "ninguno", 100)
        base = os.path.join(SALIDA, os.path.splitext(os.path.basename(glb))[0])
        paleta = multicolor.pintar_impresion(pieza, textura, 4, base + "_4colores.3mf",
                                             base + "_4colores_vista.glb")
        print(base, len(pieza.faces), "caras; cerrada:", pieza.is_watertight,
              "medidas:", pieza.extents.round(1), "paleta:", paleta.tolist())


if __name__ == "__main__":
    main()
