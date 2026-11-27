"""Prueba 2: cupón de calibración de juntas en piezas separadas (PETG, boquilla 0.4).

Uso: python pruebas/cupon_petg.py
Genera en pruebas/salida una placa STL por tarjeta y resultados_cupon.csv:
  A giro (bloque de agujeros + ejes acostados), B bisagra (eje = filamento de
  1.75 mm, topes a unos 30 grados), C rótula (bola + copa ranurada).
h = holgura radial en mm (por lado).
"""
import csv
import os

import manifold3d as m3d
import numpy as np
import trimesh

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salida")
M = m3d.Manifold
m3d.set_min_circular_angle(4)
m3d.set_min_circular_edge_length(0.1)

DIAMETROS = [2.0, 2.5, 3.0, 4.0]
HOLGURAS_A = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
BISAGRAS = [(w, h) for w in (8.0, 10.0, 12.0) for h in (0.20, 0.30, 0.40)]
BOCAS = [0.90, 0.94, 0.97]
# 12 de las 36 combinaciones: cada boca sale 4 veces.
ROTULAS = [(d, h, BOCAS[(i + j) % 3]) for i, d in enumerate((6.0, 8.0, 10.0))
           for j, h in enumerate((0.15, 0.25, 0.35, 0.45))]


def caja(x0, y0, z0, x1, y1, z1):
    return M.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cil_y(r, y0, y1):
    """Cilindro con el eje a lo largo de Y."""
    return M.cylinder(y1 - y0, r).rotate([-90, 0, 0]).translate([0, y0, 0])


# --- Números de 7 segmentos (sin depender de fuentes) ---------------------------
SEGMENTOS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
             "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg"}


def texto(s, alto=4.0, relieve=0.6, t=0.7):
    """Texto en el plano XY, desde el origen. Devuelve (sólido, ancho)."""
    w, x, res, m = alto * 0.5, 0.0, M(), alto / 2
    seg = {"a": (0, alto - t, w, alto), "g": (0, m - t / 2, w, m + t / 2), "d": (0, 0, w, t),
           "f": (0, m, t, alto), "b": (w - t, m, w, alto), "e": (0, 0, t, m), "c": (w - t, 0, w, m)}
    for c in s:
        if c == ".":
            res += caja(x, 0, 0, x + t, t, relieve)
            x += t + 0.6
            continue
        for k in SEGMENTOS[c]:
            x0, y0, x1, y1 = seg[k]
            res += caja(x + x0, y0, 0, x + x1, y1, relieve)
        x += w + 0.7
    return res, x - (0.6 if s.endswith(".") else 0.7)


# --- A. Giro --------------------------------------------------------------------
def agujeros_a():
    """(d, h, x, y) de cada celda."""
    return [(d, h, 18 + 13 * j, 8 + 13 * i)
            for i, d in enumerate(DIAMETROS) for j, h in enumerate(HOLGURAS_A)]


def bloque_a():
    res = caja(0, 0, 0, 18 + 13 * 6 + 9, 8 + 13 * 3 + 15, 8)
    for d, h, x, y in agujeros_a():
        res -= M.cylinder(10, d / 2 + h).translate([x, y, -1])
    for i, d in enumerate(DIAMETROS):
        res += texto(f"{d:g}")[0].translate([2, 6 + 13 * i, 7.9])
    for j, h in enumerate(HOLGURAS_A):
        s, w = texto(f"{round(h * 100)}")
        res += s.translate([18 + 13 * j - w / 2, 8 + 13 * 3 + 8, 7.9])
    return res


def eje_a(d):
    """Eje acostado con una cara plana mínima (8 % del diámetro) para la cama."""
    r, base = d / 2, d * 0.42
    alto = max(base + r, 2.5)
    res = cil_y(r, 0, 16).translate([0, 0, base]) ^ caja(-9, -1, 0, 9, 20, 20)
    s, w = texto(f"{d:g}")
    return res + caja(-5, -7, 0, 5, 0.5, alto) + s.translate([-w / 2, -5.5, alto - 0.1])


# --- B. Bisagra -----------------------------------------------------------------
FIL = 1.75
R_NUDO = 2.8
T_HOJA = 4.3                                      # mayor que el nudo, para que haya tope
TOPE = float(T_HOJA * np.tan(np.radians(15)))     # las hojas se tocan a 30 grados
L_HOJA = 14.0


def hojas_b(w, h):
    """Dos hojas en posición montada; eje en el origen, a lo largo de Y."""
    k = (w - 2 * h) / 3
    y1, y2, y3, y4 = k, k + h, 2 * k + h, 2 * k + 2 * h
    a = caja(-L_HOJA, 0, -T_HOJA, -TOPE, w, T_HOJA)
    b = caja(TOPE, 0, -T_HOJA, L_HOJA, w, T_HOJA)
    libre = R_NUDO + h
    a = a - cil_y(libre, y1, y4) + cil_y(R_NUDO, 0, y1) + cil_y(R_NUDO, y4, w)
    b = b - cil_y(libre, -1, y2) - cil_y(libre, y3, w + 1) + cil_y(R_NUDO, y2, y3)
    a -= cil_y(FIL / 2 + 0.05, -1, w + 1)         # ajuste justo en la hoja padre
    b -= cil_y(FIL / 2 + h, -1, w + 1)            # la holgura que se ensaya
    return a, b


def hoja_para_imprimir(s, n, es_padre, w):
    """De pie, con los nudos arriba (sin soportes) y el número en la cara."""
    s = s.rotate([0, -90 if es_padre else 90, 0])
    s = s.translate([T_HOJA, 0, -s.bounding_box()[2]])
    num, an = texto(str(n))
    num = num.rotate([90, 0, 90]).translate([2 * T_HOJA - 0.1, (w - an) / 2, 3])
    return s + num


# --- C. Rótula --------------------------------------------------------------------
PARED = 1.6


def copa_c(d, h, m):
    """Copa centrada en el origen (sin la base). Devuelve (sólido, z inferior)."""
    r_bola = d / 2
    ri, boca = r_bola + h, m * r_bola
    zb = float(np.sqrt(ri ** 2 - boca ** 2))      # altura de la boca
    z0, z1 = -ri - PARED, zb + 0.8
    s = M.cylinder(z1 - z0, ri + PARED).translate([0, 0, z0]) - M.sphere(ri)
    s -= M.cylinder(0.82, boca, boca + 0.82).translate([0, 0, zb - 0.01])   # alivio a 45 grados
    s -= caja(-ri - 5, -0.4, 0, ri + 5, 0.4, z1 + 1)                        # dos ranuras
    return s, z0


def copa_con_base(d, h, m, n):
    s, z0 = copa_c(d, h, m)
    r = d / 2 + h + PARED
    s = s.translate([0, 0, -z0]) + caja(-r - 1, -r - 8, 0, r + 1, r + 1, 1.6)
    num, an = texto(str(n))
    return s + num.translate([-an / 2, -r - 6.5, 1.5])


def bola_c(d):
    """Bola y cuello acostados con una cara plana pequeña (4 % del diámetro)."""
    r = d / 2
    z, largo = r * 0.92, r + 3.5
    s = (M.sphere(r) + cil_y(0.5 * r, -largo, 0)).translate([0, 0, z]) ^ caja(-20, -30, 0, 20, 20, 30)
    alto = z + 0.5 * r
    num, an = texto(f"{d:g}")
    return s + caja(-5, -largo - 8, 0, 5, -largo + 0.5, alto) \
        + num.translate([-an / 2, -largo - 6, alto - 0.1])


# --- Placas, comprobaciones y exportación -------------------------------------------
def colocar(piezas, ancho=160, sep=5):
    """Coloca las piezas en filas."""
    x = y = fila = 0.0
    res = []
    for p in piezas:
        b = p.bounding_box()
        dx, dy = b[3] - b[0], b[4] - b[1]
        if x and x + dx > ancho:
            x, y, fila = 0.0, y + fila + sep, 0.0
        res.append(p.translate([x - b[0], y - b[1], -b[2]]))
        x, fila = x + dx + sep, max(fila, dy)
    return res


def comprobar():
    """Comprueba en 3D lo que cada tarjeta promete. Devuelve la lista de fallos."""
    fallos = []
    for d, h, m in ROTULAS:
        copa, _ = copa_c(d, h, m)
        bola = M.sphere(d / 2)
        if (copa ^ bola).volume() > 1e-6:
            fallos.append(f"rotula {d}/{h}/{m}: la bola toca la copa en reposo")
        if max((copa ^ bola.translate([0, 0, z])).volume() for z in np.linspace(0, d, 40)) < 1e-4:
            fallos.append(f"rotula {d}/{h}/{m}: sin retencion")
    for w, h in BISAGRAS:
        a, b = hojas_b(w, h)
        for x in (a, b):
            if len(x.decompose()) != 1:
                fallos.append(f"bisagra {w}/{h}: hoja en varios trozos")
        for ang in range(-28, 29, 4):
            v = (a ^ b.rotate([0, ang, 0])).volume()
            if v > 1e-6:
                fallos.append(f"bisagra {w}/{h}: choca a {ang} grados ({v:.3f} mm3)")
        for ang in (-33, 33):
            if (a ^ b.rotate([0, ang, 0])).volume() < 1e-3:
                fallos.append(f"bisagra {w}/{h}: sin tope a {ang} grados")
    return fallos


def malla(s):
    x = s.to_mesh()
    return trimesh.Trimesh(np.asarray(x.vert_properties)[:, :3], np.asarray(x.tri_verts), process=False)


def main():
    os.makedirs(SALIDA, exist_ok=True)
    fallos = comprobar()
    if fallos:
        raise SystemExit("\n".join(fallos))
    print("Comprobaciones de giro, tope y retencion: sin fallos")

    b = []
    for n, (w, h) in enumerate(BISAGRAS, 1):
        ha, hb = hojas_b(w, h)
        b += [hoja_para_imprimir(ha, n, True, w), hoja_para_imprimir(hb, n, False, w)]
    placas = {
        "2A_giro": [bloque_a()] + [eje_a(d) for d in DIAMETROS for _ in range(2)],
        "2B_bisagra": b,
        "2C_rotula": [copa_con_base(*c, n) for n, c in enumerate(ROTULAS, 1)]
        + [bola_c(d) for d in (6.0, 8.0, 10.0) for _ in range(4)],
    }
    for nombre, piezas in placas.items():
        for i, p in enumerate(piezas):
            if len(p.decompose()) != 1:
                raise SystemExit(f"{nombre}: la pieza {i} tiene {len(p.decompose())} trozos")
        placa = malla(M.compose(colocar(piezas)))
        dim = np.ptp(placa.vertices, axis=0)
        assert placa.is_watertight and (dim < 175).all(), (nombre, dim)
        placa.export(os.path.join(SALIDA, nombre + ".stl"))
        print(f"{nombre}: {len(piezas)} piezas, {dim.round(1)} mm, {placa.volume * 1.27e-3:.0f} g si fuera macizo")

    ruta = os.path.join(SALIDA, "resultados_cupon.csv")
    if os.path.exists(ruta):
        print("Se conserva resultados_cupon.csv (puede tener tus medidas)")
        return
    with open(ruta, "w", newline="", encoding="utf-8-sig") as f:
        o = csv.writer(f)
        o.writerow(["tarjeta", "numero", "diametro_o_ancho", "holgura_h", "boca", "entra", "gira_libre",
                    "juego_flojo_bien_duro", "se_sale_sola", "angulo_max", "ciclos", "fallo", "notas"])
        for n, (d, h, _, _) in enumerate(agujeros_a(), 1):
            o.writerow(["A", n, d, h, ""] + [""] * 8)
        for n, (w, h) in enumerate(BISAGRAS, 1):
            o.writerow(["B", n, w, h, ""] + [""] * 8)
        for n, (d, h, m) in enumerate(ROTULAS, 1):
            o.writerow(["C", n, d, h, m] + [""] * 8)


if __name__ == "__main__":
    main()
