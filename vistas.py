"""Vistas ortográficas del modelo con cuadrícula en mm (para la IA y para revisar).

Coordenadas de la malla de impresión: mm, Z arriba, frente hacia -Y.
- frente: mirando desde -Y (horizontal = X, vertical = Z)
- lado:   mirando desde +X (horizontal = Y, vertical = Z)
- arriba: mirando desde +Z (horizontal = X, vertical = Y)
"""
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw

import rasterizador

VISTAS = {
    # nombre: (índice eje horizontal, índice eje vertical, índice profundidad, signo profundidad)
    "frente": (0, 2, 1, 1.0),
    "lado": (1, 2, 0, -1.0),
    "arriba": (0, 1, 2, -1.0),
}
NOMBRES_EJE = "XYZ"


def _paso_cuadricula(tam):
    for paso in (1, 2, 5, 10, 20, 25, 50, 100, 200):
        if tam / paso <= 12:
            return paso
    return 500


def vista(malla: trimesh.Trimesh, nombre: str, res: int = 640, cajas=None,
          colores_caras=None) -> Image.Image:
    """Render ortográfico con sombreado, cuadrícula y escala en mm. `cajas` es una
    lista de (min, max, color, etiqueta) que se dibujan encima."""
    h, v, p, signo = VISTAS[nombre]
    lo, hi = malla.bounds
    margen = 0.08 * float(np.max(hi - lo))
    lo_v, hi_v = lo - margen, hi + margen
    escala = (res - 80) / max(hi_v[h] - lo_v[h], hi_v[v] - lo_v[v])
    ancho = int((hi_v[h] - lo_v[h]) * escala) + 80
    alto = int((hi_v[v] - lo_v[v]) * escala) + 80

    def a_pantalla(x, y):
        return 60 + (x - lo_v[h]) * escala, alto - 40 - (y - lo_v[v]) * escala

    verts = np.asarray(malla.vertices)
    px, py = a_pantalla(verts[:, h], verts[:, v])
    prof = signo * verts[:, p]
    prof = (prof - prof.min()) / max(np.ptp(prof), 1e-9)
    pos = np.stack([px / ancho * 2 - 1, py / alto * 2 - 1, prof, np.ones(len(verts))], 1)
    # El rasterizador usa el mismo convenio que Hunyuan: (x, y) en [-1, 1].
    fi, _ = rasterizador.rasterize(torch.tensor(pos, dtype=torch.float32)[None],
                                   torch.tensor(malla.faces, dtype=torch.int32), (alto, ancho))
    normal = np.zeros(3)
    normal[p] = -signo
    luz = 0.6 * normal + np.array([0.3, -0.3, 0.6])
    luz /= np.linalg.norm(luz)
    sombra = np.clip(np.abs(malla.face_normals @ luz), 0, 1) * 0.7 + 0.3
    base = np.asarray(colores_caras, float) if colores_caras is not None else \
        np.tile([205.0, 205.0, 210.0], (len(malla.faces), 1))
    f = fi.numpy().astype(np.int64)
    img = np.where((f > 0)[..., None], (base * sombra[:, None])[np.maximum(f - 1, 0)], 255)
    im = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(im)

    paso = _paso_cuadricula(max(hi_v[h] - lo_v[h], hi_v[v] - lo_v[v]))
    for eje, (a0, a1) in ((h, (lo_v[h], hi_v[h])), (v, (lo_v[v], hi_v[v]))):
        for t in np.arange(np.ceil(a0 / paso) * paso, a1, paso):
            if eje == h:
                x, _ = a_pantalla(t, 0)
                d.line([(x, 20), (x, alto - 40)], fill=(150, 170, 220), width=1)
                d.text((x - 8, alto - 34), f"{t:g}", fill=(30, 30, 120))
            else:
                _, y = a_pantalla(0, t)
                d.line([(60, y), (ancho - 20, y)], fill=(150, 170, 220), width=1)
                d.text((4, y - 6), f"{t:g}", fill=(30, 30, 120))
    d.text((ancho // 2 - 40, alto - 18), f"{NOMBRES_EJE[h]} (mm)  vista: {nombre}", fill=(0, 0, 0))
    d.text((4, 4), f"{NOMBRES_EJE[v]} (mm)", fill=(0, 0, 0))

    for caja_min, caja_max, color, etiqueta in cajas or []:
        x0, y0 = a_pantalla(caja_min[h], caja_min[v])
        x1, y1 = a_pantalla(caja_max[h], caja_max[v])
        d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], outline=color, width=3)
        d.text((min(x0, x1) + 3, min(y0, y1) + 2), etiqueta, fill=color)
    return im


def vistas(malla, cajas=None) -> dict:
    return {n: vista(malla, n, cajas=cajas) for n in VISTAS}


def mosaico(imagenes: dict) -> Image.Image:
    ims = list(imagenes.values())
    alto = max(i.height for i in ims)
    out = Image.new("RGB", (sum(i.width for i in ims), alto), "white")
    x = 0
    for i in ims:
        out.paste(i, (x, 0))
        x += i.width
    return out
