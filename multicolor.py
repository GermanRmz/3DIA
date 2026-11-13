"""Color para impresoras multicolor (Bambu AMS, Prusa MMU, etc.).

La textura del modelo pintado se pasa a la malla de impresión y se reduce a N
colores de filamento (k-means). Se guarda un 3MF con cada triángulo pintado
con su filamento, en los dos formatos que leen los laminadores:

- `paint_color`: Bambu Studio y OrcaSlicer.
- `slic3rpe:mmu_segmentation`: PrusaSlicer.

Además se incluyen los colores como materiales estándar de 3MF, para que se vean
en visores genéricos.
"""
import zipfile

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _codigo_pintura(filamento: int) -> str:
    """Código de un triángulo entero pintado con el filamento 1..16 (formato de
    TriangleSelector de PrusaSlicer, que también usa Bambu Studio)."""
    if filamento == 1:
        return "4"
    if filamento == 2:
        return "8"
    return f"{filamento - 3:X}C"


def _imagen_textura(malla: trimesh.Trimesh) -> np.ndarray:
    material = malla.visual.material
    imagen = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if imagen is None:
        raise ValueError("El modelo no tiene textura de color.")
    return np.asarray(imagen.convert("RGB"))


def colores_en_puntos(malla_textura: trimesh.Trimesh, puntos: np.ndarray,
                      muestras: int = 500_000) -> np.ndarray:
    """Color RGB (uint8) de la textura en el punto de la superficie más cercano."""
    pts, caras = trimesh.sample.sample_surface(malla_textura, muestras, seed=0)
    bary = trimesh.triangles.points_to_barycentric(malla_textura.triangles[caras], pts)
    uv = (malla_textura.visual.uv[malla_textura.faces[caras]] * bary[:, :, None]).sum(axis=1)
    img = _imagen_textura(malla_textura)
    alto, ancho = img.shape[:2]
    col = np.clip(np.rint(uv[:, 0] * (ancho - 1)), 0, ancho - 1).astype(int)
    fila = np.clip(np.rint((1 - uv[:, 1]) * (alto - 1)), 0, alto - 1).astype(int)
    colores = img[fila, col]
    _, idx = cKDTree(pts).query(puntos, workers=-1)
    return colores[idx]


DISTANCIA_MINIMA = 40  # en RGB (0-255); colores más cercanos se unen


def cuantizar(colores: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a `n` colores como máximo (une los casi iguales). Devuelve (etiqueta
    0..k-1 por elemento, paleta uint8 de k colores). La etiqueta 0 es el color más
    abundante (filamento 1)."""
    datos = colores.astype(np.float32)
    # k-means en una muestra (rápido) y luego asignación de todos al centro más cercano.
    rng = np.random.default_rng(0)
    muestra = datos[rng.choice(len(datos), min(len(datos), 200_000), replace=False)]
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    cv2.setRNGSeed(0)
    _, _, centros = cv2.kmeans(muestra, n, None, criterio, 4, cv2.KMEANS_PP_CENTERS)
    # Une colores casi iguales (no tiene sentido gastar dos filamentos en ellos).
    centros = list(centros)
    while len(centros) > 1:
        c = np.array(centros)
        d = np.linalg.norm(c[:, None] - c[None], axis=-1) + np.eye(len(c)) * 1e9
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] >= DISTANCIA_MINIMA:
            break
        centros[i] = (c[i] + c[j]) / 2
        del centros[j]
    centros = np.array(centros, np.float32)
    n = len(centros)
    etiquetas = cKDTree(centros).query(datos, workers=-1)[1]
    orden = np.argsort(-np.bincount(etiquetas, minlength=n))
    remapeo = np.empty(n, np.int64)
    remapeo[orden] = np.arange(n)
    return remapeo[etiquetas], np.clip(centros[orden], 0, 255).astype(np.uint8)


def exportar_3mf(ruta: str, malla: trimesh.Trimesh, etiquetas: np.ndarray, paleta: np.ndarray):
    """3MF en mm con un triángulo = un filamento (etiqueta 0 -> filamento 1)."""
    v = malla.vertices
    f = malla.faces
    codigos = np.array([_codigo_pintura(i + 1) for i in range(len(paleta))])[etiquetas]
    vertices = "".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in v)
    triangulos = "".join(
        f'<triangle v1="{a}" v2="{b}" v3="{c}" pid="2" p1="{e}" paint_color="{k}" '
        f'slic3rpe:mmu_segmentation="{k}"/>'
        for (a, b, c), e, k in zip(f.tolist(), etiquetas.tolist(), codigos.tolist()))
    materiales = "".join(f'<base name="Filamento {i + 1}" displaycolor="#{r:02X}{g:02X}{b:02X}FF"/>'
                         for i, (r, g, b) in enumerate(paleta.tolist()))
    modelo = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">'
        '<metadata name="Application">3DIA</metadata>'
        f'<resources><basematerials id="2">{materiales}</basematerials>'
        f'<object id="1" type="model" pid="2" pindex="0"><mesh><vertices>{vertices}</vertices>'
        f'<triangles>{triangulos}</triangles></mesh></object></resources>'
        '<build><item objectid="1"/></build></model>')
    tipos = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
             '</Types>')
    relaciones = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                  '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                  '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
                  'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
                  '</Relationships>')
    with zipfile.ZipFile(ruta, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", tipos)
        z.writestr("_rels/.rels", relaciones)
        z.writestr("3D/3dmodel.model", modelo)


def pintar_impresion(malla_impresion: trimesh.Trimesh, malla_textura: trimesh.Trimesh,
                     n_colores: int, ruta_3mf: str, ruta_vista: str | None = None) -> np.ndarray:
    """Pasa el color de `malla_textura` (coordenadas del modelo) a la malla de
    impresión (creada por acabados.aplicar), lo reduce a `n_colores`, guarda el
    3MF y, si se pide, una vista previa .glb con esos colores. Devuelve la paleta."""
    if isinstance(malla_textura, trimesh.Scene):
        malla_textura = malla_textura.dump(concatenate=True)
    transformacion = malla_impresion.metadata["transformacion"]
    base = malla_impresion.metadata["vertices_base"]
    textura = malla_textura.copy()
    textura.apply_transform(transformacion)
    centros = base[malla_impresion.faces].mean(axis=1)
    etiquetas, paleta = cuantizar(colores_en_puntos(textura, centros), n_colores)
    exportar_3mf(ruta_3mf, malla_impresion, etiquetas, paleta)
    if ruta_vista:
        vista = trimesh.Trimesh(malla_impresion.vertices, malla_impresion.faces, process=False)
        vista.visual.face_colors = np.column_stack([paleta[etiquetas], np.full(len(etiquetas), 255)])
        # Los visores de .glb esperan Y arriba (el 3MF lleva Z arriba, como los laminadores).
        vista.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        vista.export(ruta_vista)
    return paleta
