"""Segmentación de la pieza en partes para imprimirlas por separado y pegarlas.

Cortes planos perpendiculares a X, Y o Z (en mm, sobre la malla de impresión,
con Z hacia arriba). En cada cara de corte se colocan uniones dentro de la
sección (con margen hasta el borde y comprobando que caben a ambos lados):

- "ninguna": corte plano, para pegar directamente.
- "espiga": cilindro macho en la parte de abajo (o de -X/-Y) y agujero con
  holgura en la otra; encaja como un puzle.
- "pasadores": agujeros en ambas partes y pasadores sueltos para imprimir
  aparte (o usar varilla/filamento del mismo diámetro).
- "imanes": alojamientos para imanes redondos en ambas caras.

Las operaciones booleanas usan manifold3d (resultados cerrados y estancos).
"""
import math
import re
import zipfile
from dataclasses import dataclass, field

import numpy as np
import trimesh

UNIONES = {
    "ninguna": "Corte plano (pegar)",
    "espiga": "Espiga macho-hembra",
    "pasadores": "Pasadores sueltos",
    "imanes": "Imanes",
}
MODOS = {
    "horizontal": "Capas horizontales (Z)",
    "vertical_x": "Rodajas verticales (X)",
    "vertical_y": "Rodajas verticales (Y)",
    "cama": "Ajustar al volumen de la impresora",
    "manual": "Manual (p. ej. z=30, x=0)",
}
EJES = {"x": 0, "y": 1, "z": 2}


@dataclass
class OpcionesUnion:
    tipo: str = "espiga"
    diametro: float = 5.0       # espiga / pasador (mm)
    largo: float = 8.0          # largo total de la espiga o del pasador (mm)
    holgura: float = 0.2        # juego radial para que encaje (mm)
    iman_diametro: float = 6.0
    iman_alto: float = 3.0
    margen: float = 1.5         # distancia mínima de la unión al borde (mm)


@dataclass
class Resultado:
    partes: list = field(default_factory=list)       # trimesh por parte
    pasadores: list = field(default_factory=list)    # trimesh por pasador
    avisos: list = field(default_factory=list)
    nombres: list = field(default_factory=list)      # opcional: nombre de cada parte


# --- Conversión -------------------------------------------------------------------

def _a_manifold(malla: trimesh.Trimesh):
    import manifold3d as m3d

    m = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(malla.vertices, np.float32),
                              tri_verts=np.asarray(malla.faces, np.uint32)))
    if m.status() != m3d.Error.NoError:
        raise RuntimeError(f"La malla no es cerrada ({m.status()}); no se puede segmentar.")
    return m


def _a_trimesh(m) -> trimesh.Trimesh:
    mesh = m.simplify(1e-3).to_mesh()  # quita triángulos sin área que dejan los cortes
    # Sin "process": la salida de manifold ya comparte vértices y es cerrada;
    # fusionar vértices cercanos puede romperla (caras degeneradas de los cortes).
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts),
                           process=False)


def _marco(eje: str):
    """Matriz 4x3 que lleva el eje de corte a Z (permutación cíclica, sin espejo) y su inversa."""
    i = EJES[eje]
    orden = [(i + 1) % 3, (i + 2) % 3, i]          # nuevo (x, y, z)
    p = np.zeros((3, 3))
    for nuevo, viejo in enumerate(orden):
        p[nuevo, viejo] = 1
    ida = np.hstack([p, np.zeros((3, 1))])
    vuelta = np.hstack([p.T, np.zeros((3, 1))])
    return ida, vuelta


# --- Planificación de cortes ----------------------------------------------------------

def planificar(malla: trimesh.Trimesh, modo: str, partes: int = 2,
               volumen=(220, 220, 250), manual: str = "") -> list[tuple[str, float]]:
    """Lista de cortes (eje, posición en mm)."""
    mn, mx = malla.bounds

    def repartir(eje, n):
        i = EJES[eje]
        return [(eje, float(mn[i] + (mx[i] - mn[i]) * k / n)) for k in range(1, n)]

    if modo == "horizontal":
        return repartir("z", max(partes, 1))
    if modo == "vertical_x":
        return repartir("x", max(partes, 1))
    if modo == "vertical_y":
        return repartir("y", max(partes, 1))
    if modo == "cama":
        cortes = []
        for eje, limite in zip("xyz", volumen):
            # 95 % del volumen para dejar margen a las uniones y a la colocación.
            n = math.ceil((mx[EJES[eje]] - mn[EJES[eje]]) / (0.95 * limite))
            cortes += repartir(eje, n)
        return cortes
    if modo == "manual":
        cortes = []
        for eje, valor in re.findall(r"([xyzXYZ])\s*=\s*(-?\d+(?:[.,]\d+)?)", manual):
            cortes.append((eje.lower(), float(valor.replace(",", "."))))
        if not cortes:
            raise ValueError("Cortes manuales no válidos. Ejemplo: z=30, x=0")
        return cortes
    raise ValueError(f"Modo de segmentación desconocido: {modo}")


# --- Colocación de uniones ----------------------------------------------------------------

def _dentro(puntos, poligonos):
    """Regla par-impar: True si el punto está dentro de la región."""
    dentro = np.zeros(len(puntos), bool)
    x, y = puntos[:, 0], puntos[:, 1]
    for p in poligonos:
        x1, y1 = p[:, 0], p[:, 1]
        x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
        cruza = ((y1[None] > y[:, None]) != (y2[None] > y[:, None])) & (
            x[:, None] < (x2 - x1)[None] * (y[:, None] - y1[None]) / ((y2 - y1)[None] + 1e-12) + x1[None])
        dentro ^= (cruza.sum(axis=1) % 2).astype(bool)
    return dentro


def _puntos_union(m_marco, pos, radio, margen, profundidad):
    """Centros (x, y) en el marco de corte donde cabe una unión de `radio`."""
    import manifold3d as m3d

    # La unión entra `profundidad` mm a cada lado: debe caber en las 3 secciones.
    seccion = m_marco.slice(pos)
    for d in (-profundidad, profundidad):
        seccion = seccion ^ m_marco.slice(pos + d)
    centros = []
    for region in seccion.decompose():
        util = region.offset(-(radio + margen), m3d.JoinType.Round)
        if util.is_empty():
            continue
        poligonos = [np.asarray(p) for p in util.to_polygons()]
        (x0, y0), (x1, y1) = np.min([p.min(0) for p in poligonos], 0), np.max([p.max(0) for p in poligonos], 0)
        gx, gy = np.meshgrid(np.linspace(x0, x1, 40), np.linspace(y0, y1, 40))
        candidatos = np.column_stack([gx.ravel(), gy.ravel()])
        candidatos = candidatos[_dentro(candidatos, poligonos)]
        if len(candidatos) == 0:
            continue
        centro = candidatos.mean(axis=0)
        a = candidatos[np.argmin(np.linalg.norm(candidatos - centro, axis=1))]
        p1 = candidatos[np.argmax(np.linalg.norm(candidatos - a, axis=1))]
        p2 = candidatos[np.argmax(np.linalg.norm(candidatos - p1, axis=1))]
        # Dos uniones si caben separadas (evita que las piezas giren); si no, una
        # centrada. Se colocan a 1/4 y 3/4 de la mayor distancia dentro de la sección.
        if np.linalg.norm(p2 - p1) >= 5 * radio:
            for t in (0.25, 0.75):
                q = p1 + (p2 - p1) * t
                centros.append(candidatos[np.argmin(np.linalg.norm(candidatos - q, axis=1))])
        else:
            centros.append(a)
    return centros


def _cilindro(x, y, z0, z1, radio):
    import manifold3d as m3d

    return m3d.Manifold.cylinder(z1 - z0, radio, radio, 48).translate((x, y, z0))


# --- Proceso --------------------------------------------------------------------------

def segmentar(malla: trimesh.Trimesh, cortes: list[tuple[str, float]],
              union: OpcionesUnion | None = None) -> Resultado:
    """Corta `malla` (mm, Z arriba) por los planos dados y añade las uniones."""
    union = union or OpcionesUnion()
    res = Resultado()
    partes = [_a_manifold(malla)]
    for eje, pos in cortes:
        ida, vuelta = _marco(eje)
        nuevas = []
        for parte in partes:
            m = parte.transform(ida)
            sup, inf = m.split_by_plane((0, 0, 1), pos)
            if sup.is_empty() or inf.is_empty() or sup.volume() < 1 or inf.volume() < 1:
                nuevas.append(parte)
                continue
            sup, inf, sueltos = _unir(m, sup, inf, pos, union, res.avisos, f"{eje}={pos:.1f}")
            res.pasadores += sueltos
            nuevas += [inf.transform(vuelta), sup.transform(vuelta)]
        partes = nuevas
    res.partes = [_a_trimesh(p) for p in partes]
    return res


def _unir(m, sup, inf, pos, u: OpcionesUnion, avisos, nombre):
    """Añade la unión elegida en el corte z=pos (marco de corte). Devuelve
    (superior, inferior, pasadores sueltos)."""
    sueltos = []
    if u.tipo == "ninguna":
        return sup, inf, sueltos
    if u.tipo == "imanes":
        radio, profundidad = u.iman_diametro / 2, u.iman_alto + 0.2
    else:
        radio, profundidad = u.diametro / 2, u.largo / 2
    centros = []
    for factor in (1.0, 0.75, 0.5):          # si no cabe, prueba más pequeña
        centros = _puntos_union(m, pos, radio * factor, u.margen, profundidad + 0.5)
        if centros:
            if factor < 1:
                avisos.append(f"Corte {nombre}: la unión no cabía; se usó un {factor:.0%} del tamaño.")
            radio *= factor
            break
    if not centros:
        avisos.append(f"Corte {nombre}: la sección es demasiado pequeña para uniones; queda plano.")
        return sup, inf, sueltos

    agujero = radio + u.holgura
    for x, y in centros:
        if u.tipo == "espiga":
            # Macho en la parte inferior (entra 1 mm en ella para fundirse).
            inf = inf + _cilindro(x, y, pos - 1.0, pos + profundidad, radio)
            sup = sup - _cilindro(x, y, pos - 0.01, pos + profundidad + 0.4, agujero)
        elif u.tipo == "pasadores":
            inf = inf - _cilindro(x, y, pos - profundidad - 0.3, pos + 0.01, agujero)
            sup = sup - _cilindro(x, y, pos - 0.01, pos + profundidad + 0.3, agujero)
            sueltos.append(_a_trimesh(_cilindro(0, 0, 0, 2 * profundidad, radio)))
        elif u.tipo == "imanes":
            inf = inf - _cilindro(x, y, pos - profundidad, pos + 0.01, agujero)
            sup = sup - _cilindro(x, y, pos - 0.01, pos + profundidad, agujero)
    return sup, inf, sueltos


# --- Salidas --------------------------------------------------------------------------

def colocar_en_cama(mallas: list, separacion: float = 8.0) -> list:
    """Copias apoyadas en z=0 y repartidas en filas sobre la cama."""
    colocadas, x, y, fila = [], 0.0, 0.0, 0.0
    ancho_fila = max(200.0, max(m.extents[0] for m in mallas) if mallas else 0)
    for m in mallas:
        m = m.copy()
        m.apply_translation(-m.bounds[0])
        if x > 0 and x + m.extents[0] > ancho_fila:
            x, y, fila = 0.0, y + fila + separacion, 0.0
        m.apply_translation([x, y, 0])
        x += m.extents[0] + separacion
        fila = max(fila, m.extents[1])
        colocadas.append(m)
    return colocadas


def vista_explosionada(partes: list, ruta: str, separacion: float = 0.25):
    """GLB con las partes separadas desde el centro y en colores distintos (Y arriba)."""
    centro = np.mean([p.centroid for p in partes], axis=0)
    escena = []
    paleta = [(230, 90, 70), (70, 150, 230), (90, 190, 110), (240, 190, 60), (170, 100, 210),
              (60, 190, 190), (230, 120, 170), (150, 150, 150)]
    tam = max(np.ptp(np.vstack([p.bounds for p in partes]), axis=0))
    for i, p in enumerate(partes):
        q = p.copy()
        d = q.centroid - centro
        if np.linalg.norm(d) > 1e-6:
            q.apply_translation(d / np.linalg.norm(d) * tam * separacion)
        q.visual.face_colors = (*paleta[i % len(paleta)], 255)
        escena.append(q)
    union = trimesh.util.concatenate(escena)
    union.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    union.export(ruta)


def exportar_zip(ruta: str, res: Resultado):
    """ZIP con un STL por parte y por pasador, listos para el laminador."""
    with zipfile.ZipFile(ruta, "w", zipfile.ZIP_DEFLATED) as z:
        for i, p in enumerate(colocar_en_cama(res.partes), 1):
            nombre = res.nombres[i - 1] if len(res.nombres) >= i else f"parte {i}"
            archivo = re.sub(r"[^a-z0-9]+", "_", nombre.lower()).strip("_")
            z.writestr(f"{i:02d}_{archivo}.stl", p.export(file_type="stl"))
        if res.pasadores:
            z.writestr(f"pasadores_x{len(res.pasadores)}.stl",
                       trimesh.util.concatenate(colocar_en_cama(res.pasadores, 4.0)).export(file_type="stl"))


def colores_de_partes(res: Resultado, original: trimesh.Trimesh, etiquetas: np.ndarray) -> list:
    """Etiqueta de color por cara de cada parte, tomada de la cara más cercana de
    la pieza pintada (las caras nuevas del corte toman el color del borde)."""
    from scipy.spatial import cKDTree

    arbol = cKDTree(original.triangles_center)
    return [etiquetas[arbol.query(p.triangles_center, workers=-1)[1]] for p in res.partes]


def exportar_3mf(ruta: str, res: Resultado, etiquetas_partes: list | None = None,
                 paleta: np.ndarray | None = None):
    """3MF con cada parte (y los pasadores) como objeto independiente ya colocado.
    Con `etiquetas_partes` y `paleta`, las caras quedan pintadas por filamento."""
    from multicolor import _codigo_pintura, escribir_paquete_3mf

    objetos = colocar_en_cama(res.partes + res.pasadores)
    nombres = [res.nombres[i - 1] if len(res.nombres) >= i else f"Parte {i}"
               for i in range(1, len(res.partes) + 1)] + \
              [f"Pasador {i}" for i in range(1, len(res.pasadores) + 1)]
    etiquetas = list(etiquetas_partes or []) + [None] * (len(objetos) - len(etiquetas_partes or []))
    recursos, items = [], []
    if paleta is not None:
        bases = "".join(f'<base name="Filamento {i + 1}" displaycolor="#{r:02X}{g:02X}{b:02X}FF"/>'
                        for i, (r, g, b) in enumerate(np.asarray(paleta).tolist()))
        recursos.append(f'<basematerials id="1000">{bases}</basematerials>')
    for i, (m, nombre, e) in enumerate(zip(objetos, nombres, etiquetas), 1):
        vertices = "".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in m.vertices)
        if e is None or paleta is None:
            triangulos = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in m.faces.tolist())
            atributos = ""
        else:
            codigos = np.array([_codigo_pintura(k + 1) for k in range(len(paleta))])[e]
            triangulos = "".join(
                f'<triangle v1="{a}" v2="{b}" v3="{c}" pid="1000" p1="{k}" paint_color="{cod}" '
                f'slic3rpe:mmu_segmentation="{cod}"/>'
                for (a, b, c), k, cod in zip(m.faces.tolist(), e.tolist(), codigos.tolist()))
            atributos = ' pid="1000" pindex="0"'
        recursos.append(f'<object id="{i}" name="{nombre}" type="model"{atributos}><mesh><vertices>'
                        f'{vertices}</vertices><triangles>{triangulos}</triangles></mesh></object>')
        items.append(f'<item objectid="{i}"/>')
    modelo = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
              'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
              'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">'
              f'<resources>{"".join(recursos)}</resources><build>{"".join(items)}</build></model>')
    escribir_paquete_3mf(ruta, modelo)
