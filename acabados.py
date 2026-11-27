"""Acabados de superficie para impresión 3D.

El acabado es geometría real (relieve), no color: la superficie se subdivide
hasta tener triángulos más pequeños que el detalle del patrón y cada vértice se
desplaza a lo largo de su normal según un patrón procedural medido en mm.

- Patrones 3D (martillado, piedra, anillos, madera): se evalúan directamente en
  el espacio, sin costuras.
- Patrones 2D (tejido, moleteado, estrías): proyección triplanar, es decir, se
  proyectan desde los 3 ejes y se mezclan según la orientación de la superficie.

El resultado queda en mm, con Z hacia arriba y apoyado en la cama (z = 0),
listo para el laminador.
"""
import numpy as np
import trimesh

MAX_CARAS = 2_000_000  # límite para que el STL no pase de ~100 MB


# --- Utilidades -----------------------------------------------------------------

def _tri(x):
    """Onda triangular 0..1..0 con periodo 1."""
    return 1.0 - 2.0 * np.abs(x - np.floor(x) - 0.5)


def _ruido(p, escala, semilla=0, octavas=3):
    """Ruido suave barato (suma de senos en direcciones aleatorias), ~[-1, 1]."""
    rng = np.random.default_rng(semilla)
    total = np.zeros(len(p))
    amp, frec, suma = 1.0, 1.0 / escala, 0.0
    for _ in range(octavas):
        for _ in range(4):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            total += amp * np.sin(p @ d * frec * 2 * np.pi + rng.uniform(0, 2 * np.pi))
            suma += amp
        amp *= 0.5
        frec *= 2.1
    return total / suma * 2


def _worley(p, celda, semilla=0, trozo=200_000):
    """Distancias al punto aleatorio más cercano (F1) y al segundo (F2), en mm."""
    f1 = np.empty(len(p))
    f2 = np.empty(len(p))
    vecinos = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 3)
    for i in range(0, len(p), trozo):
        q = p[i:i + trozo] / celda
        base = np.floor(q).astype(np.int64)
        c = base[:, None, :] + vecinos[None]                       # (n, 27, 3)
        h = (c[..., 0] * 73856093) ^ (c[..., 1] * 19349663) ^ (c[..., 2] * 83492791) ^ semilla
        jitter = np.stack([((h * k) >> 8) % 1024 for k in (1, 7, 13)], -1) / 1024.0
        d = np.linalg.norm(c + jitter - q[:, None, :], axis=-1)
        d.sort(axis=1)
        f1[i:i + trozo], f2[i:i + trozo] = d[:, 0] * celda, d[:, 1] * celda
    return f1, f2


def _triplanar(p, n, patron2d, escala):
    """Aplica un patrón 2D (u, v) -> altura con proyección triplanar.
    v es siempre la altura (Z), así los patrones verticales quedan verticales."""
    w = np.abs(n) ** 4
    w /= w.sum(axis=1, keepdims=True) + 1e-12
    x, y, z = p[:, 0] / escala, p[:, 1] / escala, p[:, 2] / escala
    return (w[:, 0] * patron2d(y, z) + w[:, 1] * patron2d(x, z) + w[:, 2] * patron2d(x, y))


# --- Patrones (devuelven altura en [0, 1], 1 = sobresale) --------------------------

def _tejido(u, v):
    """Punto jersey: columnas de 'V' (con un surco entre columnas); cada V son dos
    hebras gruesas que bajan hacia el centro y montan un poco sobre la fila de abajo."""
    ancho, alto = 1.0, 0.7                      # en unidades de la escala del patrón
    x = (u / ancho - np.floor(u / ancho) - 0.5) * ancho      # -0.5..0.5 físico
    fila = v / alto
    altura = np.zeros_like(u)
    for desplaz in (0.0, 1.0):                  # la hebra de la fila de arriba solapa
        y = (fila - np.floor(fila) - 0.5 - desplaz) * alto
        for lado in (-1.0, 1.0):
            # Hebra de (lado*0.36, +alto/2) a (lado*0.04, -alto/2): baja hacia el centro.
            p0 = np.array([lado * 0.36, 0.5 * alto])
            p1 = np.array([lado * 0.04, -0.5 * alto])
            d = (p1 - p0) / np.linalg.norm(p1 - p0)
            cx, cy = (p0 + p1) / 2
            largo = np.linalg.norm(p1 - p0) / 2 + 0.06
            b = (x - cx) * d[0] + (y - cy) * d[1]           # a lo largo
            a = -(x - cx) * d[1] + (y - cy) * d[0]          # a lo ancho
            q = (a / 0.17) ** 2 + (b / largo) ** 2
            bulto = np.sqrt(np.clip(1 - q, 0, 1)) * (1.0 - 0.25 * desplaz)
            altura = np.maximum(altura, bulto)
    return altura


def _moleteado(u, v):
    """Diamantes en relieve (como el mango de una herramienta)."""
    return np.minimum(_tri((u + v) / 2), _tri((u - v) / 2))


def _estrias(u, v):
    """Estrías verticales redondeadas."""
    return 0.5 + 0.5 * np.cos(2 * np.pi * u)


def _patron_martillado(p, n, escala):
    f1, _ = _worley(p, escala)
    return -np.clip(1 - (f1 / (0.75 * escala)) ** 2, 0, 1)        # hoyuelos hacia dentro


def _patron_piedra(p, n, escala):
    f1, f2 = _worley(p, escala, semilla=7)
    return np.clip((f2 - f1) / (0.25 * escala), 0, 1) ** 0.5      # celdas con juntas


def _patron_anillos(p, n, escala):
    return 0.5 + 0.5 * np.cos(2 * np.pi * p[:, 2] / escala)


def _patron_madera(p, n, escala):
    centro = p[:, :2].mean(axis=0)
    r = np.linalg.norm(p[:, :2] - centro, axis=1) + 0.6 * escala * _ruido(p * [1, 1, 0.15], 4 * escala)
    vetas = _tri(r / escala) ** 3
    return 0.8 * vetas + 0.2 * (0.5 + 0.5 * _ruido(p * [3, 3, 0.3], escala, semilla=3))


def _2d(f):
    return lambda p, n, escala: _triplanar(p, n, f, escala)


# nombre: (función, escala del patrón en mm, profundidad en mm, descripción)
ACABADOS = {
    "tejido": (_2d(_tejido), 2.5, 0.5, "Punto de lana (columnas de V)"),
    "moleteado": (_2d(_moleteado), 2.0, 0.4, "Diamantes en relieve"),
    "estrias": (_2d(_estrias), 2.0, 0.4, "Estrías verticales"),
    "anillos": (_patron_anillos, 1.5, 0.3, "Anillos horizontales"),
    "martillado": (_patron_martillado, 3.0, 0.4, "Metal martillado (hoyuelos)"),
    "piedra": (_patron_piedra, 4.0, 0.4, "Celdas tipo piedra o escamas"),
    "madera": (_patron_madera, 2.0, 0.3, "Vetas de madera"),
}


# --- Proceso --------------------------------------------------------------------

INCLINACION_MAX_BASE = 25   # grados: se endereza una base inclinada hasta este ángulo
RECORTE_BASE_MM = 0.6       # se corta esto de abajo para que la base apoye plana


def giro_base_plana(vertices: np.ndarray, inclinacion_max: float = INCLINACION_MAX_BASE) -> np.ndarray:
    """Matriz 4x4 que deja horizontal la cara plana más grande de abajo del casco
    convexo (la base de la figura). Los modelos de IA suelen dejar la base
    inclinada según la cámara de la imagen; así la figura queda derecha y la base
    apoya entera. Si no hay cara de abajo con menos de `inclinacion_max` grados,
    no gira."""
    casco = trimesh.convex.convex_hull(vertices)
    n, a = casco.face_normals, casco.area_faces
    abajo = n[:, 2] < -np.cos(np.radians(inclinacion_max))
    if not abajo.any():
        return np.eye(4)
    i = int(np.argmax(np.where(abajo, a, 0)))
    grupo = abajo & (n @ n[i] > np.cos(np.radians(5)))
    d = (n[grupo] * a[grupo, None]).sum(0)
    d /= np.linalg.norm(d)
    eje = np.cross(d, [0.0, 0.0, -1.0])
    if np.linalg.norm(eje) < 1e-9:
        return np.eye(4)
    return trimesh.transformations.rotation_matrix(np.arccos(np.clip(-d[2], -1, 1)),
                                                   eje / np.linalg.norm(eje))


def recortar_base(malla: trimesh.Trimesh, mm: float = RECORTE_BASE_MM):
    """Corta `mm` de abajo (malla apoyada en z = 0) y la vuelve a apoyar: la base
    queda plana. Devuelve (malla, recorte hecho en mm); si la malla no es cerrada
    no se corta (recorte 0)."""
    from segmentar import _a_manifold
    try:
        cortada = _a_manifold(malla).trim_by_plane((0.0, 0.0, 1.0), mm)
    except RuntimeError:
        return malla, 0.0
    if cortada.is_empty():
        return malla, 0.0
    # Sin simplificar: cada triángulo lleva luego su propio color de filamento.
    m = cortada.to_mesh()
    malla = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
    malla.apply_translation([0, 0, -mm])
    return malla, mm


def a_impresion(malla: trimesh.Trimesh, altura_mm: float, base_plana: bool = True) -> trimesh.Trimesh:
    """Escala a `altura_mm`, pone Z hacia arriba (el modelo viene con Y arriba)
    y lo apoya en la cama. Con `base_plana`, además endereza la base y recorta
    RECORTE_BASE_MM de abajo para que apoye entera (la pieza queda así de pie,
    que es lo que da menos escalones de capa en la cara y el frente).
    La matriz usada queda en metadata["transformacion"]."""
    malla = trimesh.Trimesh(malla.vertices.copy(), malla.faces.copy(), process=True)
    giro = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    if base_plana:
        giro = giro_base_plana(trimesh.transform_points(malla.vertices, giro)) @ giro
    v = trimesh.transform_points(malla.vertices, giro)
    escala = altura_mm / np.ptp(v[:, 2])
    v = v * escala
    centro = (v.min(axis=0) + v.max(axis=0)) / 2
    traslacion = trimesh.transformations.translation_matrix([-centro[0], -centro[1], -v[:, 2].min()])
    transformacion = traslacion @ np.diag([escala, escala, escala, 1.0]) @ giro
    malla.apply_transform(transformacion)
    if base_plana:
        malla, recorte = recortar_base(malla)
        transformacion = trimesh.transformations.translation_matrix([0, 0, -recorte]) @ transformacion
    malla.metadata["transformacion"] = transformacion
    return malla


def aplicar(malla: trimesh.Trimesh, acabado: str, altura_mm: float = 60,
            escala_mm: float | None = None, profundidad_mm: float | None = None,
            avisos: list | None = None, base_plana: bool = True) -> trimesh.Trimesh:
    """Devuelve la malla lista para imprimir (mm, Z arriba) con el acabado.
    En metadata quedan "transformacion" y "vertices_base" (antes del relieve)."""
    malla = a_impresion(malla, altura_mm, base_plana)
    transformacion = malla.metadata["transformacion"]
    if acabado in (None, "", "ninguno"):
        malla.metadata["vertices_base"] = malla.vertices.copy()
        return malla
    funcion, escala_def, prof_def, _ = ACABADOS[acabado]
    escala = escala_mm or escala_def
    profundidad = profundidad_mm or prof_def

    # Triángulos de ~1/9 del patrón; si salen demasiados, se agrandan. La estimación
    # por área es solo el primer intento: al partir aristas por la mitad quedan
    # entre 1/2 y 1 vez `borde` y salen hasta ~6 veces más caras. Se cuenta y se
    # repite con triángulos más grandes hasta quedar por debajo de MAX_CARAS.
    pedido = borde = escala / 9
    caras_estimadas = malla.area / (0.43 * borde ** 2)
    if caras_estimadas > MAX_CARAS:
        borde *= np.sqrt(caras_estimadas / MAX_CARAS)
    for _ in range(6):
        v, f = trimesh.remesh.subdivide_to_size(malla.vertices, malla.faces, max_edge=borde,
                                                max_iter=12)
        if len(f) <= MAX_CARAS or len(f) <= len(malla.faces):
            break
        borde *= 1.05 * np.sqrt(len(f) / MAX_CARAS)
    if borde > pedido and avisos is not None:
        if borde > escala / 4:     # menos de 4 triángulos por repetición: el patrón se pierde
            avisos.append(f"Acabado: con este tamaño el patrón de {escala:g} mm casi no se verá (el "
                          f"límite de {MAX_CARAS // 1_000_000} millones de caras obliga a triángulos de "
                          f"{borde:.2f} mm). Sube la escala del patrón a {4 * borde:.1f} mm o más.")
        else:
            avisos.append(f"Acabado: patrón pequeño para este tamaño; se usa menos detalle (triángulos "
                          f"de {borde:.2f} mm en vez de {pedido:.2f}). Sube la escala del patrón.")
    malla = trimesh.Trimesh(v, f, process=True)
    base = malla.vertices.copy()
    normales = malla.vertex_normals
    altura = funcion(malla.vertices, normales, escala)
    altura[base[:, 2] < 1e-3] = 0          # la cara de apoyo queda plana, sin relieve
    v = malla.vertices + normales * (altura * profundidad)[:, None]
    v[:, 2] = np.maximum(v[:, 2], 0.0)     # nada por debajo de la cama
    malla.vertices = v
    # Para pasar el color: posiciones antes del relieve y transformación del modelo.
    malla.metadata["transformacion"] = transformacion
    malla.metadata["vertices_base"] = base
    return malla
