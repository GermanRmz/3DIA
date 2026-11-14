"""Trozos sueltos (partes flotantes) en la malla generada.

Hunyuan3D a veces deja trozos separados del cuerpo. Hay dos tipos:

- Partes reales que se han desconectado: la bolita de una antena, la punta de
  una cola o un cuerno... El tallo que las une es más fino que la rejilla con la
  que se extrae la superficie y se pierde. Están cerca del cuerpo (o de otro
  trozo que sí lo está) y tienen cierto tamaño: se reconectan con un puente fino
  hasta el punto más cercano.
- Restos: motas lejanas o diminutas (sombras o restos del fondo de la foto). Se
  quitan.

Hunyuan solo quita los trozos de menos del 0,5 % de las caras, sin mirar dónde
están; con mallas de más resolución, restos visibles superan ese umbral.
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree

UNIR = 0.04         # distancia máxima (fracción del tamaño) para reconectar un trozo
TAMANO_MINIMO = 0.006  # trozos más pequeños que esto (fracción del tamaño) se quitan


def _trozos(malla):
    m = trimesh.Trimesh(malla.vertices, malla.faces, process=False)
    m.merge_vertices()
    return sorted(m.split(only_watertight=False), key=lambda c: -len(c.faces))


def _puente(a, b, radio, segmentos=12):
    return trimesh.creation.cylinder(radius=radio, segment=np.array([a, b]), sections=segmentos)


def _unir(partes):
    """Unión booleana (sólido limpio para imprimir); si alguna parte no es
    cerrada, se juntan sin más (los programas de impresión las unen igual)."""
    try:
        import manifold3d as m3d

        from segmentar import _a_manifold, _a_trimesh
        return _a_trimesh(m3d.Manifold.batch_boolean([_a_manifold(p) for p in partes], m3d.OpType.Add))
    except Exception:
        return trimesh.util.concatenate(partes)


def reparar(malla: trimesh.Trimesh, avisos: list | None = None) -> trimesh.Trimesh:
    """Reconecta las partes desprendidas cercanas y quita los restos sueltos."""
    trozos = _trozos(malla)
    if len(trozos) < 2:
        return malla
    cuerpo = trozos[0]
    tamano = float(cuerpo.extents.max())
    guardados, puentes = [cuerpo], []
    pendientes = [t for t in trozos[1:] if t.extents.max() >= TAMANO_MINIMO * tamano]
    quitados = len(trozos) - 1 - len(pendientes)
    # De cerca a lejos: un trozo puede engancharse a otro ya reconectado (bolita ->
    # trozo de tallo -> cabeza).
    while pendientes:
        puntos = np.vstack([g.vertices for g in guardados])
        normales = np.vstack([g.vertex_normals for g in guardados])
        arbol = cKDTree(puntos)
        distancias = [arbol.query(t.vertices) for t in pendientes]
        i = int(np.argmin([d.min() for d, _ in distancias]))
        d, idx = distancias[i]
        if d.min() > UNIR * tamano:
            break
        trozo = pendientes.pop(i)
        k = int(np.argmin(d))
        p, q = trozo.vertices[k], puntos[idx[k]]
        radio = float(np.clip(0.25 * trozo.extents.min(), 0.004 * tamano, 0.012 * tamano))
        # El puente entra un poco en el trozo y en el cuerpo para que queden unidos.
        hacia_trozo = trozo.centroid - p
        hacia_trozo /= max(np.linalg.norm(hacia_trozo), 1e-9)
        a = p + hacia_trozo * radio
        b = q - normales[idx[k]] * radio
        if np.linalg.norm(a - b) > 1e-6:
            puentes.append(_puente(a, b, radio))
        guardados.append(trozo)
    quitados += len(pendientes)
    unidos = len(guardados) - 1
    if unidos:
        malla = _unir(guardados + puentes)
    else:
        malla = cuerpo
    if avisos is not None and (unidos or quitados):
        avisos.append(f"Trozos sueltos: {unidos} reconectados al cuerpo, {quitados} restos quitados.")
    return malla
