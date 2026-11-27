"""Despiece para lijar y pintar: corta la figura por "costuras escondidas", pone
pasadores y orienta cada pieza para imprimirla.

1. Costura escondida = borde de color (la línea de pegado queda en la línea de
   pintura) o pliegue hundido (queda en sombra: cuello, bajo un ala, donde el pie
   toca la base).
2. Se recorre la figura con planos de ~80 inclinaciones cada 0.75 mm; cada zona
   cerrada de cada sección se mide: qué fracción de su contorno cae en una costura.
   Con >= 60 % y >= 15 mm² es candidata.
3. Cortes LOCALES (solo se separa lo unido a esa zona) y de uno en uno, eligiendo
   el mejor: costura >= 75 %, sección >= 50 mm², piezas >= 800 mm³ y no láminas
   (volumen / sección >= 4 mm). Se premia el área "de otro color" que se quita:
       puntaje = costura * (1 + área quitada / (5 % del área total))
4. Uniones: agujero en las dos piezas y pasador suelto, para que las dos caras de
   corte puedan apoyar en la cama.
5. Orientación por pieza: si puede apoyar >= 100 mm² (o el 80 % del mejor apoyo
   posible) es obligatorio; entre esas, la de menos escalones y soporte en lo que
   se ve (frente y cabeza pesan más) y con los cortes horizontales. Las caras de
   corte no cuentan: quedan ocultas al pegar. Si una pieza apoya poco porque algo
   cruza un poco su cara de corte (< 2 mm y < 50 mm³, p. ej. puntas de
   pelo), eso se quita y se vuelve a orientar.

Medido en pruebas/LEEME.md (pruebas 4 y 5).
"""
import time

import manifold3d as m3
import numpy as np
import trimesh
from scipy.spatial import cKDTree

import segmentar

PASO = 0.75            # mm entre planos
COSTURA_MIN = 0.6      # fracción del contorno sobre una costura (para buscar candidatas)
COSTURA_CORTE = 0.75   # fracción mínima para cortar de verdad
DIST_COSTURA = 0.8     # mm: un punto del contorno "está en la costura" si está así de cerca
AREA_MIN = 15.0        # mm² de la zona cortada (para pegar y poner espiga)
VOL_MIN = 800.0        # mm³ de cada pieza, también de lo que queda (unos 9x9x9 mm)
TROZO_MIN = 20.0       # mm³: trozos menores invalidan el corte
SECCION_MIN = 50.0     # mm² de contacto para pegar con espiga
GROSOR_MIN = 4.0       # mm: volumen / sección; menos es una lámina que no se pega bien
APLANAR_MM = 2.0       # mm: lo que sobresale de una cara de corte menos que esto...
APLANAR_MM3 = 50.0     # ...y con menos volumen se quita, para que la cara apoye plana

CAPA = 0.2
ANGULO_SOPORTE = 40      # grados desde la horizontal; más plano necesita soporte


# --- Colores y costuras ----------------------------------------------------------

def puntos_borde(malla, et, paso=0.3):
    """Puntos cada `paso` mm sobre las aristas donde cambia el color."""
    p, e = malla.face_adjacency, malla.face_adjacency_edges
    e = e[et[p[:, 0]] != et[p[:, 1]]]
    a, b = malla.vertices[e[:, 0]], malla.vertices[e[:, 1]]
    n = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / paso)).astype(int)
    t = np.concatenate([np.arange(k) / k for k in n])
    i = np.repeat(np.arange(len(e)), n)
    return a[i] + (b[i] - a[i]) * t[:, None]


def puntos_hundidos(malla, iteraciones=40, umbral=0.2):
    """Vértices en pliegues hundidos (cuello, bajo un ala, donde un pie toca la
    base): al suavizar la malla se mueven hacia afuera más de `umbral` mm. Ahí la
    línea de pegado queda en sombra. 40 pasadas ~ escala de 1 mm con la malla de
    la app (aristas de ~0.16 mm)."""
    from scipy import sparse
    n = len(malla.vertices)
    e = malla.edges_unique
    a = sparse.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                          shape=(n, n)).tocsr()
    a = sparse.diags(1 / np.maximum(a.sum(1).A1, 1)) @ a
    v = malla.vertices.copy()
    for _ in range(iteraciones):
        v = a @ v
    hundido = ((v - malla.vertices) * malla.vertex_normals).sum(1) > umbral
    return malla.vertices[hundido]


def marco(normal):
    """(ida 3x4, vuelta 3x4): la rotación que lleva `normal` a +Z y su inversa."""
    z = np.array([0.0, 0.0, 1.0])
    eje = np.cross(normal, z)
    s, c = np.linalg.norm(eje), float(normal @ z)
    if s < 1e-9:
        r = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        r = trimesh.transformations.rotation_matrix(np.arctan2(s, c), eje / s)[:3, :3]
    return np.hstack([r, np.zeros((3, 1))]), np.hstack([r.T, np.zeros((3, 1))])


def normales(n=80):
    """Direcciones en media esfera (un plano y su opuesto son el mismo)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - i / n)
    tita = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(tita) * np.sin(phi), np.sin(tita) * np.sin(phi), np.cos(phi)]


def contorno(poligonos, paso=0.4):
    pts = []
    for p in poligonos:
        q = np.vstack([p, p[:1]])
        for a, b in zip(q[:-1], q[1:]):
            k = max(1, int(np.ceil(np.linalg.norm(b - a) / paso)))
            pts.append(a + (b - a) * (np.arange(k) / k)[:, None])
    return np.vstack(pts)


def a3d(xy, z, vuelta):
    return np.c_[xy, np.full(len(xy), z)] @ vuelta[:, :3].T


def punto_interior(region):
    """Un punto dentro de la zona (no su centroide: puede caer fuera)."""
    chica = region.offset(-0.3, m3.JoinType.Round)
    polis = [np.asarray(p) for p in (chica if not chica.is_empty() else region).to_polygons()]
    caja = np.vstack(polis)
    gx, gy = np.meshgrid(np.linspace(*caja[:, 0][[caja[:, 0].argmin(), caja[:, 0].argmax()]], 25),
                         np.linspace(*caja[:, 1][[caja[:, 1].argmin(), caja[:, 1].argmax()]], 25))
    c = np.c_[gx.ravel(), gy.ravel()]
    c = c[segmentar._dentro(c, polis)]
    if len(c) == 0:
        return polis[0].mean(0)
    return c[np.argmin(np.linalg.norm(c - c.mean(0), axis=1))]


def costura(poligonos, z, vuelta, arbol):
    d = arbol.query(a3d(contorno(poligonos), z, vuelta), workers=-1)[0]
    return float((d < DIST_COSTURA).mean())


def buscar(man, arbol):
    """Zonas de corte candidatas: (costura, área, normal, z, punto interior 2D)."""
    out = []
    for n in normales():
        ida, vuelta = marco(n)
        mr = man.transform(ida)
        b = mr.bounding_box()
        for z in np.arange(b[2] + 1.0, b[5] - 1.0, PASO):
            for region in mr.slice(z).decompose():
                a = region.area()
                if a < AREA_MIN:
                    continue
                polis = [np.asarray(p) for p in region.to_polygons()]
                f = costura(polis, z, vuelta, arbol)
                if f >= COSTURA_MIN:
                    out.append((f, a, n, z, punto_interior(region)))
    return out


def unicas(cands, radio=4.0):
    """Quita candidatas casi iguales (misma zona vista desde planos vecinos)."""
    cands = sorted(cands, key=lambda c: -c[0] * np.sqrt(c[1]))
    elegidas, centros = [], []
    for c in cands:
        _, vuelta = marco(c[2])
        p = a3d(c[4][None], c[3], vuelta)[0]
        if all(np.linalg.norm(p - q) > radio for q in centros):
            elegidas.append(c)
            centros.append(p)
    return elegidas


# --- Corte local --------------------------------------------------------------------

def toca(comp, z, lado, punto):
    s = comp.slice(z + 0.02 * lado)
    polis = [np.asarray(p) for p in s.to_polygons()]
    return bool(polis) and bool(segmentar._dentro(punto[None], polis)[0])


def corte_local(pieza, n, z, punto):
    """Separa de `pieza` lo que cuelga de la zona del plano (n, z) que contiene
    `punto`. Devuelve (separada, resto, lado) en el marco del plano, o motivo."""
    ida, _ = marco(n)
    mr = pieza.transform(ida)
    sup, inf = mr.split_by_plane((0.0, 0.0, 1.0), z)
    if sup.is_empty() or inf.is_empty():
        return None, "no corta"
    opciones, motivos = [], []
    for lado, este, otro in ((1, sup, inf), (-1, inf, sup)):
        comps = este.decompose()
        unidas = [c for c in comps if toca(c, z, lado, punto)]
        if not unidas:
            motivos.append("no toca")
            continue
        if min(c.volume() for c in unidas) < TROZO_MIN:
            motivos.append("trozo chico")
            continue
        sep = m3.Manifold.batch_boolean(unidas, m3.OpType.Add)
        resto = m3.Manifold.batch_boolean([otro] + [c for c in comps if not any(c is u for u in unidas)],
                                          m3.OpType.Add)
        partes = sorted(resto.decompose(), key=lambda c: -c.volume())
        if len(partes) > 1:
            # Lo que solo colgaba de la pieza (un mechón bajo el plano) va con ella.
            sep = m3.Manifold.batch_boolean([sep] + partes[1:], m3.OpType.Add)
            resto = partes[0]
            if len(sep.decompose()) > 1:
                motivos.append("el resto queda partido")
                continue
        if min(sep.volume(), resto.volume()) < VOL_MIN:
            motivos.append("pieza chica")
            continue
        opciones.append((sep.volume(), sep, resto, lado))
    if not opciones:
        return None, "/".join(sorted(set(motivos)))
    _, sep, resto, lado = min(opciones, key=lambda o: o[0])   # la pieza chica se separa
    return (sep, resto, lado), None


def seccion_contacto(sep, resto, z, lado):
    """Zona del plano donde la pieza separada se apoya en el resto (la que se pega)."""
    return sep.slice(z + 0.02 * lado) ^ resto.slice(z - 0.02 * lado)


def caras_originales(man, oid):
    """Índices de los triángulos de la figura original que tiene la pieza (las
    caras nuevas del corte y de las uniones no cuentan)."""
    me = man.to_mesh()
    ri = np.asarray(me.run_index) // 3
    fid = np.asarray(me.face_id)
    ids = [fid[ri[r]:ri[r + 1]] for r, o in enumerate(me.run_original_id) if o == oid]
    return np.unique(np.concatenate(ids)) if ids else np.zeros(0, int)


def enmascarar(ids, area, et):
    """Área que no es del color principal (lo que hay que pintar de otro color)."""
    if len(ids) == 0:
        return 0.0
    por_color = np.bincount(et[ids], area[ids])
    return float(por_color.sum() - por_color.max())



# --- Orientación ----------------------------------------------------------------

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


def girar_a(arriba):
    """Matriz 4x4 que lleva `arriba` a +Z."""
    z = np.array([0.0, 0.0, 1.0])
    eje = np.cross(arriba, z)
    s, c = np.linalg.norm(eje), float(arriba @ z)
    if s < 1e-9:
        return np.eye(4) if c > 0 else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    return trimesh.transformations.rotation_matrix(np.arctan2(s, c), eje / s)


FRENTE = np.array([0.0, -1.0, 0.0])
PESOS = {"escalones": 2.0, "soporte_vis": 2.0, "soporte_mm3": 1.0, "uniones": 1.0}
APOYO_MIN = 100.0         # mm² de apoyo exigido (o el 80 % del mejor posible, si es menor)
APOYO_POSIBLE = 30.0      # mm²: con menos en todas las orientaciones, va sobre soportes


def caras_ocultas(pieza, original, cortes):
    """Caras que no se ven montadas: dentro de la zona de algún corte (a ±5 mm del
    plano) y fuera de la superficie original."""
    centros = pieza.triangles_center
    cand = np.zeros(len(centros), bool)
    for c in cortes:
        if c.get("zona") is None:
            continue
        ida, _ = marco(c["normal"])
        local = centros @ ida[:, :3].T + ida[:, 3]
        cerca = np.abs(local[:, 2] - c["z"]) < 5.0
        if cerca.any():
            polis = [np.asarray(p) for p in c["zona"].offset(0.1, m3.JoinType.Round).to_polygons()]
            idx = np.flatnonzero(cerca)
            for k in range(0, len(idx), 4000):
                trozo = idx[k:k + 4000]
                cand[trozo[segmentar._dentro(local[trozo, :2], polis)]] = True
    if cand.any():
        # Caras de la superficie original: sus 3 vértices ya estaban en la malla.
        dist, _ = cKDTree(original.vertices).query(pieza.vertices[pieza.faces[cand]].reshape(-1, 3), workers=-1)
        nueva = (dist.reshape(-1, 3) > 1e-3).any(1)
        cand[np.flatnonzero(cand)[~nueva]] = False
    return cand


def medir(malla, arriba, importancia, normales_corte):
    tri, area = malla.triangles, malla.area_faces
    nd = malla.face_normals @ arriba
    h = tri @ arriba
    piso = h.min()
    hmin, hmax = h.min(1) - piso, h.max(1) - piso
    escalon = np.where(np.abs(nd) < 0.985, CAPA * np.abs(nd), 0.0)
    colgante = (nd < -np.sin(np.radians(ANGULO_SOPORTE))) & (hmin > 0.3)
    apoyo = (nd < -0.94) & (hmax < 0.05)
    inclinacion = sum(1 - abs(float(n @ arriba)) for n in normales_corte)
    return {
        "escalones": float((escalon * area * importancia).sum()),
        "soporte_vis": float((area * importancia)[colgante].sum()),
        "soporte_mm3": float((area * -nd * (hmin + hmax) / 2)[colgante].sum()),
        "uniones": inclinacion,
        "apoyo_mm2": float(area[apoyo].sum()),
        "altura": float(hmax.max()),
    }


def mejor_orientacion(pieza, ocultas, z_cabeza, normales_corte):
    """Devuelve (arriba, medidas, puntaje, referencia sin girar)."""
    frente = np.clip(pieza.face_normals @ FRENTE, 0, None)
    importancia = (0.3 + frente) * (1 + (pieza.triangles_center[:, 2] > z_cabeza))
    importancia[ocultas] = 0.0
    dirs = [direcciones(), caras_planas(pieza)]
    if normales_corte:
        dirs.append(np.vstack([np.array(normales_corte), -np.array(normales_corte)]))
    dirs = np.vstack(dirs)
    filas = [medir(pieza, d, importancia, normales_corte) for d in dirs]
    claves = list(PESOS)
    m = np.array([[f[c] for c in claves] for f in filas], float)
    norm = m / np.maximum(np.median(m, 0), 1e-9)
    apoyo = np.array([f["apoyo_mm2"] for f in filas])
    puntaje = norm @ np.array([PESOS[c] for c in claves])
    # Si la pieza puede apoyarse bien, es obligatorio; si no (ninguna cara plana
    # sin nada colgando debajo), la sostienen los soportes.
    lim = min(APOYO_MIN, 0.8 * apoyo.max())
    if apoyo.max() >= APOYO_POSIBLE:
        puntaje = np.where(apoyo >= lim, puntaje, np.inf)
    i = int(np.argmin(puntaje))
    return dirs[i], filas[i], float(puntaje[i]), (filas[2], float(puntaje[2]))


def colocar(pieza, arriba):
    g = pieza.copy()
    g.apply_transform(girar_a(arriba))
    g.apply_translation(-g.bounds[0])
    return g


# --- Proceso completo ---------------------------------------------------------------

def etiquetas_color(malla, textura, n_colores=4):
    """Color por cara (4 grupos, sin manchas) desde la textura; None sin textura."""
    import multicolor
    if textura is None or malla.metadata.get("transformacion") is None:
        return None
    tx = textura.copy()
    tx.apply_transform(malla.metadata["transformacion"])
    base = malla.metadata.get("vertices_base")
    centros = base[malla.faces].mean(axis=1) if base is not None else malla.triangles_center
    et, _ = multicolor.cuantizar(multicolor.colores_en_puntos(tx, centros), n_colores)
    return multicolor.absorber_manchas(multicolor.suavizar(et, malla), malla)


def _evaluar(pieza, cand, oid, area_o, et, total, arbol):
    """Prueba un corte candidato sobre `pieza`. Devuelve (puntaje, datos) o (None, motivo)."""
    f, a, n, z, p = cand
    r, motivo = corte_local(pieza, n, z, p)
    if r is None:
        return None, motivo
    sep, resto, lado = r
    ganancia = (enmascarar(caras_originales(pieza, oid), area_o, et)
                - enmascarar(caras_originales(sep, oid), area_o, et)
                - enmascarar(caras_originales(resto, oid), area_o, et))
    # Costura real: todo lo que se corta (puede incluir mechones que cruzan).
    seccion = seccion_contacto(sep, resto, z, lado)
    if seccion.area() < AREA_MIN:
        return None, "contacto chico"
    polis = [np.asarray(x) for x in seccion.to_polygons()]
    f_real = costura(polis, z, marco(n)[1], arbol) if polis else 0
    if f_real < COSTURA_CORTE:
        return None, "costura baja"
    if seccion.area() < SECCION_MIN:
        return None, "sección chica"
    if sep.volume() / seccion.area() < GROSOR_MIN:
        return None, "lámina"
    puntaje = f_real * (1 + max(ganancia, 0.0) / (0.05 * total))
    return puntaje, dict(sep=sep, resto=resto, normal=n, z=z, punto=p, lado=lado, costura=f_real,
                         area=seccion.area(), ganancia=ganancia)


def _pieza_de(piezas, n, z, p):
    ida, _ = marco(n)
    return next((i for i, pz in enumerate(piezas)
                 if toca(pz.transform(ida), z, 1, p) or toca(pz.transform(ida), z, -1, p)), None)


def cortar(malla, et, max_piezas=8, candidatas=None, avisar=None):
    """Cortes de uno en uno, el mejor cada vez. Devuelve (piezas Manifold, cortes,
    candidatas)."""
    avisar = avisar or (lambda f, txt: print(txt))
    t0 = time.time()
    arbol = cKDTree(np.vstack([puntos_borde(malla, et), puntos_hundidos(malla)]))
    man = m3.Manifold(m3.Mesh(vert_properties=malla.vertices.astype(np.float32),
                              tri_verts=malla.faces.astype(np.uint32),
                              face_id=np.arange(len(malla.faces), dtype=np.uint32)))
    if man.status() != m3.Error.NoError:
        raise RuntimeError(f"La malla no es cerrada ({man.status()}); no se puede cortar.")
    oid = man.to_mesh().run_original_id[0]
    area_o, total = malla.area_faces, malla.area
    if candidatas is None:
        avisar(0.05, "Buscando dónde cortar (unos minutos)...")
        candidatas = unicas(buscar(man, arbol))
    piezas, cortes, usadas = [man], [], set()
    # Candidata -> (pieza con la que se evaluó, resultado). Un corte solo cambia una
    # pieza: las candidatas de las demás no hace falta volver a probarlas.
    evaluadas = {}
    while len(piezas) < max_piezas:
        avisar(0.3 + 0.5 * len(cortes) / max(max_piezas - 1, 1), f"Buscando el corte {len(cortes) + 1}...")
        mejor = None
        for ci, cand in enumerate(candidatas[:120]):
            if ci in usadas:
                continue
            pi = _pieza_de(piezas, cand[2], cand[3], cand[4])
            if pi is None:
                continue
            previo = evaluadas.get(ci)
            if previo is None or previo[0] is not piezas[pi]:
                previo = (piezas[pi], _evaluar(piezas[pi], cand, oid, area_o, et, total, arbol))
                evaluadas[ci] = previo
            puntaje, datos = previo[1]
            if puntaje is not None and (mejor is None or puntaje > mejor[0]):
                mejor = (puntaje, ci, pi, datos)
        if mejor is None:
            break
        _, ci, pi, d = mejor
        usadas.add(ci)
        vuelta = marco(d["normal"])[1]
        piezas[pi] = d["resto"].transform(vuelta)
        piezas.append(d["sep"].transform(vuelta))
        cortes.append({k: v for k, v in d.items() if k not in ("sep", "resto")})
    avisar(0.8, f"{len(cortes)} cortes en {time.time() - t0:.0f} s.")
    return piezas, cortes, candidatas


def aplanar_cortes(piezas, cortes, solo, prof_max=APLANAR_MM, vol_max=APLANAR_MM3):
    """Quita lo poco que sobresale de las caras de corte de las piezas `solo` hacia
    el otro lado (puntas de pelo que cruzan el plano), para que la cara quede plana
    y pueda apoyar en la cama. Solo si es menos de `prof_max` mm y `vol_max` mm³.
    Usa las piezas anotadas por unir(). Devuelve avisos."""
    avisos = []
    for c in cortes:
        ida, vuelta = marco(c["normal"])
        z = c["z"]
        for lado, i in ((c["lado"], c.get("pieza")), (-c["lado"], c.get("resto"))):
            if i is None or i not in solo:
                continue
            pr = piezas[i].transform(ida)
            b = pr.bounding_box()
            prof = (z - b[2]) if lado > 0 else (b[5] - z)
            if prof <= 0.01 or prof > prof_max:
                continue
            fuera = pr.trim_by_plane((0.0, 0.0, -float(lado)), -lado * z)
            if fuera.volume() > vol_max:
                continue
            queda = pr.trim_by_plane((0.0, 0.0, float(lado)), lado * z)
            comps = queda.decompose()
            if len(comps) > 1:   # motas sueltas que deja el recorte
                queda = m3.Manifold.batch_boolean([k for k in comps if k.volume() >= TROZO_MIN]
                                                  or [max(comps, key=lambda k: k.volume())], m3.OpType.Add)
            piezas[i] = queda.transform(vuelta)
            if fuera.volume() > 0.5:
                avisos.append(f"Pieza {i + 1}: se quitaron {fuera.volume():.0f} mm³ que sobresalían "
                              f"{prof:.1f} mm de su cara de corte (para que apoye plana).")
    return avisos


def unir(piezas, cortes, diametro=3.0, largo=4.0, holgura=0.2):
    """Agujero a ambos lados de cada corte y pasadores sueltos. Modifica `piezas` y
    anota en cada corte sus dos piezas y su zona. Devuelve (pasadores, avisos)."""
    # Si no cabe, primero más corto y con menos pared (1.0 mm - holgura), luego más fino.
    escalera = [(diametro, largo, 1.2), (diametro, largo - 1, 1.0),
                (diametro * 5 / 6, largo - 1, 1.0), (diametro * 2 / 3, largo - 1, 1.0)]
    pasadores, avisos = [], []
    for k, c in enumerate(cortes, 1):
        ida, vuelta = marco(c["normal"])
        z, lado = c["z"], c["lado"]
        # Los cortes siguientes cambian las piezas: se buscan las dos que tocan la
        # zona de este corte, una a cada lado del plano.
        i_sep = next((i for i, pz in enumerate(piezas) if toca(pz.transform(ida), z, lado, c["punto"])), None)
        i_resto = next((i for i, pz in enumerate(piezas) if toca(pz.transform(ida), z, -lado, c["punto"])), None)
        if i_sep is None or i_resto is None or i_sep == i_resto:
            avisos.append(f"Corte {k}: no se encontraron las dos piezas; queda sin pasador.")
            continue
        c["pieza"], c["resto"] = i_sep, i_resto
        sep, resto = piezas[i_sep].transform(ida), piezas[i_resto].transform(ida)
        zona = seccion_contacto(sep, resto, z, lado)
        c["zona"] = zona
        prisma = m3.Manifold.extrude(zona.offset(0.01, m3.JoinType.Round), 20.0).translate((0, 0, z - 10.0))
        # Solapadas 0.05 mm: si solo se tocan, la sección justo en el plano sale rota.
        local = (sep.translate((0, 0, -0.05 * lado)) + resto) ^ prisma
        sup, inf = (sep, resto) if lado > 0 else (resto, sep)
        for n, (d, largo_, margen) in enumerate(escalera):
            if segmentar._puntos_union(local, z, d / 2, margen, largo_ / 2 + 0.5):
                break
        else:
            avisos.append(f"Corte {k}: no cabe ni un pasador de {escalera[-1][0]:g} mm; se pega sin él.")
            continue
        if n:
            avisos.append(f"Corte {k}: pasador más pequeño, de {d:g} x {largo_:g} mm (no cabía el normal); "
                          f"va en su propio archivo.")
        union = segmentar.OpcionesUnion(tipo="pasadores", diametro=d, largo=largo_, holgura=holgura,
                                        margen=margen)
        sup, inf, sueltos = segmentar._unir(local, sup, inf, z, union, avisos, str(k))
        pasadores += sueltos
        sep, resto = (sup, inf) if lado > 0 else (inf, sup)
        piezas[i_sep], piezas[i_resto] = sep.transform(vuelta), resto.transform(vuelta)
    return pasadores, avisos


def orientar_pieza(i, pieza, malla, cortes):
    """(pieza girada a su mejor orientación y apoyada en z=0, sus medidas)."""
    z_cabeza = malla.bounds[0, 2] + 0.6 * malla.extents[2]
    propios = [c for c in cortes if i in (c.get("pieza"), c.get("resto"))]
    ocultas = caras_ocultas(pieza, malla, propios)
    arriba, f, _, _ = mejor_orientacion(pieza, ocultas, z_cabeza, [c["normal"] for c in propios])
    return colocar(pieza, arriba), f


def orientar(piezas, partes, malla, cortes, informe):
    """Orienta cada pieza. Las que apoyan menos de APOYO_MIN se prueban aplanando sus
    caras de corte (si lo que sobresale es poco); se queda si apoya claramente mejor. Modifica `piezas`
    y `partes`. Devuelve (piezas giradas, medidas de cada una)."""
    giradas, filas = map(list, zip(*[orientar_pieza(i, p, malla, cortes) for i, p in enumerate(partes)]))
    for i, f in enumerate(filas):
        if f["apoyo_mm2"] >= APOYO_MIN:
            continue
        prueba = list(piezas)
        avisos = aplanar_cortes(prueba, cortes, {i})
        if prueba[i] is piezas[i]:
            continue                      # nada que quitar
        parte = segmentar._a_trimesh(prueba[i])
        girada, f2 = orientar_pieza(i, parte, malla, cortes)
        if f2["apoyo_mm2"] >= f["apoyo_mm2"] + APOYO_POSIBLE:   # solo si apoya claramente mejor
            piezas[i], partes[i], giradas[i], filas[i] = prueba[i], parte, girada, f2
            informe += avisos
    return giradas, filas


def dividir_para_pintar(malla, et=None, max_piezas=8, diametro=3.0, holgura=0.2, avisar=None,
                        candidatas=None):
    """Proceso completo. `malla`: malla de impresión cerrada, en mm, Z arriba y frente
    hacia -Y. `et`: color por cara (None = solo pliegues hundidos). Devuelve
    (Resultado con las piezas orientadas, líneas de informe, datos para pruebas)."""
    avisar = avisar or (lambda f, txt: print(txt))
    informe = []
    if et is None:
        et = np.zeros(len(malla.faces), int)
        informe.append("Sin textura: solo se corta por pliegues hundidos (con el .glb con color "
                       "también corta por los bordes de color).")
    piezas, cortes, candidatas = cortar(malla, et, max_piezas, candidatas, avisar)
    if not cortes:
        informe.append("No se encontró ningún corte con la unión escondida; la figura queda entera.")
    avisar(0.85, "Poniendo pasadores...")
    pasadores, avisos = unir(piezas, cortes, diametro, holgura=holgura)
    informe += avisos
    partes = [segmentar._a_trimesh(p) for p in piezas]
    avisar(0.9, "Orientando cada pieza...")
    giradas, filas = orientar(piezas, partes, malla, cortes, informe)
    nombres = [f"Pieza {i + 1}" for i in range(len(partes))]
    for k, c in enumerate(cortes, 1):
        if "pieza" in c:
            informe.append(f"Corte {k} ({nombres[c['resto']]} / {nombres[c['pieza']]}): unión escondida "
                           f"en el {c['costura']:.0%} del contorno, {c['area']:.0f} mm² para pegar.")
    sin_apoyo = [nombres[i] for i, f in enumerate(filas) if f["apoyo_mm2"] < APOYO_POSIBLE]
    if sin_apoyo:
        informe.append(f"{', '.join(sin_apoyo)}: sin cara plana para apoyar; va sobre soportes "
                       "(actívalos en el laminador).")
    res = segmentar.Resultado(partes=giradas, pasadores=pasadores, nombres=nombres)
    return res, informe, dict(cortes=cortes, candidatas=candidatas, partes=partes, filas=filas)
