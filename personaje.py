"""División de un personaje en partes anatómicas (cabeza, torso, brazos, piernas)
con uniones de espiga o de rótula (bola), para imprimirlo por piezas.

Trabaja sobre la malla de impresión (mm, Z arriba, frente hacia -Y), con el
personaje de pie y de frente. Los puntos de corte se detectan analizando las
secciones horizontales de la figura:

- cuello: el estrechamiento más marcado entre la cabeza y los hombros;
- entrepierna: la altura más alta en la que las dos piernas siguen separadas;
- hombros: la altura más alta en la que los brazos están separados del torso
  (axila); el corte vertical va en el costado del torso.
"""
from dataclasses import dataclass, field

import numpy as np
import trimesh

from segmentar import _a_manifold, _a_trimesh, _marco

# Plantillas: grupos de las 6 regiones base que forman cada parte.
REGIONES = ["cabeza", "torso", "brazo_izq", "brazo_der", "pierna_izq", "pierna_der"]
PLANTILLAS = {
    "6": ("6 partes: cabeza, torso, brazos y piernas",
          [["cabeza"], ["torso"], ["brazo_izq"], ["brazo_der"], ["pierna_izq"], ["pierna_der"]]),
    "5": ("5 partes: cabeza, torso, brazos y piernas juntas",
          [["cabeza"], ["torso"], ["brazo_izq"], ["brazo_der"], ["pierna_izq", "pierna_der"]]),
    "4": ("4 partes: cabeza, cuerpo con piernas y brazos",
          [["cabeza"], ["torso", "pierna_izq", "pierna_der"], ["brazo_izq"], ["brazo_der"]]),
    "3": ("3 partes: cabeza, cuerpo con brazos y piernas",
          [["cabeza"], ["torso", "brazo_izq", "brazo_der"], ["pierna_izq", "pierna_der"]]),
    "2c": ("2 partes: cabeza y cuerpo",
           [["cabeza"], ["torso", "brazo_izq", "brazo_der", "pierna_izq", "pierna_der"]]),
    "2p": ("2 partes: parte de arriba y piernas",
           [["cabeza", "torso", "brazo_izq", "brazo_der"], ["pierna_izq", "pierna_der"]]),
}
BRAZO_MINIMO = 0.035  # fracción mínima del volumen total para aceptar un brazo separado
ANGULO_HOMBRO = 20  # grados respecto a la vertical del corte del hombro

UNIONES = {"espiga": "Espiga (pin y agujero)", "bola": "Rótula de bola (articulada)",
           "ninguna": "Sin unión (pegar)"}


@dataclass
class OpcionesUnion:
    tipo: str = "espiga"
    diametro: float = 0.0      # 0 = automático según el tamaño del corte
    holgura: float = 0.2       # juego en mm (la rótula encaja a presión con un poco menos)


@dataclass
class Puntos:
    cuello: float | None = None
    entrepierna: float | None = None
    axila: float | None = None
    hombro_izq: float | None = None     # x del costado del torso en la axila (lado izquierdo)
    hombro_der: float | None = None
    brazos: dict = field(default_factory=dict)  # lado -> (x interior del brazo, punto del brazo)
    centro_x: float = 0.0


@dataclass
class Division:
    partes: list = field(default_factory=list)    # (nombre, trimesh)
    puntos: Puntos | None = None
    avisos: list = field(default_factory=list)


# --- Detección de articulaciones -------------------------------------------------

def _secciones(m, zmin, zmax, n=140):
    """Por cada altura: lista de componentes (área, xmin, xmax, x centro, y centro)."""
    zs = np.linspace(zmin, zmax, n + 2)[1:-1]
    datos = []
    for z in zs:
        comps = []
        for c in m.slice(float(z)).decompose():
            a = c.area()
            if a <= 0:
                continue
            (x0, y0), (x1, y1) = np.asarray(c.bounds()).reshape(2, 2)
            comps.append((a, x0, x1, (x0 + x1) / 2, (y0 + y1) / 2))
        datos.append(comps)
    return zs, datos


def detectar(malla: trimesh.Trimesh) -> Puntos:
    m = _a_manifold(malla)
    (x0, _, z0), (x1, _, z1) = malla.bounds
    alto = z1 - z0
    zs, sec = _secciones(m, z0, z1)
    p = Puntos(centro_x=float((x0 + x1) / 2))
    area = np.array([sum(c[0] for c in s) for s in sec])
    suave = np.convolve(area, np.ones(5) / 5, mode="same")

    # Entrepierna: subiendo desde abajo, mientras haya una pierna a cada lado
    # del centro (componentes cercanos al centro, no las manos de los costados).
    ancho = x1 - x0
    for z, s in zip(zs, sec):
        if z > z0 + 0.65 * alto:
            break
        grandes = [c for c in s if c[0] > 0.12 * max(cc[0] for cc in s)] if s else []
        izq = [c for c in grandes if c[3] < p.centro_x and c[2] < p.centro_x + 0.05 * ancho
               and abs(c[3] - p.centro_x) < 0.3 * ancho]
        der = [c for c in grandes if c[3] > p.centro_x and c[1] > p.centro_x - 0.05 * ancho
               and abs(c[3] - p.centro_x) < 0.3 * ancho]
        if izq and der:
            p.entrepierna = float(z)
        elif p.entrepierna is not None:
            break

    # Cuello: mínimo de área entre la mitad y el 95 % de la altura, con más
    # área por encima (la cabeza) que en el propio cuello.
    base = (p.entrepierna or z0 + 0.3 * alto)
    ventana = (zs > max(base, z0 + 0.45 * alto)) & (zs < z0 + 0.95 * alto)
    if ventana.any():
        idx = np.nonzero(ventana)[0]
        i = idx[np.argmin(suave[idx])]
        if suave[i:].max() > 1.5 * suave[i] and suave[:i + 1][zs[:i + 1] > base].max(initial=0) > 1.2 * suave[i]:
            p.cuello = float(zs[i])

    # Hombros: subiendo desde la entrepierna, el primer tramo continuo de alturas
    # con un brazo suelto a cada lado del torso; su parte más alta es la axila.
    techo = p.cuello or z0 + 0.8 * alto
    en_tramo = False
    libres = {"izq": [], "der": []}
    for z, s in zip(zs, sec):
        if z <= base or z >= techo:
            continue
        separados = False
        centrales = [c for c in s if c[1] <= p.centro_x <= c[2]]
        if centrales and len(s) >= 3:
            torso = max(centrales, key=lambda c: c[0])
            izq = [c for c in s if c[2] < torso[1]]
            der = [c for c in s if c[1] > torso[2]]
            if izq and der:
                separados = True
                margen = max(0.5, 0.01 * ancho)
                p.axila = float(z)
                p.hombro_izq, p.hombro_der = float(torso[1] - margen), float(torso[2] + margen)
                # El trozo de brazo más cercano al torso en cada lado.
                libres["izq"].append((z, max(izq, key=lambda c: c[2])))
                libres["der"].append((z, min(der, key=lambda c: c[1])))
        if en_tramo and not separados:
            break
        en_tramo = en_tramo or separados
    for lado, filas in libres.items():
        if not filas:
            continue
        # Borde interior del brazo (lo más cerca del centro que llega mientras cuelga
        # suelto) y un punto del brazo a media altura del tramo libre.
        interior = max(c[2] for _, c in filas) if lado == "izq" else min(c[1] for _, c in filas)
        z_medio, c_medio = filas[len(filas) // 2]
        p.brazos[lado] = (float(interior), (float(c_medio[3]), float(c_medio[4]), float(z_medio)))
    return p


# --- Separación -------------------------------------------------------------------

def _region(cuerpo, zona, elegir):
    """Separa del cuerpo los trozos de cuerpo∩zona que `elegir` acepte. El resto
    se obtiene restando lo elegido (así no se cortan ni se vuelven a pegar los
    trozos que no se separan)."""
    import manifold3d as m3d

    trozos = sorted((cuerpo ^ zona).decompose(), key=lambda t: -t.volume())
    elegidos = elegir(trozos)
    if not elegidos:
        return cuerpo, None
    pieza = m3d.Manifold.batch_boolean(elegidos, m3d.OpType.Add)
    return cuerpo - pieza, pieza


def _semiespacio_plano(punto, normal, grande=1e4):
    """Manifold que ocupa {p : normal·(p - punto) <= 0}."""
    import manifold3d as m3d

    ida, vuelta = _marco_plano(punto, normal)
    cubo = m3d.Manifold.cube((2 * grande, 2 * grande, grande)).translate((-grande, -grande, -grande))
    return cubo.transform(vuelta)


def _marco_plano(punto, normal):
    """Matrices 4x3 (ida, vuelta) que llevan el plano (punto, normal) a z=0 con
    la normal hacia +z."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    aux = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(aux, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    r = np.vstack([u, v, n])                 # filas: nuevos ejes
    t = -r @ np.asarray(punto, float)
    ida = np.hstack([r, t[:, None]])
    vuelta = np.hstack([r.T, np.asarray(punto, float)[:, None]])
    return ida, vuelta


def _elegir_por_punto(punto):
    """Elige el trozo más cercano a `punto` (un punto conocido de la pieza)."""
    import manifold3d as m3d

    def elegir(trozos):
        if not trozos:
            return []
        marca = m3d.Manifold.sphere(1.0, 16).translate(punto)
        con_punto = [t for t in trozos if not (t ^ marca).is_empty()]
        return con_punto[:1] or [max(trozos, key=lambda t: t.bounding_box()[5])]
    return elegir


def _caja(xmin=-1e4, xmax=1e4, zmin=-1e4, zmax=1e4):
    import manifold3d as m3d

    return m3d.Manifold.cube((xmax - xmin, 2e4, zmax - zmin)).translate((xmin, -1e4, zmin))


def dividir(malla: trimesh.Trimesh, plantilla: str = "6",
            union: OpcionesUnion | None = None) -> Division:
    """Divide el personaje según la plantilla y añade las uniones."""
    import manifold3d as m3d

    union = union or OpcionesUnion()
    grupos = PLANTILLAS[plantilla][1]
    del_torso = next(g for g in grupos if "torso" in g)
    separar = {r for r in REGIONES if r not in del_torso}
    res = Division(puntos=detectar(malla))
    p = res.puntos
    cuerpo = _a_manifold(malla)
    volumen_total = cuerpo.volume()
    regiones = {}
    cortes = []     # (región hija, eje, posición, lado de la hija)

    if "cabeza" in separar:
        if p.cuello is None:
            res.avisos.append("No se encontró el cuello; la cabeza queda unida.")
        else:
            cuerpo, cabeza = _region(cuerpo, _caja(zmin=p.cuello), lambda ts: ts[:1])
            regiones["cabeza"] = cabeza
            cortes.append(("cabeza", "z", p.cuello, +1))
    for lado, x_torso, signo in (("izq", p.hombro_izq, -1), ("der", p.hombro_der, +1)):
        nombre = f"brazo_{lado}"
        if nombre not in separar:
            continue
        if x_torso is None or lado not in p.brazos:
            res.avisos.append("Los brazos están pegados al cuerpo; no se pueden separar.")
            break
        x_interior, punto = p.brazos[lado]
        # Corte de hombro: plano casi vertical (inclinado ANGULO_HOMBRO hacia fuera)
        # que sube desde la axila; por debajo de la axila, plano vertical por el
        # borde interior del brazo (así no se corta el brazo a lo largo).
        a = np.array([x_torso, 0.0, p.axila])
        normal = np.array([-signo, 0.0, np.tan(np.radians(ANGULO_HOMBRO))])   # hacia el torso
        normal /= np.linalg.norm(normal)
        zona = _semiespacio_plano(a, normal) ^ (
            _caja(xmax=x_interior + 0.5) if signo < 0 else _caja(xmin=x_interior - 0.5))
        brazo_elegido = _elegir_por_punto(punto)
        cuerpo_sin, brazo = _region(cuerpo, zona, brazo_elegido)
        # Si solo cuelga suelta la mano o el antebrazo (brazo pegado al cuerpo casi
        # entero), el trozo es pequeño: mejor no cortar que cortar la mano.
        if brazo is None or brazo.volume() < BRAZO_MINIMO * volumen_total:
            res.avisos.append(f"El {nombre.replace('_', ' ')} está pegado al cuerpo casi entero; "
                              "se deja unido al torso (para separarlo, el personaje debe tener "
                              "los brazos separados del cuerpo).")
            continue
        cuerpo = cuerpo_sin
        regiones[nombre] = brazo
        cortes.append((nombre, ("plano", a, normal), None, None))
    piernas = [r for r in ("pierna_izq", "pierna_der") if r in separar]
    if piernas:
        if p.entrepierna is None:
            res.avisos.append("No se encontró la entrepierna; las piernas quedan unidas.")
        else:
            corte = p.entrepierna
            # Solo en la franja del torso: las manos que cuelgan más abajo no se tocan.
            ancho = malla.extents[0]
            xi = p.hombro_izq if p.hombro_izq is not None else p.centro_x - 0.3 * ancho
            xd = p.hombro_der if p.hombro_der is not None else p.centro_x + 0.3 * ancho
            cuerpo, ambas = _region(cuerpo, _caja(xmin=xi, xmax=xd, zmax=corte), lambda ts: ts[:2])
            if ambas is not None:
                juntas = any("pierna_izq" in g and "pierna_der" in g for g in grupos)
                if juntas:
                    regiones["pierna_izq"] = ambas
                else:
                    izq = ambas ^ _caja(xmax=p.centro_x)
                    der = ambas ^ _caja(xmin=p.centro_x)
                    regiones["pierna_izq"], regiones["pierna_der"] = izq, der
                cortes.append(("piernas", "z", corte, -1))
    regiones["torso"] = cuerpo

    # Uniones (la pieza hija lleva el macho; el torso, el hueco).
    for hija, eje, pos, signo in cortes:
        if isinstance(eje, tuple):           # plano inclinado (punto, normal hacia el torso)
            ida, vuelta = _marco_plano(eje[1], eje[2])
            pos, signo = 0.0, -1
        else:
            ida, vuelta = _marco(eje)
        for pieza in (("pierna_izq", "pierna_der") if hija == "piernas" else (hija,)):
            if pieza in regiones:
                regiones[pieza], regiones["torso"] = _unir(
                    regiones[pieza], regiones["torso"], ida, vuelta, pos, signo, union,
                    res.avisos, pieza)

    # Agrupa según la plantilla.
    for grupo in grupos:
        piezas = [regiones[r] for r in grupo if r in regiones]
        if not piezas:
            continue
        pieza = m3d.Manifold.batch_boolean(piezas, m3d.OpType.Add)
        # Fuera restos sin volumen que dejan los cortes tangentes.
        trozos = pieza.decompose()
        mayor = max(t.volume() for t in trozos)
        pieza = m3d.Manifold.batch_boolean([t for t in trozos if t.volume() > 0.01 * mayor],
                                           m3d.OpType.Add)
        res.partes.append((_nombre(grupo), _a_trimesh(pieza)))
    return res


def _nombre(grupo):
    nombres = []
    for r in grupo:
        if r.endswith("_der") and r.replace("_der", "_izq") in grupo:
            continue
        if r.endswith("_izq") and r.replace("_izq", "_der") in grupo:
            nombres.append(r.split("_")[0] + "s")
        else:
            nombres.append(r.replace("_", " "))
    return " + ".join(nombres)


# --- Uniones ------------------------------------------------------------------------

def _centro_y_radio(hija_marco, pos, signo):
    """Centro (x, y) de la sección de la hija junto al corte y radio máximo
    que cabe dentro (polo de inaccesibilidad aproximado)."""
    import manifold3d as m3d

    seccion = hija_marco.slice(pos + 0.05 * signo)
    if seccion.is_empty():
        return None, 0.0
    seccion = max(seccion.decompose(), key=lambda c: c.area())
    lo, hi = 0.0, 50.0
    for _ in range(12):
        mid = (lo + hi) / 2
        if seccion.offset(-mid, m3d.JoinType.Round).is_empty():
            hi = mid
        else:
            lo = mid
    nucleo = seccion.offset(-0.8 * lo, m3d.JoinType.Round)
    puntos = np.vstack([np.asarray(q) for q in nucleo.to_polygons()])
    return puntos.mean(axis=0), lo


def _unir(hija, padre, ida, vuelta, pos, signo, u: OpcionesUnion, avisos, nombre):
    """Une la hija (al lado `signo` del plano z=pos en el marco `ida`) con el padre."""
    import manifold3d as m3d

    if u.tipo == "ninguna":
        return hija, padre
    h, pa = hija.transform(ida), padre.transform(ida)
    centro, radio_max = _centro_y_radio(h, pos, signo)
    if centro is None or radio_max < 1.5:
        avisos.append(f"{nombre}: el corte es demasiado fino para una unión.")
        return hija, padre
    r = (u.diametro / 2) if u.diametro > 0 else min(0.45 * radio_max, 8.0)
    r = min(r, 0.6 * radio_max)
    x, y = centro
    s = -signo   # dirección desde la hija hacia el padre

    def cilindro(z0, z1, radio):
        a, b = min(z0, z1), max(z0, z1)
        return m3d.Manifold.cylinder(b - a, radio, radio, 48).translate((x, y, a))

    def esfera(z, radio):
        return m3d.Manifold.sphere(radio, 48).translate((x, y, z))

    if u.tipo == "espiga":
        largo = max(2.5 * r, 4.0)
        h = h + cilindro(pos - s * 1.0, pos + s * largo, r)
        pa = pa - cilindro(pos - s * 0.01, pos + s * (largo + 0.4), r + u.holgura)
    elif u.tipo == "bola":
        # Bola en un cuello que sale de la hija; alojamiento esférico en el padre
        # con una boca algo más estrecha que la bola para que encaje a presión
        # y quede sujeta. El cuello, más fino que la boca, permite inclinarla.
        rb = r
        c = pos + s * (rb + 1.0)
        h = h + cilindro(pos - s * 1.0, c, 0.45 * rb) + esfera(c, rb)
        pa = pa - esfera(c, rb + u.holgura) - cilindro(pos - s * 0.01, c, 0.88 * rb)
    return h.transform(vuelta), pa.transform(vuelta)
