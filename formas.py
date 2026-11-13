"""División automática de cualquier forma (no solo personajes).

Busca apéndices: partes que sobresalen del cuerpo principal y se unen a él por
una sola zona (patas, cola, alas, cuernos, asas, las patas de una mesa...). Para
cada una de las 6 direcciones recorre la figura por secciones desde el extremo
hacia dentro y sigue cada trozo de sección:

- si un trozo acaba fusionándose con otro, es un apéndice: se corta un poco antes
  de la fusión;
- si un trozo se estrecha mucho y luego se ensancha (una cabeza sobre un cuello),
  se corta en el estrechamiento.

Cada apéndice se une al cuerpo con la unión elegida (espiga, rótula, imanes o
ninguna), con el macho en el apéndice.
"""
from dataclasses import dataclass, field

import numpy as np
import trimesh

from personaje import OpcionesUnion, _marco_plano, _unir
from segmentar import _a_manifold, _a_trimesh

DIRECCIONES = {
    "abajo": (0, 0, -1), "arriba": (0, 0, 1), "izquierda": (-1, 0, 0),
    "derecha": (1, 0, 0), "delante": (0, -1, 0), "detrás": (0, 1, 0),
}
NIVELES = 110
VOLUMEN_MINIMO = 0.008   # fracción del volumen total para considerar un apéndice
VOLUMEN_MAXIMO = 0.35    # un apéndice no puede llevarse más de esto (protege el cuerpo)
VOLUMEN_MAXIMO_CUELLO = 0.7  # en un estrechamiento claro (cabeza grande de chibi) se permite más
LARGO_MINIMO = 0.06      # fracción del tamaño en esa dirección
ALARGAMIENTO = 1.0       # largo mínimo / ancho del corte
CORTE_MAXIMO = 0.3      # área del corte / sección más ancha de la figura
SOLIDEZ_MINIMA = 0.5
MIRAR_ADELANTE = 8       # niveles que se miran tras una fusión
CRECIMIENTO = 2.5        # si la sección fusionada crece esto, lo que llega son apéndices     # área de las secciones / área de su envolvente convexa


@dataclass
class Apendice:
    direccion: str
    inicio: float          # nivel (en el marco de la dirección) del extremo
    corte: float           # nivel del corte
    secciones: list        # CrossSections del trozo, del extremo al corte
    volumen: float
    cuello: bool = False   # corte en un estrechamiento (si no, en una fusión)


@dataclass
class DivisionForma:
    partes: list = field(default_factory=list)     # (nombre, trimesh)
    avisos: list = field(default_factory=list)


def _cadenas(m_marco, z0, z1):
    """Sigue los trozos de sección de abajo arriba. Devuelve las cadenas terminadas
    (candidatas a apéndice, sin filtrar), el paso entre niveles y la mayor sección.

    Dos pasadas: primero se registran todas las secciones y cómo se solapan entre
    niveles; luego, en cada fusión, se mira unos niveles más allá: si la sección
    fusionada crece mucho (una pata que entra en el cuerpo) las que llegan son
    apéndices; si apenas crece (un trozo de ropa que se pega a la pierna), la
    mayor sigue y solo terminan las demás."""
    zs = np.linspace(z0, z1, NIVELES + 2)[1:-1]
    paso = zs[1] - zs[0]
    niveles = []          # por nivel: (z, trozos, solapes con el anterior, sucesor de cada previo)
    previos = []
    for z in zs:
        trozos = [c for c in m_marco.slice(float(z)).decompose() if c.area() > 1e-6]
        solapes = [[i for i, p in enumerate(previos) if (p ^ t).area() > 1e-6] for t in trozos]
        sucesor = {}
        for j, ps in enumerate(solapes):
            for i in ps:
                if i not in sucesor or trozos[j].area() > trozos[sucesor[i]].area():
                    sucesor[i] = j
        niveles.append((float(z), trozos, solapes, sucesor))
        previos = trozos
    area_maxima = max((sum(t.area() for t in n[1]) for n in niveles), default=0.0)

    def crecimiento(k, j):
        """Mayor área siguiendo al sucesor principal desde (nivel k, trozo j)."""
        mayor = niveles[k][1][j].area()
        for kk in range(k + 1, min(k + 1 + MIRAR_ADELANTE, len(niveles))):
            j = niveles[kk][3].get(j)
            if j is None:
                break
            mayor = max(mayor, niveles[kk][1][j].area())
        return mayor

    activas = []
    terminadas = []
    registro = {}         # id -> cadena (para recuperar los antepasados de una fusión)
    for k, (z, trozos, solapes, sucesor) in enumerate(niveles):
        nuevas = []
        usadas = set()
        for j, t in enumerate(trozos):
            previas = solapes[j]
            if len(previas) == 1 and sucesor.get(previas[0]) != j:
                previas = []          # rama secundaria de una división: cadena nueva
            previas = [i for i in previas if i not in usadas]
            if len(previas) >= 2:
                tronco = max(previas, key=lambda i: activas[i]["areas"][-1])
                crece = crecimiento(k, j) >= CRECIMIENTO * activas[tronco]["areas"][-1]
                for i in previas:
                    usadas.add(i)
                    if crece or i != tronco:
                        terminadas.append((activas[i], z, "fusion"))
                if crece:
                    nuevas.append({"niveles": [z], "secciones": [t], "areas": [t.area()],
                                   "hijos": [id(activas[i]) for i in previas]})
                else:
                    c = activas[tronco]
                    c["niveles"].append(z)
                    c["secciones"].append(t)
                    c["areas"].append(t.area())
                    c["hijos"] += [id(activas[i]) for i in previas if i != tronco]
                    nuevas.append(c)
            elif len(previas) == 1:
                c = activas[previas[0]]
                usadas.add(previas[0])
                c["niveles"].append(z)
                c["secciones"].append(t)
                c["areas"].append(t.area())
                nuevas.append(c)
            else:
                nuevas.append({"niveles": [z], "secciones": [t], "areas": [t.area()], "hijos": []})
        activas = nuevas
        for c in activas:
            registro[id(c)] = c
    terminadas += [(c, None, "fin") for c in activas]
    return terminadas, paso, area_maxima, registro


def _antepasados(c, registro):
    """Secciones e inicio de la cadena junto con todas las que se fundieron en ella."""
    secciones, inicio, pendientes, vistos = [], c["niveles"][0], [c], set()
    while pendientes:
        x = pendientes.pop()
        if id(x) in vistos:
            continue
        vistos.add(id(x))
        secciones += x["secciones"]
        inicio = min(inicio, x["niveles"][0])
        pendientes += [registro[h] for h in x.get("hijos", []) if h in registro]
    return secciones, inicio


def _estrechamiento(areas):
    """Primer cuello claro desde el extremo: un mínimo local con área ≤ 45 % de lo
    anterior y que luego vuelve a crecer ≥ 1.6 veces (el cuerpo puede ser más
    pequeño que la cabeza). Devuelve su índice o None."""
    a = np.convolve(np.asarray(areas), np.ones(3) / 3, mode="same")
    i = 3
    while i < len(a) - 3:
        if a[i] <= 0.45 * a[:i].max() and a[i:].max() >= 1.6 * a[i]:
            # Bajar hasta el mínimo local de este estrechamiento.
            while i + 1 < len(a) - 3 and a[i + 1] <= a[i]:
                i += 1
            if a[i:].max() >= 1.6 * a[i]:
                return i
        i += 1
    return None


def detectar(malla: trimesh.Trimesh) -> list[Apendice]:
    m = _a_manifold(malla)
    total = m.volume()
    candidatos = []
    for nombre, d in DIRECCIONES.items():
        normal = -np.asarray(d, float)            # en el marco, el extremo queda abajo
        ida, _ = _marco_plano((0, 0, 0), normal)
        mm = m.transform(ida)
        b = np.asarray(mm.bounding_box())
        z0, z1 = b[2], b[5]
        tam = z1 - z0
        cadenas, paso, area_maxima, registro = _cadenas(mm, z0, z1)
        aceptadas = set()
        for c, z_fusion, motivo in cadenas:
            areas = c["areas"]
            # Si de esta cadena ya salen apéndices (p. ej. dos patas que se unen en el
            # pecho), se prefieren esas piezas a cortar el grupo.
            grupo = any(h in aceptadas for h in c.get("hijos", []))
            cortes = []
            if motivo == "fusion" and not grupo:
                n = max(1, len(areas) - 1)               # un nivel antes de la fusión
                if areas[n - 1] >= 0.2 * max(areas[:n]):  # llega con grosor, no en hilo
                    cortes.append((n, VOLUMEN_MAXIMO))
            cuello = _estrechamiento(areas)
            if cuello is not None and areas[cuello - 1] >= 0.04 * max(areas[:cuello]):
                cortes.append((cuello, VOLUMEN_MAXIMO_CUELLO))
            for n, vol_max in cortes:
                vol = float(sum(areas[:n]) * paso)
                largo = c["niveles"][n - 1] - c["niveles"][0] + paso
                if vol < VOLUMEN_MINIMO * total or vol > vol_max * total \
                        or largo < LARGO_MINIMO * tam:
                    continue
                # El corte debe ser fino comparado con la sección más ancha de la figura
                # (una pata, una cola, un cuello; no la barriga o el tronco del cuerpo).
                if areas[n - 1] > CORTE_MAXIMO * area_maxima:
                    continue
                # Un apéndice de verdad es alargado en la dirección del corte (una
                # pierna vista desde abajo); visto de lado sería un corte a lo largo.
                if largo < ALARGAMIENTO * np.sqrt(areas[n - 1]):
                    continue
                # Y macizo: sus secciones llenan buena parte de su contorno convexo
                # (una falda o una manta que cuelga son bandas finas, no apéndices).
                muestras = c["secciones"][:n][::max(1, n // 6)]
                solidez = np.mean([s.area() / max(s.hull().area(), 1e-9) for s in muestras])
                if solidez < SOLIDEZ_MINIMA:
                    continue
                secciones, inicio = c["secciones"][:n], c["niveles"][0]
                if vol_max == VOLUMEN_MAXIMO_CUELLO:
                    # Corte en un cuello: la pieza es todo lo que hay más allá (la
                    # cabeza entera con las orejas o cuernos que se fundieron en ella).
                    previas, inicio = _antepasados(c, registro)
                    secciones = previas + secciones
                candidatos.append(Apendice(nombre, float(inicio - paso),
                                           float(c["niveles"][n - 1]), secciones, vol,
                                           cuello=vol_max == VOLUMEN_MAXIMO_CUELLO))
                aceptadas.add(id(c))
                break
    # Primero los apéndices que se funden en el cuerpo (patas, cola...), de mayor a
    # menor, y al final los cortes de cuello: así las patas salen del cuerpo antes
    # de separar la cabeza.
    return sorted(candidatos, key=lambda a: (a.cuello, -a.volumen))


def dividir(malla: trimesh.Trimesh, max_partes: int = 8,
            union: OpcionesUnion | None = None) -> DivisionForma:
    import manifold3d as m3d

    union = union or OpcionesUnion()
    res = DivisionForma()
    cuerpo = _a_manifold(malla)
    volumen_total = cuerpo.volume()
    piezas = []
    cuentas = {}
    for a in detectar(malla):
        if len(piezas) + 1 >= max_partes:
            break
        normal = -np.asarray(DIRECCIONES[a.direccion], float)
        ida, vuelta = _marco_plano((0, 0, 0), normal)
        cm = cuerpo.transform(ida)
        # Cortador: la huella del apéndice (sus secciones, algo ensanchadas)
        # extruida desde el extremo hasta el corte.
        huella = a.secciones[0]
        for s in a.secciones[1:]:
            huella = huella + s
        huella = huella.offset(0.5, m3d.JoinType.Round)
        alto = a.corte - a.inicio + 1.0
        cortador = huella.extrude(alto).translate((0, 0, a.inicio - 1.0))
        pieza = cm ^ cortador
        if pieza.is_empty():
            continue
        trozos = sorted(pieza.decompose(), key=lambda t: -t.volume())
        # Si ya se cortó desde otra dirección, aquí quedaría poco (o un trozo del
        # cuerpo): se descarta.
        if trozos[0].volume() < max(VOLUMEN_MINIMO * cuerpo.volume(), 0.5 * a.volumen):
            continue
        pieza = trozos[0]
        resto = cm - cortador
        # Si casi no queda cuerpo, ese corte ya se hizo desde el otro lado (el mismo
        # cuello visto desde arriba y desde abajo).
        if resto.volume() < 0.15 * volumen_total:
            continue
        for t in trozos[1:]:                       # lo que no es del apéndice vuelve
            if t.volume() > 1.0:
                resto = resto + t
        avisos = []
        pieza, resto = _unir(pieza, resto, np.eye(4)[:3], np.eye(4)[:3], a.corte, -1, union,
                             avisos, a.direccion)
        if avisos and union.tipo != "ninguna":
            continue          # sin sitio para la unión: mejor no separar esta pieza
        cuentas[a.direccion] = cuentas.get(a.direccion, 0) + 1
        piezas.append((f"pieza {a.direccion} {cuentas[a.direccion]}", pieza.transform(vuelta)))
        cuerpo = resto.transform(vuelta)
    if not piezas:
        res.avisos.append("No se encontraron partes que sobresalgan del cuerpo; usa cortes planos.")
    res.partes = [("cuerpo", _a_trimesh(cuerpo))] + [(n, _a_trimesh(p)) for n, p in piezas]
    return res
