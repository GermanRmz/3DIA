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
VOLUMEN_MINIMO = 0.015   # fracción del volumen total para considerar un apéndice
VOLUMEN_MAXIMO = 0.35    # un apéndice no puede llevarse más de esto (protege el cuerpo)
LARGO_MINIMO = 0.06      # fracción del tamaño en esa dirección
ALARGAMIENTO = 1.0       # largo mínimo / ancho del corte
CORTE_MAXIMO = 0.3      # área del corte / sección más ancha de la figura
SOLIDEZ_MINIMA = 0.5     # área de las secciones / área de su envolvente convexa


@dataclass
class Apendice:
    direccion: str
    inicio: float          # nivel (en el marco de la dirección) del extremo
    corte: float           # nivel del corte
    secciones: list        # CrossSections del trozo, del extremo al corte
    volumen: float


@dataclass
class DivisionForma:
    partes: list = field(default_factory=list)     # (nombre, trimesh)
    avisos: list = field(default_factory=list)


def _cadenas(m_marco, z0, z1):
    """Sigue los trozos de sección de abajo arriba. Devuelve apéndices candidatos
    (sin filtrar) en el marco dado."""
    zs = np.linspace(z0, z1, NIVELES + 2)[1:-1]
    paso = zs[1] - zs[0]
    activas = []          # cadenas: dict(niveles, secciones, areas)
    terminadas = []
    area_maxima = 0.0
    for z in zs:
        trozos = [c for c in m_marco.slice(float(z)).decompose() if c.area() > 1e-6]
        area_maxima = max(area_maxima, sum(t.area() for t in trozos))
        nuevas = []
        usadas = set()
        solapes = [[i for i, c in enumerate(activas) if (c["secciones"][-1] ^ t).area() > 1e-6]
                   for t in trozos]
        # Si un trozo se divide, la cadena sigue por el mayor de sus sucesores.
        sucesor = {}
        for j, previas in enumerate(solapes):
            for i in previas:
                if i not in sucesor or trozos[j].area() > trozos[sucesor[i]].area():
                    sucesor[i] = j
        for j, t in enumerate(trozos):
            previas = solapes[j]
            if len(previas) == 1 and sucesor.get(previas[0]) != j:
                previas = []          # rama secundaria de una división: cadena nueva
            if len(previas) >= 2:
                # Fusión: los trozos claramente más pequeños que la sección fusionada
                # son apéndices que terminan aquí (dos piernas que se unen en la
                # cadera, una pata que se une al cuerpo). Si el mayor domina la
                # sección fusionada, es el tronco y sigue.
                previas = [i for i in previas if i not in usadas]
                area = t.area()
                tronco = max(previas, key=lambda i: activas[i]["areas"][-1])
                sigue = activas[tronco]["areas"][-1] > 0.6 * area
                for i in previas:
                    usadas.add(i)
                    if sigue and i == tronco:
                        continue
                    if activas[i]["areas"][-1] <= 0.6 * area:
                        terminadas.append((activas[i], float(z), "fusion"))
                if sigue:
                    c = activas[tronco]
                    c["niveles"].append(z)
                    c["secciones"].append(t)
                    c["areas"].append(t.area())
                    nuevas.append(c)
                else:
                    nuevas.append({"niveles": [z], "secciones": [t], "areas": [t.area()]})
            elif len(previas) == 1 and previas[0] not in usadas:
                c = activas[previas[0]]
                usadas.add(previas[0])
                c["niveles"].append(z)
                c["secciones"].append(t)
                c["areas"].append(t.area())
                nuevas.append(c)
            else:
                nuevas.append({"niveles": [z], "secciones": [t], "areas": [t.area()]})
        activas = nuevas
    terminadas += [(c, None, "fin") for c in activas]
    return terminadas, paso, area_maxima


def _estrechamiento(areas):
    """Índice de un cuello claro (área ≤ 45 % de lo anterior y luego ≥ 2.2 veces),
    o None."""
    a = np.convolve(np.asarray(areas), np.ones(3) / 3, mode="same")
    mejor, ratio = None, 1.0
    for i in range(3, len(a) - 3):
        antes, despues = a[:i].max(), a[i:].max()
        if a[i] <= 0.45 * antes and despues >= 2.2 * a[i]:
            r = a[i] / antes
            if r < ratio:
                mejor, ratio = i, r
    return mejor


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
        cadenas, paso, area_maxima = _cadenas(mm, z0, z1)
        for c, z_fusion, motivo in cadenas:
            areas = c["areas"]
            cortes = []
            if motivo == "fusion":
                n = max(1, len(areas) - 1)               # un nivel antes de la fusión
                if areas[n - 1] >= 0.2 * max(areas[:n]):  # llega con grosor, no en hilo
                    cortes.append(n)
            cuello = _estrechamiento(areas)
            if cuello is not None and areas[cuello - 1] >= 0.04 * max(areas[:cuello]):
                cortes.append(cuello)
            for n in cortes:
                vol = float(sum(areas[:n]) * paso)
                largo = c["niveles"][n - 1] - c["niveles"][0] + paso
                if vol < VOLUMEN_MINIMO * total or vol > VOLUMEN_MAXIMO * total \
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
                candidatos.append(Apendice(nombre, float(c["niveles"][0] - paso),
                                           float(c["niveles"][n - 1]), c["secciones"][:n], vol))
    return sorted(candidatos, key=lambda a: -a.volumen)


def dividir(malla: trimesh.Trimesh, max_partes: int = 8,
            union: OpcionesUnion | None = None) -> DivisionForma:
    import manifold3d as m3d

    union = union or OpcionesUnion()
    res = DivisionForma()
    cuerpo = _a_manifold(malla)
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
        for t in trozos[1:]:                       # lo que no es del apéndice vuelve
            if t.volume() > 1.0:
                resto = resto + t
        pieza, resto = _unir(pieza, resto, np.eye(4)[:3], np.eye(4)[:3], a.corte, -1, union,
                             res.avisos, a.direccion)
        cuentas[a.direccion] = cuentas.get(a.direccion, 0) + 1
        piezas.append((f"pieza {a.direccion} {cuentas[a.direccion]}", pieza.transform(vuelta)))
        cuerpo = resto.transform(vuelta)
    if not piezas:
        res.avisos.append("No se encontraron partes que sobresalgan del cuerpo; usa cortes planos.")
    res.partes = [("cuerpo", _a_trimesh(cuerpo))] + [(n, _a_trimesh(p)) for n, p in piezas]
    return res
