"""Mecanismos: piezas móviles funcionales (impresas) y animación.

1. analizar(): Claude mira la imagen original y tres vistas del modelo con
   cuadrícula en mm, y propone las piezas móviles: tipo (giro o bisagra), la caja
   que las contiene, el pivote, el eje y el rango de movimiento.
2. construir(): separa cada pieza con holgura y le pone su eje o pasador:
   - "montada" (print-in-place): el pasador sale del cuerpo y la pieza gira en
     él; si solo se apoya por un lado (una rueda), una pestaña la retiene.
   - "separadas": agujeros en ambas partes y pasadores sueltos para montar.
3. El resultado se exporta como .zip/.3mf para imprimir y .glb animado.

Necesita una clave de la API de Anthropic (variable ANTHROPIC_API_KEY o
`ant auth login`) solo para analizar; construir funciona con cualquier JSON.
"""
import base64
import io
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import trimesh

import animacion
import vistas
from personaje import _marco_plano
from segmentar import Resultado, _a_manifold, _a_trimesh

MODELO_IA = "claude-opus-5-5"

ESQUEMA = {
    "type": "object",
    "properties": {
        "resumen": {"type": "string"},
        "piezas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string"},
                    "tipo": {"type": "string", "enum": ["giro", "bisagra"]},
                    "caja_min": {"type": "array", "items": {"type": "number"}},
                    "caja_max": {"type": "array", "items": {"type": "number"}},
                    "pivote": {"type": "array", "items": {"type": "number"}},
                    "eje": {"type": "string", "enum": ["x", "y", "z"]},
                    "angulo_min": {"type": "number"},
                    "angulo_max": {"type": "number"},
                    "movimiento": {"type": "string", "enum": ["lanzar", "oscilar"]},
                    "fase": {"type": "number"},
                    "explicacion": {"type": "string"},
                },
                "required": ["nombre", "tipo", "caja_min", "caja_max", "pivote", "eje",
                             "angulo_min", "angulo_max", "movimiento", "fase", "explicacion"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["resumen", "piezas"],
    "additionalProperties": False,
}

INSTRUCCIONES = """Eres un ingeniero de mecanismos para impresión 3D y animación. Te paso la \
imagen original de un objeto y tres vistas ortográficas de su modelo 3D con cuadrícula en \
milímetros.

Sistema de coordenadas (mm): Z hacia arriba, el frente del objeto mira hacia -Y.
- Vista "frente": horizontal = X, vertical = Z (la izquierda de la imagen es -X).
- Vista "lado": horizontal = Y, vertical = Z (el frente del objeto está a la izquierda, -Y).
- Vista "arriba": horizontal = X, vertical = Y (el frente del objeto está abajo, -Y).
Medidas del modelo: {medidas}.

Identifica las piezas que en el objeto real se moverían: ruedas que giran sobre su eje, el \
brazo de una catapulta, una puerta, una tapa, una mandíbula, una hélice; en animales, \
robots o personajes: patas, cabeza, cola, brazos. Para cada una:
- tipo: "giro" si da vueltas completas (ruedas, hélices), "bisagra" si va y viene entre \
dos ángulos.
- caja_min y caja_max: [x, y, z] de una caja que contenga SOLO esa pieza. La cara de la \
caja del lado de la unión va justo en la articulación: la caja de una pata acaba donde la \
pata entra en el cuerpo, sin coger la barriga ni el pecho; la de una cabeza empieza en el \
cuello. Comprueba la caja en las TRES vistas: en cada una debe rodear la pieza y no \
invadir otras partes. Las piezas simétricas (patas izquierda y derecha) llevan cajas \
simétricas.
- pivote: [x, y, z] del centro de la articulación, dentro de la pieza y en su extremo \
unido al cuerpo: el centro del eje de una rueda, la cadera u hombro de una pata (arriba de \
la pata), la base de la cola, el cuello.
- eje: "x", "y" o "z". El eje (el pasador) debe atravesar la pieza y entrar en el cuerpo \
por un lado: una pata unida al costado gira sobre el eje que va hacia ese costado; una \
cola que sale del lomo hacia arriba se menea sobre "z". Una pata que camina hacia delante gira sobre "x"; una cabeza que dice \
que no gira sobre "z" y una que asiente sobre "x"; una rueda sobre el eje de su buje.
- angulo_min y angulo_max en grados (0 = posición del modelo; signo según la regla de la \
mano derecha sobre el eje positivo). Para "giro", 0 y 360. Usa amplitudes realistas: \
patas de -25 a 25, cabeza de -20 a 20, cola de -30 a 30, brazo de catapulta 0 a -70.
- movimiento: "lanzar" si sube rápido y vuelve despacio (catapulta, golpe), "oscilar" si \
es un vaivén continuo (patas al andar, cola, cabeza, alas).
- fase: de 0 a 1, para coordinar: en un animal de cuatro patas que camina, las patas \
diagonales van juntas (delantera izquierda y trasera derecha 0, las otras dos 0.5). Si no \
hace falta, 0.
Incluye solo piezas que se puedan separar de forma razonable. Lee las coordenadas en las \
cuadrículas con cuidado. Responde en español.{descripcion}"""

REVISION = """Esta es la vista con tus cajas dibujadas (cada una con su nombre). Revísalas en \
las tres vistas: cada caja debe rodear solo su pieza, la cara del lado de la unión debe \
quedar justo en la articulación (sin coger cuerpo, barriga ni pecho), el pivote dentro de \
la pieza en su extremo unido y las piezas simétricas con cajas simétricas. Devuelve la \
lista completa corregida (aunque no cambies nada)."""


@dataclass
class PiezaMovil:
    nombre: str
    tipo: str
    caja_min: list
    caja_max: list
    pivote: list
    eje: str
    angulo_min: float = 0.0
    angulo_max: float = 360.0
    movimiento: str = "lanzar"
    fase: float = 0.0
    explicacion: str = ""


@dataclass
class Mecanismo:
    partes: list = field(default_factory=list)       # Articulacion (animacion.py)
    pasadores: list = field(default_factory=list)    # trimesh
    avisos: list = field(default_factory=list)


EJES = {"x": (1.0, 0, 0), "y": (0, 1.0, 0), "z": (0, 0, 1.0)}
COLORES = [(230, 90, 70), (70, 150, 230), (90, 190, 110), (240, 190, 60), (170, 100, 210),
           (60, 190, 190), (230, 120, 170), (150, 150, 90)]


# --- 1. Análisis con IA ------------------------------------------------------------

def _png_b64(imagen) -> dict:
    buf = io.BytesIO()
    imagen.convert("RGB").save(buf, format="PNG")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def analizar(malla: trimesh.Trimesh, imagen_original=None, descripcion: str = "",
             revisar: bool = True) -> tuple[list, str]:
    """Pide a Claude las piezas móviles. Devuelve (lista de PiezaMovil, resumen).

    Con `revisar`, se le enseñan sus propias cajas dibujadas sobre las vistas para que
    las corrija (cuesta el doble, pero las cajas salen mucho mejor)."""
    import anthropic

    lo, hi = malla.bounds
    medidas = ", ".join(f"{e} de {a:.0f} a {b:.0f}" for e, a, b in zip("XYZ", lo, hi))
    texto = INSTRUCCIONES.format(
        medidas=medidas,
        descripcion=f"\n\nIndicaciones del usuario: {descripcion}" if descripcion.strip() else "")
    contenido = []
    if imagen_original is not None:
        contenido += [{"type": "text", "text": "Imagen original:"}, _png_b64(imagen_original)]
    for nombre, im in vistas.vistas(malla).items():
        contenido += [{"type": "text", "text": f"Vista {nombre}:"}, _png_b64(im)]
    contenido.append({"type": "text", "text": texto})
    mensajes = [{"role": "user", "content": contenido}]

    cliente = anthropic.Anthropic()

    def pedir():
        respuesta = cliente.beta.messages.create(
            model=MODELO_IA,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                 # si el modelo declina, otro lo intenta
            output_config={"effort": "high",
                           "format": {"type": "json_schema", "schema": ESQUEMA}},
            messages=mensajes,
        )
        if respuesta.stop_reason == "refusal":
            raise RuntimeError("La IA no quiso analizar este modelo.")
        if respuesta.stop_reason == "max_tokens":
            raise RuntimeError("La respuesta de la IA se cortó; inténtalo de nuevo.")
        return next(b.text for b in respuesta.content if b.type == "text")

    crudo = pedir()
    datos = json.loads(crudo)
    if revisar and datos["piezas"]:
        piezas = [PiezaMovil(**p) for p in datos["piezas"]]
        mensajes += [{"role": "assistant", "content": crudo},
                     {"role": "user", "content": [_png_b64(vista_previa(malla, piezas)),
                                                  {"type": "text", "text": REVISION}]}]
        datos = json.loads(pedir())
    return [PiezaMovil(**p) for p in datos["piezas"]], datos["resumen"]


def a_json(piezas: list, resumen: str = "") -> str:
    return json.dumps({"resumen": resumen, "piezas": [asdict(p) for p in piezas]},
                      ensure_ascii=False, indent=2)


def desde_json(texto: str) -> list:
    datos = json.loads(texto)
    campos = set(PiezaMovil.__dataclass_fields__)
    return [PiezaMovil(**{k: v for k, v in p.items() if k in campos}) for p in datos.get("piezas", [])]


def vista_previa(malla, piezas: list):
    """Las tres vistas con las cajas de las piezas dibujadas."""
    cajas = [(p.caja_min, p.caja_max, COLORES[i % len(COLORES)], p.nombre)
             for i, p in enumerate(piezas)]
    return vistas.mosaico(vistas.vistas(malla, cajas))


# --- 2. Construcción de las piezas ---------------------------------------------------

def _caja(minimo, maximo):
    import manifold3d as m3d

    a, b = np.minimum(minimo, maximo), np.maximum(minimo, maximo)
    return m3d.Manifold.cube(tuple(b - a)).translate(tuple(a))


def _intervalos(solido, largo=1e3):
    """Tramos del eje Z (marco de la articulación) ocupados por `solido` cerca del eje."""
    import manifold3d as m3d

    linea = m3d.Manifold.cylinder(2 * largo, 0.4, 0.4, 12).translate((0, 0, -largo))
    tramos = []
    for t in (solido ^ linea).decompose():
        b = t.bounding_box()
        tramos.append((b[2], b[5]))
    return sorted(tramos)


def construir(malla: trimesh.Trimesh, piezas: list, modo: str = "montada",
              holgura: float = 0.3, diametro: float = 0.0) -> Mecanismo:
    """Separa las piezas móviles y les pone su eje o pasador."""
    import manifold3d as m3d

    res = Mecanismo()
    cuerpo = _a_manifold(malla)
    moviles = []
    apendices = []
    if any(p.tipo == "bisagra" for p in piezas):
        import formas
        for a in formas.detectar(malla):
            _, vuelta = formas.marco(a)
            apendices.append((a, formas.cortador_apendice(a).transform(vuelta), vuelta))
    for p in piezas:
        lo, hi = np.minimum(p.caja_min, p.caja_max), np.maximum(p.caja_min, p.caja_max)
        if p.tipo == "bisagra":
            corte = _cortar_articulacion(cuerpo, p, lo, hi, holgura, res, apendices=apendices)
            if corte is None:
                res.avisos.append(f"{p.nombre}: la caja no contiene nada del modelo; se omite.")
                continue
            hija, cuerpo = corte
            moviles.append((p, hija))
            continue
        lo, hi = _ajustar_caja(cuerpo, lo, hi, p, res, holgura)
        hija = _elegir_hija(cuerpo, lo, hi, holgura, np.asarray(p.pivote, float))
        if hija is None:
            res.avisos.append(f"{p.nombre}: la caja no contiene nada del modelo; se omite.")
            continue
        # Se quita del cuerpo la pieza y, alrededor de ella, la capa de holgura de las
        # caras de la caja (donde la pieza se unía al resto). Lo demás que hubiera en
        # la caja (otra pieza, la barriga) se queda en el cuerpo.
        b = np.asarray(hija.bounding_box())
        capa = (_caja(lo, hi) - _caja(lo + holgura, hi - holgura)) ^ \
            _caja(b[:3] - holgura - 0.5, b[3:] + holgura + 0.5)
        cuerpo = cuerpo - hija - capa
        moviles.append((p, hija))

    for i, (p, hija) in enumerate(moviles):
        hija, cuerpo = _eje(p, hija, cuerpo, modo, holgura, diametro, res)
        # Fuera migas sueltas (p. ej. un adorno del buje que el agujero dejó aislado).
        trozos = sorted(hija.decompose(), key=lambda t: -t.volume())
        hija = m3d.Manifold.batch_boolean([t for t in trozos if t.volume() > 0.01 * trozos[0].volume()],
                                          m3d.OpType.Add)
        res.partes.append(animacion.Articulacion(
            p.nombre, _a_trimesh(hija), "giro" if p.tipo == "giro" else "bisagra",
            tuple(p.pivote), tuple(_vector(p)), p.angulo_min, p.angulo_max,
            movimiento=p.movimiento, fase=float(p.fase), color=COLORES[i % len(COLORES)]))
    trozos = sorted(cuerpo.decompose(), key=lambda t: -t.volume())
    cuerpo = m3d.Manifold.batch_boolean([t for t in trozos if t.volume() > 0.002 * trozos[0].volume()],
                                        m3d.OpType.Add)
    res.partes.insert(0, animacion.Articulacion("cuerpo", _a_trimesh(cuerpo), "fija",
                                                color=(210, 210, 210)))
    return res


def _vector(p):
    """Dirección del eje: la del corte inclinado si se calculó, si no la de la IA."""
    v = getattr(p, "vector", None)
    return tuple(v) if v is not None else EJES[p.eje]


def _en_marco(solido_marco, vuelta):
    return solido_marco.transform(vuelta)


def _cerca_de(trozos, punto):
    """El trozo más cercano a `punto` (de los que no son migas)."""
    trozos = [t for t in trozos if t.volume() > 0.02 * max(x.volume() for x in trozos)]

    def distancia(t):
        b = np.asarray(t.bounding_box())
        return np.linalg.norm(np.maximum(0, np.maximum(b[:3] - punto, punto - b[3:])))
    return min(trozos, key=distancia)


def _cortar_articulacion(cuerpo, p, lo, hi, holgura, res, L=400.0, apendices=()):
    """Separa una pieza de bisagra con un corte que le deja girar sin chocar: la
    superficie de unión tiene que ser de revolución alrededor del eje.

    - Eje a lo largo de la pieza (cabeza que gira, cola): plano por el cuello, el
      de menor sección cerca del pivote, que puede ir inclinado; la pieza gira
      alrededor de la perpendicular a ese plano.
    - Eje cruzado con la pieza (pata, brazo, mandíbula): un disco alrededor del eje
      (como un hombro) más lo que cuelga más allá del pivote. Así la parte de la
      pata pegada al costado se va con la pata.
    Devuelve (pieza, cuerpo sin la pieza y con la holgura) o None."""
    import manifold3d as m3d

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    piv = np.asarray(p.pivote, float)
    eje = np.asarray(EJES[p.eje], float)
    hacia = (lo + hi) / 2 - piv                       # del pivote hacia la pieza
    if np.linalg.norm(hacia) < 1e-6 or (cuerpo ^ _caja(lo, hi)).is_empty():
        return None
    h2 = holgura / 2

    def semiespacio(punto, normal, desde):
        ida, vuelta = _marco_plano(punto, normal)
        return _caja(np.array([-L, -L, desde]), np.array([L, L, L])).transform(vuelta)

    coseno = abs(hacia @ eje) / np.linalg.norm(hacia)
    if coseno > 0.64:
        # --- Corte plano por el cuello (eje a lo largo de la pieza) ---
        base = eje * np.sign(hacia @ eje)
        aux = np.array([1.0, 0, 0]) if abs(base[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(base, aux)
        u /= np.linalg.norm(u)
        v = np.cross(base, u)
        normales = [base] + [np.cos(np.radians(g)) * base + np.sin(np.radians(g)) *
                             (np.cos(f) * u + np.sin(f) * v)
                             for g in (15, 30) for f in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        # Zona amplia: si se recortara justo en la caja, en su borde la sección
        # parecería mínima (falso cuello).
        # Zona de la pieza: la caja ensanchada, y hacia el cuerpo un poco más para
        # ver dónde empieza (el salto de sección).
        hacia_cuerpo = -base
        ext_lo = np.where(hacia_cuerpo < -0.5, 10.0, 3.0)
        ext_hi = np.where(hacia_cuerpo > 0.5, 10.0, 3.0)
        zona = cuerpo ^ _caja(lo - ext_lo, hi + ext_hi)
        esquinas = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                             for z in (lo[2], hi[2])])
        en_caja = (cuerpo ^ _caja(lo, hi)).volume()
        mejor = None
        for n in normales:
            ida, _ = _marco_plano(piv, n)
            r = zona.transform(ida)
            r_caja = (cuerpo ^ _caja(lo, hi)).transform(ida)
            punta = float(((esquinas - piv) @ n).max())
            niveles = np.arange(punta - 0.5, -8.0, -0.5)        # de la punta hacia el cuerpo
            areas = np.array([r.slice(float(z)).area() for z in niveles])
            corte = None
            for k in range(4, len(niveles)):
                previas = areas[max(0, k - 6):k]
                previas = previas[previas > 0]
                if len(previas) and niveles[k] < 6.0 and areas[k] > 2.2 * previas.min():
                    corte = k - 1          # justo antes de que la sección salte al cuerpo
                    break
            if corte is None:              # sin salto claro: la sección más fina cerca del pivote
                cerca = np.where(np.abs(niveles) <= 4.0)[0]
                cerca = [k for k in cerca if areas[k] > 0]
                if not cerca:
                    continue
                corte = min(cerca, key=lambda k: areas[k])
            # El corte debe dejar del lado de la pieza casi toda la pieza (un plano
            # que corta la cola de lado deja solo un trozo).
            caja_n = np.array([r_caja.slice(float(z)).area() for z in niveles[:corte + 1]])
            if caja_n.sum() * 0.5 < 0.7 * en_caja:
                continue
            coste = areas[corte] * (1 + 1.5 * (1 - n @ base))
            if mejor is None or coste < mejor[0]:
                mejor = (coste, n, float(niveles[corte]) - 0.25)
        if mejor is None:
            return None
        _, n, desp = mejor
        p0 = piv + n * desp
        # El pasador, en el centro del cuello (el pivote de la IA puede caer al borde).
        ida, vuelta = _marco_plano(p0, n)
        seccion = zona.transform(ida).slice(0.0)
        partes = [np.asarray(q) for q in seccion.to_polygons() if len(q) >= 3]
        if partes:
            cerca = min(partes, key=lambda q: np.linalg.norm(q.mean(0)))
            c2 = cerca.mean(0)
            if np.linalg.norm(c2) < 12:
                p0 = vuelta[:, :3] @ np.array([c2[0], c2[1], 0.0]) + vuelta[:, 3]
        lado = cuerpo ^ semiespacio(p0, n, h2) ^ _caja(lo - 6.0, hi + 6.0)
        trozos = lado.decompose()
        if not trozos:
            return None
        hija = _cerca_de(trozos, p0 + 3 * n)
        b = np.asarray(hija.bounding_box())
        ida, vuelta = _marco_plano(p0, n)
        rendija = _caja(np.array([-L, -L, -h2]), np.array([L, L, h2])).transform(vuelta) ^ \
            _caja(b[:3] - 1, b[3:] + 1)
        cuerpo = cuerpo - hija - rendija
        inclinado = np.degrees(np.arccos(np.clip(n @ base, -1, 1)))
        p.pivote = [round(float(x), 2) for x in p0]
        p.vector = [round(float(x), 4) for x in n * np.sign(n @ eje)]
        res.avisos.append(f"{p.nombre}: corte por el cuello" +
                          (f", inclinado {inclinado:.0f}°" if inclinado > 1 else "") +
                          "; gira alrededor de la perpendicular al corte.")
        return hija, cuerpo

    # --- Corte de hombro: disco alrededor del eje + lo que cuelga ---
    # El pasador debe ir en el centro del disco y tener cuerpo al lado donde
    # apoyarse: se prueba el pivote y, si no, puntos cercanos.
    d = hacia - (hacia @ eje) * eje
    d /= np.linalg.norm(d)
    zona = None
    # Si la división automática encontró este apéndice (una pata entera hasta donde
    # se funde con el cuerpo), se usa su forma y su corte en vez de la caja.
    apendice = _apendice_de(cuerpo, lo, hi, apendices, eje)
    if apendice is not None:
        zona, d, piv = apendice
        res.avisos.append(f"{p.nombre}: se separa la pieza entera hasta donde se une al cuerpo.")
    e = np.cross(eje, d)
    e /= np.linalg.norm(e)
    candidatos = [piv] + [piv + i * d + j * e for i in np.arange(-4.5, 4.6, 1.5)
                          for j in np.arange(-4.5, 4.6, 1.5) if i or j]
    candidatos = sorted(candidatos, key=lambda q: np.linalg.norm(q - piv))
    primero = None
    for q in candidatos:
        corte = _hombro(cuerpo, p, lo, hi, holgura, q, eje, d, e, L, zona)
        if corte is None:
            continue
        primero = primero or (corte, q)
        if _tiene_apoyo(corte[0], corte[1], q, eje, holgura):
            break
    else:
        if primero is None:
            return None
        corte, q = primero
    hija, cuerpo, radio = corte
    if np.linalg.norm(q - piv) > 0.1:
        res.avisos.append(f"{p.nombre}: pivote movido {np.linalg.norm(q - piv):.1f} mm para que "
                          "el pasador apoye en el cuerpo.")
    p.pivote = [round(float(x), 2) for x in q]
    res.avisos.append(f"{p.nombre}: articulación de hombro de {2 * radio:.1f} mm alrededor del eje "
                      "(la pieza se lleva la parte pegada al cuerpo).")
    return hija, cuerpo


def _tiene_apoyo(hija, cuerpo, punto, eje, holgura):
    ida, _ = _marco_plano(punto, eje)
    th = _intervalos(hija.transform(ida))
    if not th:
        return False
    h0, h1 = min(th, key=lambda t: abs((t[0] + t[1]) / 2))
    if h0 > 0.5 or h1 < -0.5:
        return False
    cerca = 3 * holgura + 2
    return any((t[1] <= h0 + 0.1 and h0 - t[1] < cerca) or (t[0] >= h1 - 0.1 and t[0] - h1 < cerca)
               for t in _intervalos(cuerpo.transform(ida)))


def _apendice_de(cuerpo, lo, hi, apendices, eje):
    """El apéndice detectado (formas.py) que corresponde a la caja de la IA:
    (zona del cortador, dirección hacia la punta, centro del corte) o None."""
    import formas

    en_caja = (cuerpo ^ _caja(lo, hi)).volume()
    mejor = None
    for a, cortador, vuelta in apendices:
        d = np.asarray(formas.DIRECCIONES[a.direccion], float)
        if abs(d @ eje) > 0.5:          # el eje debe cruzar la pieza
            continue
        trozos = (cuerpo ^ cortador).decompose()
        if not trozos:
            continue
        pieza = max(trozos, key=lambda t: t.volume())
        v = pieza.volume()
        dentro = (pieza ^ _caja(lo - 2.0, hi + 2.0)).volume()
        if dentro < 0.6 * v or dentro < 0.4 * en_caja:
            continue
        if mejor is None or dentro > mejor[0]:
            mejor = (dentro, a, cortador, vuelta, pieza)
    if mejor is None:
        return None
    _, a, cortador, vuelta, pieza = mejor
    d = np.asarray(formas.DIRECCIONES[a.direccion], float)
    # Centro de la sección de la pieza junto al corte.
    ida = np.linalg.inv(np.vstack([vuelta, [0, 0, 0, 1]]))[:3]
    seccion = pieza.transform(ida).slice(float(a.corte) - 0.3)
    puntos = [np.asarray(q) for q in seccion.to_polygons() if len(q) >= 3]
    if not puntos:
        return None
    c2 = max(puntos, key=len).mean(0)
    centro = vuelta[:, :3] @ np.array([c2[0], c2[1], a.corte]) + vuelta[:, 3]
    return cortador, d, centro


def _hombro(cuerpo, p, lo, hi, holgura, piv, eje, d, e, L, zona=None):
    """Corte de hombro con el pivote `piv`: (pieza, cuerpo, radio) o None. `zona`
    limita lo que cuelga (por defecto, la caja de la IA algo ensanchada)."""
    import manifold3d as m3d

    if zona is None:
        zona = _caja(lo - 2.0, hi + 2.0)

    h2 = holgura / 2

    def semiespacio(punto, normal, desde):
        _, vuelta = _marco_plano(punto, normal)
        return _caja(np.array([-L, -L, desde]), np.array([L, L, L])).transform(vuelta)

    colgante = cuerpo ^ semiespacio(piv, d, h2) ^ zona
    trozos = colgante.decompose()
    if not trozos:
        return None
    colg = _cerca_de(trozos, piv + 3 * d)
    # Radio del hombro: media anchura de la pieza justo debajo del pivote.
    ida_d, vuelta_d = _marco_plano(piv + 2.5 * d, d)
    seccion = colg.transform(ida_d).slice(0.0)
    puntos = [np.asarray(q) for q in seccion.to_polygons()]
    if puntos:
        pts = np.vstack(puntos)
        p3 = pts @ vuelta_d[:, :2].T                  # direcciones en 3D (sin traslación)
        ancho = float(np.ptp(p3 @ e))
    else:
        ancho = 6.0
    radio = float(np.clip(0.5 * ancho, 2.5, 6.0))
    # Tramo a lo largo del eje: el de la pieza que cuelga.
    vb = np.asarray(colg.to_mesh().vert_properties)[:, :3]
    t = (vb - piv) @ eje
    t0, t1 = float(t.min()), float(t.max())
    ida_a, vuelta_a = _marco_plano(piv, eje)

    def cilindro(r, a, b):
        return m3d.Manifold.cylinder(b - a, r, r, 64).translate((0, 0, a)).transform(vuelta_a)

    region = cilindro(radio - h2, t0, t1) + (semiespacio(piv, d, h2) ^ zona)
    trozos = (cuerpo ^ region).decompose()
    if not trozos:
        return None
    marca = m3d.Manifold.sphere(1.0, 12).translate(tuple(piv))
    con = [x for x in trozos if not (x ^ marca).is_empty()]
    hija = max(con, key=lambda x: x.volume()) if con else _cerca_de(trozos, piv)
    b = np.asarray(hija.bounding_box())
    exterior = cilindro(radio + h2, t0 - holgura, t1 + holgura) + \
        (semiespacio(piv, d, -h2) ^ _caja(b[:3] - 0.5, b[3:] + 0.5))
    cuerpo = cuerpo - hija - (exterior - hija)
    return hija, cuerpo, radio



def _elegir_hija(cuerpo, lo, hi, holgura, pivote):
    """Trozo del modelo dentro de la caja (encogida por la holgura) que contiene el
    pivote o, si ninguno lo contiene, el más cercano a él."""
    import manifold3d as m3d

    trozos = (cuerpo ^ _caja(lo + holgura, hi - holgura)).decompose()
    if not trozos:
        return None
    marca = m3d.Manifold.sphere(1.5, 12).translate(tuple(pivote))
    con_pivote = [t for t in trozos if not (t ^ marca).is_empty()]
    if con_pivote:
        return max(con_pivote, key=lambda t: t.volume())

    def distancia(t):
        b = np.asarray(t.bounding_box())
        return np.linalg.norm(np.maximum(0, np.maximum(b[:3] - pivote, pivote - b[3:])))
    grandes = [t for t in trozos if t.volume() > 0.05 * max(x.volume() for x in trozos)]
    return min(grandes, key=distancia)


def _ajustar_caja(cuerpo, lo, hi, p, res, holgura, pasos=48):
    """Corrige la caja de la IA:
    - la cara de unión (la que toca el cuerpo, la más cercana al pivote) se acerca
      a la sección más estrecha, sin pasar del pivote: así una pata no se lleva
      la barriga;
    - las demás caras, si cortan la pieza (caja demasiado justa), crecen hasta
      abarcarla entera: así una cola fina no queda partida."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    piv = np.asarray(p.pivote, float)

    def fuera(a, lado, lo, hi, grosor=1.5):
        f_lo, f_hi = lo.copy(), hi.copy()
        if lado == 0:
            f_lo[a], f_hi[a] = lo[a] - grosor, lo[a]
        else:
            f_lo[a], f_hi[a] = hi[a], hi[a] + grosor
        return (cuerpo ^ _caja(f_lo, f_hi)).volume()

    caras = [(a, lado) for a in range(3) for lado in (0, 1)]
    con_cuerpo = [(a, l) for a, l in caras if fuera(a, l, lo, hi) > 1.0]
    union = min(con_cuerpo, key=lambda c: abs((lo if c[1] == 0 else hi)[c[0]] - piv[c[0]])) \
        if con_cuerpo else None

    # 1. Cara de unión: al cuello, si corta algo mucho más ancho que la pieza.
    if union is not None:
        a, lado = union
        orden = [(a + 1) % 3, (a + 2) % 3, a]
        m = np.zeros((3, 4))
        m[np.arange(3), orden] = 1.0
        rodajas = (cuerpo ^ _caja(lo, hi)).transform(m)
        zs = np.linspace(lo[a], hi[a], pasos + 2)[1:-1]
        areas = np.array([rodajas.slice(float(z)).area() for z in zs])
        if lado == 1:
            zs, areas = zs[::-1], areas[::-1]
        if areas.max() > 0:
            tipica = np.median(areas[areas > 0])
            tramo = areas[: int(pasos * 0.45)].copy()
            tramo[(zs[: len(tramo)] > piv[a] - 1.0) if lado == 0 else (zs[: len(tramo)] < piv[a] + 1.0)] = 0
            k = int(np.argmin(np.where(tramo > 0, tramo, np.inf)))
            if areas[0] > 1.3 * tipica and np.isfinite(tramo[k]) and tramo[k] < 0.6 * areas[0] and k > 0:
                antes = (lo if lado == 0 else hi)[a]
                (lo if lado == 0 else hi)[a] = zs[k]
                res.avisos.append(f"{p.nombre}: caja recortada en {'XYZ'[a]} de {antes:.1f} a "
                                  f"{zs[k]:.1f} mm (no coge parte del cuerpo).")

    # 2. Las demás caras crecen mientras corten la pieza.
    hija = _elegir_hija(cuerpo, lo, hi, holgura, piv)
    if hija is None:
        return lo, hi
    crecido = np.zeros((3, 2))
    for _ in range(8):
        cambio = False
        b = np.asarray(hija.bounding_box())
        for a, lado in caras:
            if (a, lado) == union or crecido[a, lado] >= 4:
                continue
            borde = (lo[a] + holgura) if lado == 0 else (hi[a] - holgura)
            if abs((b[a] if lado == 0 else b[a + 3]) - borde) > 0.3:
                continue                       # la pieza no llega a esa cara
            lo2, hi2 = lo.copy(), hi.copy()
            if lado == 0:
                lo2[a] -= 2.0
            else:
                hi2[a] += 2.0
            nueva = _elegir_hija(cuerpo, lo2, hi2, holgura, piv)
            if nueva is None or nueva.volume() > 1.35 * hija.volume() + 5:
                crecido[a, lado] = 99          # se metería en el cuerpo
                continue
            lo, hi, hija = lo2, hi2, nueva
            crecido[a, lado] += 2.0
            b = np.asarray(hija.bounding_box())
            cambio = True
        if not cambio:
            break
    for a, lado in caras:
        if 0 < crecido[a, lado] < 99:
            res.avisos.append(f"{p.nombre}: caja ampliada {crecido[a, lado]:.0f} mm en "
                              f"{'-+'[lado]}{'XYZ'[a]} para no partir la pieza.")
    return lo, hi


def _eje(p: PiezaMovil, hija, cuerpo, modo, holgura, diametro, res):
    """Pasador a lo largo del eje de la pieza, por su pivote."""
    import manifold3d as m3d

    ida, vuelta = _marco_plano(p.pivote, _vector(p))
    h, c = hija.transform(ida), cuerpo.transform(ida)
    tramos_h = [t for t in _intervalos(h) if t[0] <= 0.5 and t[1] >= -0.5] or _intervalos(h)
    if not tramos_h:
        # El pivote propuesto cae fuera de la pieza (la IA o el usuario se pasaron unos
        # mm): se acerca al trozo de la pieza más próximo al eje y se reintenta.
        v = np.asarray(h.to_mesh().vert_properties)[:, :3]
        if len(v):
            d = np.hypot(v[:, 0], v[:, 1])
            cerca = v[d <= d.min() + 2.0]
            centro = np.array([cerca[:, 0].mean(), cerca[:, 1].mean(), 0.0])
            if np.linalg.norm(centro) < 15.0:
                nuevo = vuelta[:, :3] @ centro + vuelta[:, 3]
                res.avisos.append(f"{p.nombre}: pivote movido {np.linalg.norm(centro):.1f} mm para "
                                  "que el eje atraviese la pieza.")
                p.pivote = [round(float(x), 2) for x in nuevo]
                ida, vuelta = _marco_plano(p.pivote, _vector(p))
                h, c = hija.transform(ida), cuerpo.transform(ida)
                tramos_h = [t for t in _intervalos(h) if t[0] <= 0.5 and t[1] >= -0.5] or \
                    _intervalos(h)
    if not tramos_h:
        res.avisos.append(f"{p.nombre}: el eje no pasa por la pieza; queda sin pasador.")
        return hija, cuerpo
    h0, h1 = min(tramos_h, key=lambda t: abs((t[0] + t[1]) / 2))
    tramos_c = _intervalos(c)
    abajo = [t for t in tramos_c if t[1] <= h0 + 0.1 and h0 - t[1] < 3 * holgura + 2]
    arriba = [t for t in tramos_c if t[0] >= h1 - 0.1 and t[0] - h1 < 3 * holgura + 2]
    if not abajo and not arriba:
        # Sin cuerpo pegado (p. ej. el brazo de una catapulta entre dos largueros
        # separados): eje largo de un lado a otro si hay cuerpo a ambos lados.
        lejos = 45.0
        abajo = [t for t in tramos_c if t[1] <= h0 + 0.1 and h0 - t[1] < lejos]
        arriba = [t for t in tramos_c if t[0] >= h1 - 0.1 and t[0] - h1 < lejos]
        if abajo and arriba:
            res.avisos.append(f"{p.nombre}: eje largo de {arriba[0][0] - abajo[-1][1]:.0f} mm "
                              "entre los dos lados del cuerpo.")
        else:
            abajo, arriba = [], []
    if not abajo and not arriba:
        # Se prueban pivotes cercanos dentro de la pieza (p. ej. más arriba en una pata,
        # donde ya toca el cuerpo) hasta que el eje tenga cuerpo al lado.
        cerca = 3 * holgura + 2
        rejilla = [(dx, dy) for dx in np.arange(-9, 9.1, 1.5) for dy in np.arange(-9, 9.1, 1.5)
                   if 0 < dx * dx + dy * dy <= 81]
        for dx, dy in sorted(rejilla, key=lambda d: d[0] ** 2 + d[1] ** 2):
            nuevo = vuelta[:, :3] @ np.array([dx, dy, 0.0]) + vuelta[:, 3]
            ida2, vuelta2 = _marco_plano(nuevo, _vector(p))
            h2 = hija.transform(ida2)
            tramos2 = _intervalos(h2)
            if not tramos2:
                continue
            g0, g1 = min(tramos2, key=lambda t: abs((t[0] + t[1]) / 2 - (h0 + h1) / 2))
            c2 = cuerpo.transform(ida2)
            tc = _intervalos(c2)
            ab = [t for t in tc if t[1] <= g0 + 0.1 and g0 - t[1] < cerca]
            ar = [t for t in tc if t[0] >= g1 - 0.1 and t[0] - g1 < cerca]
            if ab or ar:
                res.avisos.append(f"{p.nombre}: pivote movido {np.hypot(dx, dy):.1f} mm hasta donde "
                                  "la pieza toca el cuerpo.")
                p.pivote = [round(float(v), 2) for v in nuevo]
                ida, vuelta, h, c, h0, h1, abajo, arriba = ida2, vuelta2, h2, c2, g0, g1, ab, ar
                break
    if not abajo and not arriba:
        res.avisos.append(f"{p.nombre}: no hay cuerpo junto al eje donde apoyar el pasador.")
        return hija, cuerpo

    def cil(z0, z1, radio):
        a, b = min(z0, z1), max(z0, z1)
        return m3d.Manifold.cylinder(b - a, radio, radio, 40).translate((0, 0, a))

    # Radio: lo que quepa en la sección de la pieza alrededor del eje, con pared.
    pared = 1.2
    buje = False
    seccion = h.slice((h0 + h1) / 2)
    r = diametro / 2 if diametro > 0 else 2.0
    while r > 1.5 and (seccion.offset(-pared, m3d.JoinType.Round) ^ m3d.CrossSection.circle(
            r + holgura, 32)).area() < 0.95 * np.pi * (r + holgura) ** 2:
        r -= 0.25
    if (seccion.offset(-pared, m3d.JoinType.Round) ^ m3d.CrossSection.circle(r + holgura, 32)).area() \
            < 0.95 * np.pi * (r + holgura) ** 2:
        # Pieza fina (p. ej. un brazo plano): se le añade un buje cilíndrico alrededor
        # del eje, sin salirse de su caja, para que el pasador tenga pared. Con buje
        # el pasador recupera su grosor normal (uno de 2,5 mm se parte).
        r = diametro / 2 if diametro > 0 else 2.0
        lo, hi = np.minimum(p.caja_min, p.caja_max), np.maximum(p.caja_min, p.caja_max)
        dentro = _caja(lo + holgura, hi - holgura).transform(ida)
        # Solo en el tramo junto al cuerpo (en una cola fina, no a todo lo largo).
        buje = True
        largo_buje = 2 * pared + 2.6          # collarín corto, lo justo para el pasador
        if abajo:
            b0, b1 = h0, min(h1, h0 + largo_buje)
        else:
            b0, b1 = max(h0, h1 - largo_buje), h1
        h = h + (cil(b0, b1, r + holgura + pared) ^ dentro)
        res.avisos.append(f"{p.nombre}: se añadió un buje de {2 * (r + holgura + pared):.1f} mm "
                          "alrededor del eje para el pasador.")

    if bool(abajo) != bool(arriba):
        # Apoyado por un solo lado: el pasador entra en la pieza solo lo necesario
        # (el collarín, o 10 mm), no a lo largo de toda una cola o una cabeza.
        penetra = (b1 - b0) if buje else 10.0
        if abajo:
            h1 = min(h1, h0 + penetra)
        else:
            h0 = max(h0, h1 - penetra)

    entra = 3.0
    z_ini = abajo[-1][1] - min(entra, abajo[-1][1] - abajo[-1][0]) if abajo else None
    z_fin = arriba[0][0] + min(entra, arriba[0][1] - arriba[0][0]) if arriba else None

    if modo == "separadas":
        ini = z_ini if z_ini is not None else h0 + 0.8
        fin = z_fin if z_fin is not None else h1 - 0.8
        h = h - cil(h0 - 1, h1 + 1, r + holgura)
        c = c - cil(ini - 0.2, fin + 0.2, r + 0.05)          # a presión en el cuerpo
        largo = fin - ini
        res.pasadores.append(_a_trimesh(m3d.Manifold.cylinder(largo, r, r, 40)))
        res.avisos.append(f"{p.nombre}: pasador suelto de {2 * r:.1f} x {largo:.1f} mm." +
                          (" Muy corto para sujetar: mejor en modo montada." if largo < 4 else ""))
        return h.transform(vuelta), c.transform(vuelta)

    # Montada: el pasador sale del cuerpo y atraviesa la pieza con holgura.
    if z_ini is not None and z_fin is not None:          # apoyado por los dos lados
        c = c + cil(z_ini, z_fin, r)
        h = h - cil(h0 - 1, h1 + 1, r + holgura)
    else:                                                 # por un lado: pestaña que retiene
        pared, grosor, r_pestana = 1.2, 1.5, r + 1.0
        if z_ini is not None:
            fin = h1 - pared
            c = c + cil(z_ini, fin, r) + cil(fin - grosor, fin, r_pestana)
            h = h - cil(h0 - 1, fin + holgura, r + holgura) - \
                cil(fin - grosor - holgura, fin + holgura, r_pestana + holgura)
        else:
            ini = h0 + pared
            c = c + cil(ini, z_fin, r) + cil(ini, ini + grosor, r_pestana)
            h = h - cil(ini - holgura, h1 + 1, r + holgura) - \
                cil(ini - holgura, ini + grosor + holgura, r_pestana + holgura)
    return h.transform(vuelta), c.transform(vuelta)


# --- 3. Salidas --------------------------------------------------------------------

def exportar(mec: Mecanismo, base: str, modo: str) -> dict:
    """Escribe <base>_mecanismo.glb (animado), .zip y .3mf para imprimir."""
    import segmentar

    rutas = {"glb": base + "_mecanismo.glb", "zip": base + "_mecanismo.zip",
             "3mf": base + "_mecanismo.3mf"}
    animacion.exportar_glb(rutas["glb"], mec.partes)
    res = Resultado(partes=[a.malla for a in mec.partes], pasadores=mec.pasadores,
                    nombres=[a.nombre for a in mec.partes])
    segmentar.exportar_zip(rutas["zip"], res)
    if modo == "montada":
        # Print-in-place: las piezas van en su sitio, ya montadas, en un solo 3MF.
        montado = Resultado(partes=[trimesh.util.concatenate([a.malla for a in mec.partes])],
                            nombres=["mecanismo montado"])
        segmentar.exportar_3mf(rutas["3mf"], montado)
    else:
        segmentar.exportar_3mf(rutas["3mf"], res)
    return rutas
