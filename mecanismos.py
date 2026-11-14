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
    for p in piezas:
        lo, hi = np.minimum(p.caja_min, p.caja_max), np.maximum(p.caja_min, p.caja_max)
        lo, hi = _recortar_caja(cuerpo, lo, hi, p, res)
        interior = cuerpo ^ _caja(lo + holgura, hi - holgura)
        trozos = sorted(interior.decompose(), key=lambda t: -t.volume())
        if not trozos:
            res.avisos.append(f"{p.nombre}: la caja no contiene nada del modelo; se omite.")
            continue
        marca = m3d.Manifold.sphere(1.5, 12).translate(tuple(p.pivote))
        con_pivote = [t for t in trozos if not (t ^ marca).is_empty()]
        hija = (con_pivote or trozos)[0]
        # Se quita del cuerpo la caja completa alrededor de la pieza; la diferencia
        # con la caja interior deja una holgura en todas sus caras.
        b = np.asarray(hija.bounding_box())
        cuerpo = cuerpo - (_caja(lo, hi) ^ _caja(b[:3] - holgura - 0.5, b[3:] + holgura + 0.5))
        moviles.append((p, hija))

    for i, (p, hija) in enumerate(moviles):
        hija, cuerpo = _eje(p, hija, cuerpo, modo, holgura, diametro, res)
        # Fuera migas sueltas (p. ej. un adorno del buje que el agujero dejó aislado).
        trozos = sorted(hija.decompose(), key=lambda t: -t.volume())
        hija = m3d.Manifold.batch_boolean([t for t in trozos if t.volume() > 0.01 * trozos[0].volume()],
                                          m3d.OpType.Add)
        res.partes.append(animacion.Articulacion(
            p.nombre, _a_trimesh(hija), "giro" if p.tipo == "giro" else "bisagra",
            tuple(p.pivote), EJES[p.eje], p.angulo_min, p.angulo_max,
            movimiento=p.movimiento, fase=float(p.fase), color=COLORES[i % len(COLORES)]))
    trozos = sorted(cuerpo.decompose(), key=lambda t: -t.volume())
    cuerpo = m3d.Manifold.batch_boolean([t for t in trozos if t.volume() > 0.002 * trozos[0].volume()],
                                        m3d.OpType.Add)
    res.partes.insert(0, animacion.Articulacion("cuerpo", _a_trimesh(cuerpo), "fija",
                                                color=(210, 210, 210)))
    return res


def _recortar_caja(cuerpo, lo, hi, p, res, pasos=48):
    """Si una cara de la caja corta algo mucho más ancho que la pieza (la caja de una
    pata que se mete en la barriga), la acerca hasta la sección más estrecha cercana:
    el "cuello" donde la pieza se une al cuerpo. Después mete el pivote en la caja."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    interior = cuerpo ^ _caja(lo, hi)
    if interior.is_empty():
        return lo, hi
    for a in range(3):
        if a == "xyz".index(p.eje) and p.tipo == "giro":
            continue                     # a lo largo del eje de una rueda no se recorta
        # Matriz que lleva el eje `a` a Z para cortar en rodajas.
        orden = [(a + 1) % 3, (a + 2) % 3, a]
        m = np.zeros((3, 4))
        m[np.arange(3), orden] = 1.0
        rodajas = interior.transform(m)
        zs = np.linspace(lo[a], hi[a], pasos + 2)[1:-1]
        areas = np.array([rodajas.slice(float(z)).area() for z in zs])
        if areas.max() <= 0:
            continue
        tipica = np.median(areas[areas > 0])
        for lado in (0, 1):
            serie = areas if lado == 0 else areas[::-1]
            if serie[0] < 1.3 * tipica:  # la cara no corta nada ancho
                continue
            # Solo si al otro lado de la cara sigue habiendo cuerpo (no en un pie
            # apoyado en el suelo, que simplemente es más ancho que la pata).
            fuera_lo, fuera_hi = lo.copy(), hi.copy()
            if lado == 0:
                fuera_lo[a], fuera_hi[a] = lo[a] - 1.0, lo[a]
            else:
                fuera_lo[a], fuera_hi[a] = hi[a], hi[a] + 1.0
            if (cuerpo ^ _caja(fuera_lo, fuera_hi)).volume() < 0.3 * serie[0]:
                continue
            tramo = serie[: int(pasos * 0.45)].copy()
            if a != "xyz".index(p.eje):
                # Nunca más allá del pivote: ahí está la articulación.
                zz = (zs if lado == 0 else zs[::-1])[: len(tramo)]
                piv = float(p.pivote[a])
                tramo[(zz > piv - 1.5) if lado == 0 else (zz < piv + 1.5)] = 0
            k = int(np.argmin(np.where(tramo > 0, tramo, np.inf)))
            if not np.isfinite(tramo[k]) or tramo[k] > 0.6 * serie[0] or k == 0:
                continue
            z = (zs if lado == 0 else zs[::-1])[k]
            antes = lo[a] if lado == 0 else hi[a]
            if lado == 0:
                lo[a] = z
            else:
                hi[a] = z
            res.avisos.append(f"{p.nombre}: caja recortada en {'XYZ'[a]} de {antes:.1f} a "
                              f"{z:.1f} mm (no coge parte del cuerpo).")
            interior = cuerpo ^ _caja(lo, hi)
            rodajas = interior.transform(m)
    # El pivote, dentro de la caja (salvo a lo largo de su eje, donde da igual).
    piv = np.array(p.pivote, float)
    eje = "xyz".index(p.eje)
    for a in range(3):
        if a != eje:
            margen = min(1.5, (hi[a] - lo[a]) / 4)
            piv[a] = np.clip(piv[a], lo[a] + margen, hi[a] - margen)
    if np.linalg.norm(piv - np.asarray(p.pivote, float)) > 0.5:
        res.avisos.append(f"{p.nombre}: pivote metido en la caja ({', '.join(f'{v:.1f}' for v in piv)}).")
    p.pivote = [round(float(v), 2) for v in piv]
    return lo, hi


def _eje(p: PiezaMovil, hija, cuerpo, modo, holgura, diametro, res, probar_ejes=True):
    """Pasador a lo largo del eje de la pieza, por su pivote."""
    import manifold3d as m3d

    if probar_ejes:
        pivote, eje, n = list(p.pivote), p.eje, len(res.avisos)
        h, c = _eje(p, hija, cuerpo, modo, holgura, diametro, res, False)
        if not res.avisos[-1:] or "no hay cuerpo" not in res.avisos[-1] or p.tipo != "bisagra":
            return h, c
        # Con ese eje el pasador no tiene dónde apoyarse (una cola que sale del lomo
        # no tiene cuerpo a los lados): se prueban los otros ejes.
        for otro in "xyz".replace(eje, ""):
            prueba = Mecanismo()
            p.eje, p.pivote = otro, list(pivote)
            h, c = _eje(p, hija, cuerpo, modo, holgura, diametro, prueba, False)
            if not any("no hay cuerpo" in a or "no pasa" in a for a in prueba.avisos):
                del res.avisos[n:]
                res.avisos.append(f"{p.nombre}: con el eje {eje.upper()} no había dónde apoyarlo; "
                                  f"gira sobre {otro.upper()}.")
                res.avisos += prueba.avisos
                res.pasadores += prueba.pasadores
                return h, c
        p.eje, p.pivote = eje, pivote
        return hija, cuerpo

    ida, vuelta = _marco_plano(p.pivote, EJES[p.eje])
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
                ida, vuelta = _marco_plano(p.pivote, EJES[p.eje])
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
            ida2, vuelta2 = _marco_plano(nuevo, EJES[p.eje])
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
    seccion = h.slice((h0 + h1) / 2)
    r = diametro / 2 if diametro > 0 else 2.0
    while r > 1.25 and (seccion.offset(-pared, m3d.JoinType.Round) ^ m3d.CrossSection.circle(
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
        h = h + (cil(h0, h1, r + holgura + pared) ^ dentro)
        res.avisos.append(f"{p.nombre}: se añadió un buje de {2 * (r + holgura + pared):.1f} mm "
                          "alrededor del eje para el pasador.")

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
        res.avisos.append(f"{p.nombre}: pasador suelto de {2 * r:.1f} x {largo:.1f} mm.")
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
