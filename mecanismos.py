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
                    "explicacion": {"type": "string"},
                },
                "required": ["nombre", "tipo", "caja_min", "caja_max", "pivote", "eje",
                             "angulo_min", "angulo_max", "explicacion"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["resumen", "piezas"],
    "additionalProperties": False,
}

INSTRUCCIONES = """Eres un ingeniero de mecanismos para impresión 3D. Te paso la imagen original \
de un objeto y tres vistas ortográficas de su modelo 3D con cuadrícula en milímetros.

Sistema de coordenadas (mm): Z hacia arriba, el frente del objeto mira hacia -Y.
- Vista "frente": horizontal = X, vertical = Z.
- Vista "lado": horizontal = Y, vertical = Z.
- Vista "arriba": horizontal = X, vertical = Y.
Medidas del modelo: {medidas}.

Identifica las piezas que en el objeto real se moverían para que funcione (por ejemplo \
ruedas que giran sobre su eje, el brazo de una catapulta que bascula, una puerta, una \
mandíbula, una hélice). Para cada una:
- tipo: "giro" si da vueltas completas (ruedas, hélices), "bisagra" si oscila entre dos \
ángulos (brazos, puertas, tapas).
- caja_min y caja_max: [x, y, z] de una caja que contenga SOLO esa pieza, lo más ajustada \
posible, sin incluir el cuerpo al que va unida (deja el eje o soporte fuera de la caja).
- pivote: [x, y, z] del centro de la articulación (el centro del eje de la rueda, el eje \
del brazo).
- eje: el eje de giro, "x", "y" o "z".
- angulo_min y angulo_max en grados para la bisagra (0 = posición del modelo; el signo \
sigue la regla de la mano derecha sobre el eje positivo). Para "giro", 0 y 360.
Incluye solo piezas que se puedan separar de forma razonable. Lee las coordenadas en las \
cuadrículas con cuidado. Responde en español.{descripcion}"""


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


def analizar(malla: trimesh.Trimesh, imagen_original=None, descripcion: str = "") -> tuple[list, str]:
    """Pide a Claude las piezas móviles. Devuelve (lista de PiezaMovil, resumen)."""
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

    cliente = anthropic.Anthropic()
    respuesta = cliente.beta.messages.create(
        model=MODELO_IA,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                 # si el modelo declina, otro lo intenta
        output_config={"effort": "high",
                       "format": {"type": "json_schema", "schema": ESQUEMA}},
        messages=[{"role": "user", "content": contenido}],
    )
    if respuesta.stop_reason == "refusal":
        raise RuntimeError("La IA no quiso analizar este modelo.")
    if respuesta.stop_reason == "max_tokens":
        raise RuntimeError("La respuesta de la IA se cortó; inténtalo de nuevo.")
    datos = json.loads(next(b.text for b in respuesta.content if b.type == "text"))
    return [PiezaMovil(**p) for p in datos["piezas"]], datos["resumen"]


def a_json(piezas: list, resumen: str = "") -> str:
    return json.dumps({"resumen": resumen, "piezas": [asdict(p) for p in piezas]},
                      ensure_ascii=False, indent=2)


def desde_json(texto: str) -> list:
    datos = json.loads(texto)
    return [PiezaMovil(**p) for p in datos.get("piezas", [])]


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
            color=COLORES[i % len(COLORES)]))
    trozos = sorted(cuerpo.decompose(), key=lambda t: -t.volume())
    cuerpo = m3d.Manifold.batch_boolean([t for t in trozos if t.volume() > 0.002 * trozos[0].volume()],
                                        m3d.OpType.Add)
    res.partes.insert(0, animacion.Articulacion("cuerpo", _a_trimesh(cuerpo), "fija",
                                                color=(210, 210, 210)))
    return res


def _eje(p: PiezaMovil, hija, cuerpo, modo, holgura, diametro, res):
    """Pasador a lo largo del eje de la pieza, por su pivote."""
    import manifold3d as m3d

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
