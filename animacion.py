"""Exporta un modelo por partes como .glb articulado y animado.

Cada parte móvil es un nodo con su pivote (el centro de la articulación) y una
animación glTF de rotación alrededor de su eje:

- "giro": vueltas completas (ruedas, engranajes, hélices);
- "bisagra": va y vuelve entre dos ángulos (brazo de catapulta, mandíbula,
  puerta);
- "fija": no se mueve (el cuerpo principal).

El .glb se puede abrir en Blender (con sus animaciones y la jerarquía para
posar), en visores web o en la propia app.
"""
from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class Articulacion:
    nombre: str
    malla: trimesh.Trimesh       # en mm, Z arriba (malla de impresión)
    tipo: str = "fija"           # "fija", "giro" o "bisagra"
    pivote: tuple = (0, 0, 0)    # punto del eje (mm)
    eje: tuple = (1, 0, 0)       # dirección del eje
    angulo_min: float = 0.0      # grados (bisagra)
    angulo_max: float = 60.0
    vueltas: float = 1.0         # giro: vueltas por ciclo
    color: tuple = (200, 200, 200)


def _cuaternion(eje, angulo_grados):
    """Cuaternión (x, y, z, w) de una rotación alrededor de `eje`."""
    e = np.asarray(eje, float)
    e = e / np.linalg.norm(e)
    a = np.radians(angulo_grados) / 2
    return [*(e * np.sin(a)), float(np.cos(a))]


def exportar_glb(ruta: str, partes: list, duracion: float = 2.0, fotogramas: int = 49,
                 y_arriba: bool = True):
    """Escribe el .glb animado. Las animaciones duran `duracion` segundos y se
    repiten en bucle en la mayoría de visores."""
    import pygltflib as g

    # 1. Escena con un nodo por parte, la malla centrada en su pivote. Todo se
    # pasa antes a Y arriba (convención glTF) y los nodos solo llevan traslación
    # (glTF no permite animar nodos definidos con matriz).
    escena = trimesh.Scene()
    giro = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]) if y_arriba else np.eye(4)
    pivotes = {}
    for p in partes:
        m = p.malla.copy()
        m.apply_transform(giro)
        pivote = trimesh.transform_points([p.pivote], giro)[0]
        m.apply_translation(-pivote)
        m.visual.face_colors = (*p.color, 255)
        pivotes[p.nombre] = pivote
        escena.add_geometry(m, node_name=p.nombre, geom_name=p.nombre,
                            transform=trimesh.transformations.translation_matrix(pivote))
    gltf = g.GLTF2().load_from_bytes(trimesh.exchange.gltf.export_glb(escena))
    for nodo in gltf.nodes:
        if nodo.name in pivotes:
            nodo.matrix = None
            nodo.translation = [float(v) for v in pivotes[nodo.name]]

    # 2. Animaciones: rotación de cada nodo móvil alrededor de su eje.
    tiempos = np.linspace(0, duracion, fotogramas).astype(np.float32)
    datos = bytearray(gltf.binary_blob() or b"")
    canales, muestreadores = [], []

    def accesor(array, tipo, con_minmax=False):
        nonlocal datos
        while len(datos) % 4:
            datos.append(0)
        inicio = len(datos)
        datos += array.astype(np.float32).tobytes()
        gltf.bufferViews.append(g.BufferView(buffer=0, byteOffset=inicio, byteLength=array.nbytes))
        acc = g.Accessor(bufferView=len(gltf.bufferViews) - 1, componentType=g.FLOAT,
                         count=len(array), type=tipo)
        if con_minmax:
            acc.min, acc.max = [float(array.min())], [float(array.max())]
        gltf.accessors.append(acc)
        return len(gltf.accessors) - 1

    acc_tiempo = accesor(tiempos, g.SCALAR, con_minmax=True)
    nodos = {n.name: i for i, n in enumerate(gltf.nodes)}
    for p in partes:
        if p.tipo == "fija" or p.nombre not in nodos:
            continue
        eje = (giro[:3, :3] @ np.asarray(p.eje, float))
        t = np.linspace(0, 1, fotogramas)
        if p.tipo == "giro":
            angulos = 360.0 * p.vueltas * t
        else:  # bisagra: sube rápido (lanzamiento) y vuelve despacio
            fase = np.where(t < 0.25, t / 0.25, 1 - (t - 0.25) / 0.75)
            suave = 0.5 - 0.5 * np.cos(np.pi * fase)
            angulos = p.angulo_min + (p.angulo_max - p.angulo_min) * suave
        cuats = np.array([_cuaternion(eje, a) for a in angulos], np.float32)
        acc_rot = accesor(cuats, g.VEC4)
        muestreadores.append(g.AnimationSampler(input=acc_tiempo, output=acc_rot,
                                                interpolation="LINEAR"))
        canales.append(g.AnimationChannel(sampler=len(muestreadores) - 1,
                                          target=g.AnimationChannelTarget(node=nodos[p.nombre],
                                                                          path="rotation")))
    if canales:
        gltf.animations.append(g.Animation(name="mecanismo", samplers=muestreadores,
                                           channels=canales))
    gltf.buffers[0].byteLength = len(datos)
    gltf.set_binary_blob(bytes(datos))
    gltf.save_binary(ruta)
