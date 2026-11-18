"""Nodo auxiliar (sin nodos visibles) para depurar la textura en AMD.

Cuando ComfyUI ejecuta "Bake Texture From Voxel", guarda sus entradas reales
(malla con UV + voxeles de color) en ComfyUI/output/amd_horneado.npz para poder
analizarlas despues con prueba_horneado_real.py, sin repetir la generacion.
Se instala copiandolo a ComfyUI/custom_nodes/. Para quitarlo, borra el archivo.
"""
import logging
import os
import sys
import time

import numpy as np

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _carpeta_salida():
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def _a_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy() if x.is_floating_point() else x.detach().cpu().numpy()
    return np.asarray(x)


def _envolver(modulo):
    original = modulo.bake_texture_from_voxel_fn
    if getattr(original, "_amd_volcado", False):
        return

    def envuelta(vertices, faces, voxel_coords, voxel_colors, resolution, texture_size, uvs, *args, **kwargs):
        try:
            ruta = os.path.join(_carpeta_salida(), "amd_horneado.npz")
            t0 = time.time()
            datos = dict(vertices=_a_numpy(vertices), faces=_a_numpy(faces), uvs=_a_numpy(uvs),
                         voxel_coords=_a_numpy(voxel_coords), voxel_colors=_a_numpy(voxel_colors),
                         resolution=np.int64(resolution), texture_size=np.int64(texture_size))
            ref = kwargs.get("reference")
            if ref is None and len(args) >= 2:
                ref = args[1]
            if ref is not None:
                datos["ref_vertices"] = _a_numpy(ref[0])
                datos["ref_faces"] = _a_numpy(ref[1])
            np.savez(ruta, **datos)
            logging.info(f"[AMD] entradas del horneado guardadas en {ruta} ({time.time() - t0:.1f}s)")
        except Exception as e:
            logging.warning(f"[AMD] no se pudo guardar el volcado del horneado: {e}")
        return original(vertices, faces, voxel_coords, voxel_colors, resolution, texture_size, uvs, *args, **kwargs)

    envuelta._amd_volcado = True
    modulo.bake_texture_from_voxel_fn = envuelta
    logging.info(f"[AMD] volcado del horneado activo ({modulo.__name__})")


_hecho = False
for _m in list(sys.modules.values()):
    _d = getattr(_m, "__dict__", None) if isinstance(_m, type(sys)) else None
    if isinstance(_d, dict) and "bake_texture_from_voxel_fn" in _d and "BakeTextureFromVoxel" in _d:
        _envolver(_m)
        _hecho = True
if not _hecho:
    try:
        import comfy_extras.nodes_mesh_postprocess as _mp
        _envolver(_mp)
    except Exception as e:
        logging.warning(f"[AMD] no encontre el modulo de horneado: {e}")
