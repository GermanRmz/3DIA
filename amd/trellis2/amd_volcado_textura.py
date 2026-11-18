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


# ---------------------------------------------------------------------------
# Diagnostico del decodificador de textura (solo con AMD_DIAG_TEXTURA=1).
# Ejecuta el decodificador en GPU registrando cada capa, despues repite la misma
# decodificacion en CPU y compara capa por capa. Escribe el informe en el log y en
# ComfyUI/output/amd_diag_textura.txt. La salida real del workflow es la de la GPU.
# ---------------------------------------------------------------------------
def _feats_de(salida):
    if isinstance(salida, (tuple, list)) and salida:
        salida = salida[0]
    if hasattr(salida, "feats"):
        salida = salida.feats
    return salida if hasattr(salida, "detach") and hasattr(salida, "is_floating_point") else None


def _filas_muestra(n, k=1024):
    import torch
    if n <= k:
        return torch.arange(n)
    return torch.linspace(0, n - 1, k).long()


def _registrar(modelo, registro, muestras):
    ganchos = []
    for nombre, mod in modelo.named_modules():
        if not nombre:
            continue

        def gancho(m, entrada, salida, nombre=nombre):
            t = _feats_de(salida)
            if t is None or not t.is_floating_point() or t.numel() == 0:
                return
            t2 = t.detach().reshape(t.shape[0], -1).float()
            finito = torch_isfinite(t2)
            maximo = float(t2.abs().nan_to_num(posinf=3e38).max())
            idx = _filas_muestra(t2.shape[0]).to(t2.device)
            muestras.append((nombre, tuple(t.shape), t2[idx].cpu()))
            registro.append((nombre, tuple(t.shape), finito, maximo))

        ganchos.append(mod.register_forward_hook(gancho))
    return ganchos


def torch_isfinite(t):
    import torch
    return float(torch.isfinite(t).float().mean())


def _diagnosticar(vae, slat, subs, original):
    import torch
    import comfy.model_management as mm
    from comfy.ldm.trellis2.vae import SparseTensor

    def copia(st, dev, dtype=None):
        f = st.feats.to(dev)
        if dtype is not None:
            f = f.to(dtype)
        return SparseTensor(feats=f, coords=st.coords.to(dev), shape=st.shape, scale=st._scale)

    lineas = []

    def log(s):
        lineas.append(s)
        logging.info("[AMD diag] " + s)

    dev = slat.feats.device
    slat_cpu = copia(slat, "cpu", torch.float32)
    subs_cpu = [copia(s, "cpu") for s in subs]
    log(f"entrada: {tuple(slat.feats.shape)} {slat.feats.dtype} en {dev}, finitos {torch_isfinite(slat.feats.float()):.4f}, "
        f"max {float(slat.feats.float().abs().max()):.3f}; subs {[tuple(s.feats.shape) for s in subs]}")

    reg_g, mue_g = [], []
    ganchos = _registrar(vae.txt_dec, reg_g, mue_g)
    t0 = time.time()
    try:
        salida = original(vae, slat, subs)
        torch.cuda.synchronize() if dev.type == "cuda" else None
    finally:
        for g in ganchos:
            g.remove()
    log(f"GPU: {time.time() - t0:.1f}s, {len(reg_g)} capas registradas")
    malas = [r for r in reg_g if r[2] < 1.0 or r[3] > 1e4]
    if malas:
        n, s, fi, mx = malas[0]
        log(f"primera capa con valores raros en GPU: {n} {s} finitos {fi:.4f} max {mx:.3e}")
    else:
        log("GPU: ninguna capa con NaN/inf ni valores > 1e4")

    try:
        dtype_orig = next(vae.parameters()).dtype
        vae.to("cpu", torch.float32)
        reg_c, mue_c = [], []
        ganchos = _registrar(vae.txt_dec, reg_c, mue_c)
        t0 = time.time()
        try:
            with torch.no_grad():
                original(vae, slat_cpu, subs_cpu)
        finally:
            for g in ganchos:
                g.remove()
        log(f"CPU: {time.time() - t0:.1f}s, {len(reg_c)} capas registradas")
        primera = None
        for (ng, sg, xg), (nc, sc, xc) in zip(mue_g, mue_c):
            if ng != nc or sg != sc:
                log(f"orden distinto: GPU {ng} {sg} / CPU {nc} {sc}")
                break
            d = (xg - xc).abs().nan_to_num(nan=3e38, posinf=3e38)
            rel = float(d.max() / xc.abs().max().clamp_min(1e-6))
            if rel > 0.05 and primera is None:
                primera = (ng, sg, rel)
                log(f"PRIMERA CAPA DISTINTA GPU vs CPU: {ng} {sg} error relativo {rel:.3e}")
                filas_mal = (d.max(1).values > 0.05 * xc.abs().max()).float().mean()
                log(f"   filas de la muestra afectadas: {float(filas_mal):.2%}")
        if primera is None:
            log("GPU y CPU coinciden en todas las capas (error relativo < 5%)")
        for (n, s, fi, mx), (_, _, fic, mxc) in zip(reg_g, reg_c):
            lineas.append(f"   {n:60s} {str(s):22s} GPU fin {fi:.4f} max {mx:10.3e} | CPU fin {fic:.4f} max {mxc:10.3e}")
    except Exception as e:
        log(f"no se pudo ejecutar en CPU: {type(e).__name__}: {e}")
    finally:
        vae.to(dev, dtype_orig)
        mm.soft_empty_cache()

    try:
        with open(os.path.join(_carpeta_salida(), "amd_diag_textura.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lineas) + "\n")
        logging.info("[AMD diag] informe completo en output\\amd_diag_textura.txt")
    except Exception as e:
        logging.warning(f"[AMD diag] no se pudo escribir el informe: {e}")
    return salida


if os.environ.get("AMD_DIAG_TEXTURA"):
    try:
        import comfy.ldm.trellis2.vae as _vae_mod
        _orig_tex = _vae_mod.TextureVae.decode_tex_slat
        if not getattr(_orig_tex, "_amd_diag", False):
            def _decode_diag(self, slat, subs):
                return _diagnosticar(self, slat, subs, _orig_tex)
            _decode_diag._amd_diag = True
            _vae_mod.TextureVae.decode_tex_slat = _decode_diag
            logging.info("[AMD] diagnostico del decodificador de textura ACTIVO (tarda varios minutos mas)")
    except Exception as e:
        logging.warning(f"[AMD] no se pudo activar el diagnostico de textura: {e}")
