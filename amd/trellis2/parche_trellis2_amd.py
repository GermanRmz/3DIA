"""Adapta ComfyUI-Trellis2 (visualbruno) para GPUs AMD con ROCm en Windows.

Uso (con el Python del entorno, desde la carpeta del runtime):
    python_embeded\\python.exe parche_trellis2_amd.py ComfyUI\\custom_nodes\\ComfyUI-Trellis2

Que hace (se puede ejecutar varias veces sin problema):
1. Instala un sustituto de ``nvdiffrast`` en PyTorch puro (nvdiffrast solo
   existe para NVIDIA). Lo toma de ``nvdiffrast_torch.py``, junto a este script.
2. nodes.py: ``import triton`` pasa a ser opcional (el nodo no lo usa).
3. Atencion densa y dispersa: si flash_attn/xformers no estan instalados,
   se usa SDPA de PyTorch. La atencion por ventanas recibe una version SDPA.
4. Convolucion dispersa: si flex_gemm no tiene Triton, usa ``explicit_gemm``.
Guarda copia ``.orig`` de cada archivo antes de cambiarlo.
"""
import os
import shutil
import sys
import sysconfig

MARCA = "# --- parche AMD (3DIA) ---"

DISPONIBLE = '''
def _amd_disponible(backend):
    modulo = {"flash_attn": "flash_attn", "flash_attn_3": "flash_attn_interface",
              "xformers": "xformers.ops"}.get(backend)
    if modulo is None:
        return True
    try:
        __import__(modulo)
        return True
    except Exception:
        return False
'''

PARCHE_CONFIG_DENSA = MARCA + DISPONIBLE + '''
_amd_set_backend_original = set_backend


def set_backend(backend):
    if not _amd_disponible(backend):
        print(f"[ATTENTION] {backend} no esta disponible en AMD; se usa sdpa")
        backend = "sdpa"
    _amd_set_backend_original(backend)


if not _amd_disponible(BACKEND):
    BACKEND = "sdpa"
'''

PARCHE_CONFIG_DISPERSA = MARCA + DISPONIBLE + '''
_amd_set_attn_backend_original = set_attn_backend


def set_attn_backend(backend):
    if not _amd_disponible(backend):
        print(f"[SPARSE] {backend} no esta disponible en AMD; se usa sdpa")
        backend = "sdpa"
    _amd_set_attn_backend_original(backend)


if not _amd_disponible(ATTN):
    ATTN = "sdpa"
'''

PARCHE_VENTANAS = MARCA + '''
import torch.nn.functional as _amd_F

_amd_self_original = sparse_windowed_scaled_dot_product_self_attention
_amd_cross_original = sparse_windowed_scaled_dot_product_cross_attention
_AMD_TOKENS_POR_BLOQUE = 262144


def _amd_usar_sdpa():
    if config.ATTN == "xformers":
        try:
            import xformers.ops  # noqa: F401
            return False
        except Exception:
            return True
    if config.ATTN == "flash_attn":
        try:
            import flash_attn  # noqa: F401
            return False
        except Exception:
            return True
    return True


def _amd_particion(tensor, window_size, shift_window):
    nombre = f"amd_windowed_{window_size}_{shift_window}"
    cache = tensor.get_spatial_cache(nombre)
    if cache is not None:
        return cache
    DIM = tensor.coords.shape[1] - 1
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    coords = tensor.coords.clone().detach()
    coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    MAX_COORDS = [i + j for i, j in zip(tensor.spatial_shape, shift_window)]
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]
    coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    indices = (coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)).sum(dim=1)
    fwd = torch.argsort(indices)
    bwd = torch.empty_like(fwd)
    bwd[fwd] = torch.arange(fwd.shape[0], device=tensor.device)
    lens = torch.bincount(indices)
    lens = lens[lens != 0]
    resultado = (fwd, bwd, lens)
    tensor.register_spatial_cache(nombre, resultado)
    return resultado


def _amd_rellenar(x, lens, inicios, w0, w1, L):
    n = w1 - w0
    sub = lens[w0:w1]
    a, b = int(inicios[w0]), int(inicios[w0] + sub.sum())
    ventana = torch.repeat_interleave(torch.arange(n, device=x.device), sub)
    pos = torch.arange(b - a, device=x.device) - torch.repeat_interleave(inicios[w0:w1] - inicios[w0], sub)
    salida = x.new_zeros((n, L) + tuple(x.shape[1:]))
    salida[ventana, pos] = x[a:b]
    return salida, ventana, pos


def _amd_atencion_ventanas(q, k, v, q_lens, kv_lens):
    # q: [Mq, H, C]; k, v: [Mk, H, C]; una secuencia por ventana.
    q_lens = q_lens.to(q.device).long()
    kv_lens = kv_lens.to(q.device).long()
    q_ini = torch.cumsum(q_lens, 0) - q_lens
    kv_ini = torch.cumsum(kv_lens, 0) - kv_lens
    Lq, Lk = int(q_lens.max()), int(kv_lens.max())
    por_bloque = max(1, _AMD_TOKENS_POR_BLOQUE // max(Lq, Lk))
    salida = q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
    nW = q_lens.shape[0]
    for w0 in range(0, nW, por_bloque):
        w1 = min(nW, w0 + por_bloque)
        qp, qv, qpos = _amd_rellenar(q, q_lens, q_ini, w0, w1, Lq)
        kp, _, _ = _amd_rellenar(k, kv_lens, kv_ini, w0, w1, Lk)
        vp, _, _ = _amd_rellenar(v, kv_lens, kv_ini, w0, w1, Lk)
        mascara = torch.arange(Lk, device=q.device)[None, :] < kv_lens[w0:w1, None]
        out = _amd_F.scaled_dot_product_attention(
            qp.transpose(1, 2), kp.transpose(1, 2), vp.transpose(1, 2),
            attn_mask=mascara[:, None, None, :],
        ).transpose(1, 2)                                   # [n, Lq, H, C]
        a = int(q_ini[w0])
        salida[a:a + qv.shape[0]] = out[qv, qpos]
    return salida


def sparse_windowed_scaled_dot_product_self_attention(qkv, window_size, shift_window=(0, 0, 0)):
    if not _amd_usar_sdpa():
        return _amd_self_original(qkv, window_size, shift_window)
    fwd, bwd, lens = _amd_particion(qkv, window_size, shift_window)
    q, k, v = qkv.feats[fwd].unbind(dim=1)
    out = _amd_atencion_ventanas(q, k, v, lens, lens)
    return qkv.replace(out[bwd])


def sparse_windowed_scaled_dot_product_cross_attention(q, kv, q_window_size, kv_window_size,
                                                       q_shift_window=(0, 0, 0), kv_shift_window=(0, 0, 0)):
    if not _amd_usar_sdpa():
        return _amd_cross_original(q, kv, q_window_size, kv_window_size, q_shift_window, kv_shift_window)
    q_fwd, q_bwd, q_lens = _amd_particion(q, q_window_size, q_shift_window)
    kv_fwd, _, kv_lens = _amd_particion(kv, kv_window_size, kv_shift_window)
    assert len(q_lens) == len(kv_lens), "Number of sequences in q and kv must match"
    k, v = kv.feats[kv_fwd].unbind(dim=1)
    out = _amd_atencion_ventanas(q.feats[q_fwd], k, v, q_lens, kv_lens)
    return q.replace(out[q_bwd])
'''

PARCHE_CONV = MARCA + '''
try:
    import flex_gemm.kernels as _amd_fgk
    if getattr(_amd_fgk, "triton", None) is None:
        # Sin Triton (Windows ROCm) solo funciona el algoritmo explicit_gemm de flex_gemm.
        FLEX_GEMM_ALGO = "explicit_gemm"
        print("[SPARSE] flex_gemm sin Triton: se usa explicit_gemm")
except Exception:
    pass
'''

TRITON_ORIGINAL = "import triton\nimport triton.compiler\n"
TRITON_NUEVO = (MARCA + "\ntry:\n    import triton\n    import triton.compiler\n"
                "except ImportError:\n    triton = None\n")


def leer(ruta):
    with open(ruta, "r", encoding="utf-8", newline="") as f:
        return f.read()


def escribir(ruta, texto):
    if not os.path.exists(ruta + ".orig"):
        shutil.copy2(ruta, ruta + ".orig")
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        f.write(texto)


def anadir_al_final(ruta, bloque):
    texto = leer(ruta)
    if MARCA in texto:
        print("  ya parcheado:", ruta)
        return
    salto = "\r\n" if "\r\n" in texto else "\n"
    escribir(ruta, texto.rstrip() + salto + salto + bloque.replace("\n", salto))
    print("  parcheado:", ruta)


def parchear_triton(ruta):
    texto = leer(ruta)
    if MARCA in texto:
        print("  ya parcheado:", ruta)
        return
    salto = "\r\n" if "\r\n" in texto else "\n"
    original = TRITON_ORIGINAL.replace("\n", salto)
    if original not in texto:
        sys.exit(f"ERROR: no encuentro 'import triton' en {ruta}; el nodo ha cambiado, avisa a Claude.")
    escribir(ruta, texto.replace(original, TRITON_NUEVO.replace("\n", salto), 1))
    print("  parcheado:", ruta)


def instalar_nvdiffrast():
    fuente = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvdiffrast_torch.py")
    if not os.path.exists(fuente):
        sys.exit(f"ERROR: falta {fuente}; descargalo junto a este script.")
    destino = os.path.join(sysconfig.get_paths()["purelib"], "nvdiffrast")
    if os.path.exists(os.path.join(destino, "__init__.py")):
        actual = leer(os.path.join(destino, "__init__.py"))
        if "3DIA" not in actual:
            sys.exit(f"ERROR: ya hay un nvdiffrast real en {destino}; no lo toco.")
    os.makedirs(destino, exist_ok=True)
    with open(os.path.join(destino, "__init__.py"), "w", encoding="utf-8") as f:
        f.write('"""Sustituto de nvdiffrast en PyTorch puro para AMD (3DIA)."""\n__version__ = "0.0.0+amd.torch"\n')
    shutil.copy2(fuente, os.path.join(destino, "torch.py"))
    print("  instalado:", destino)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    nodo = os.path.abspath(sys.argv[1])
    nodes_py = os.path.join(nodo, "nodes.py")
    if not os.path.exists(nodes_py):
        sys.exit(f"ERROR: no encuentro {nodes_py}")
    print("1) Sustituto de nvdiffrast")
    instalar_nvdiffrast()
    print("2) triton opcional")
    parchear_triton(nodes_py)
    print("3) Atencion SDPA")
    base = os.path.join(nodo, "trellis2", "modules")
    anadir_al_final(os.path.join(base, "attention", "config.py"), PARCHE_CONFIG_DENSA)
    anadir_al_final(os.path.join(base, "sparse", "config.py"), PARCHE_CONFIG_DISPERSA)
    anadir_al_final(os.path.join(base, "sparse", "attention", "windowed_attn.py"), PARCHE_VENTANAS)
    print("4) Convolucion dispersa sin Triton")
    anadir_al_final(os.path.join(base, "sparse", "conv", "config.py"), PARCHE_CONV)

    print("5) Comprobando modulos que importa el nodo")
    faltan = []
    for m in ["torch", "torchvision", "trimesh", "easydict", "utils3d", "transformers", "cv2",
              "pymeshlab", "meshlib.mrmeshpy", "cumesh", "o_voxel", "flex_gemm.ops.grid_sample",
              "nvdiffrast.torch", "huggingface_hub", "safetensors", "rembg", "open3d", "plotly", "scipy"]:
        try:
            __import__(m)
        except Exception as e:
            faltan.append(f"{m}: {type(e).__name__}: {e}")
    if faltan:
        print("  FALTAN o fallan:")
        for x in faltan:
            print("   -", x)
    else:
        print("  todos los modulos cargan")
    print("Listo.")


if __name__ == "__main__":
    main()
