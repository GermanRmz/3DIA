"""Parche para ComfyUI en GPUs AMD (ROCm en Windows): multiplicaciones grandes por trozos.

Problema encontrado: en ROCm/Windows (RX 9060 XT, gfx1200) una multiplicacion de matrices
(Linear/matmul) con mas de 524288 filas (2^19) devuelve basura a partir de esa fila, sin
dar error, sea cual sea el numero de columnas. En TRELLIS.2/Pixal3D eso rompe la textura (y parte de la forma)
en los niveles finos del decodificador (~1 millon de voxeles x 512 canales).

Este parche envuelve torch.nn.functional.linear y torch.matmul para que, en la GPU,
las operaciones grandes se hagan por bloques de filas que quedan muy por debajo del
limite. El resultado es el mismo; solo cambia como se reparte el trabajo.

Instalacion: copiar a ComfyUI/custom_nodes/. Se activa solo si torch usa ROCm (HIP).
Para desactivarlo: variable de entorno AMD_SIN_TROZOS=1 o borrar el archivo.
"""
import logging
import os

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Filas maximas por bloque: el fallo aparece a partir de la fila 2^19; 2^18 deja margen.
FILAS_MAX = int(os.environ.get("AMD_FILAS_TROZO", 1 << 18))
# Ademas, elementos maximos por bloque (para no crear temporales enormes).
LIMITE = int(os.environ.get("AMD_LIMITE_TROZO", 1 << 26))


def _filas_por_trozo(n, k):
    return max(1, min(FILAS_MAX, LIMITE // max(n, k, 1)))


def _tensor_normal(t):
    return type(t) in (torch.Tensor, torch.nn.Parameter)


def instalar():
    import torch.nn.functional as F
    if getattr(F.linear, "_amd_trozos", False):
        return

    linear_orig = F.linear
    matmul_orig = torch.matmul

    def linear(input, weight, bias=None):
        if (input.is_cuda and input.dim() >= 2 and _tensor_normal(input) and _tensor_normal(weight)
                and weight.dim() == 2):
            n, k = weight.shape
            m = input.numel() // max(input.shape[-1], 1)
            filas = _filas_por_trozo(n, k)
            if m > filas:
                x = input.reshape(m, input.shape[-1])
                primero = linear_orig(x[:filas], weight, bias)
                out = torch.empty((m, n), dtype=primero.dtype, device=primero.device)
                out[:filas] = primero
                for s in range(filas, m, filas):
                    out[s:s + filas] = linear_orig(x[s:s + filas], weight, bias)
                return out.reshape(*input.shape[:-1], n)
        return linear_orig(input, weight, bias)

    def matmul(a, b, *args, **kwargs):
        if (not args and not kwargs and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.is_cuda and a.dim() == 2 and b.dim() == 2 and _tensor_normal(a) and _tensor_normal(b)):
            m, k = a.shape
            n = b.shape[1]
            filas = _filas_por_trozo(n, k)
            if m > filas:
                primero = matmul_orig(a[:filas], b)
                out = torch.empty((m, n), dtype=primero.dtype, device=primero.device)
                out[:filas] = primero
                for s in range(filas, m, filas):
                    out[s:s + filas] = matmul_orig(a[s:s + filas], b)
                return out
        return matmul_orig(a, b, *args, **kwargs)

    linear._amd_trozos = True
    matmul._amd_trozos = True
    F.linear = linear
    torch.nn.functional.linear = linear
    torch.matmul = matmul
    logging.info(f"[AMD] parche ROCm activo: Linear/matmul por trozos de hasta {FILAS_MAX} filas")


if getattr(torch.version, "hip", None) and not os.environ.get("AMD_SIN_TROZOS"):
    try:
        instalar()
    except Exception as e:
        logging.warning(f"[AMD] no se pudo instalar el parche ROCm: {e}")
