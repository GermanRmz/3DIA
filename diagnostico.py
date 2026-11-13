"""Diagnóstico de modelos planos.

Genera la forma de tu imagen en tu GPU de varias maneras y mide el grosor de
cada resultado (lado menor / lado mayor), para encontrar qué la aplana:

  1. Como la app (todo en fp16).
  2. Igual, pero con el cálculo de atención básico en vez del optimizado.
  3. Codificador de imagen (DINOv2) en fp32, en la CPU.
  4. Codificador en fp32 (CPU) y generador (DiT) en fp32 (GPU).

Uso:
    python diagnostico.py figura.png
y pega la salida completa. Tarda unos minutos.
"""
import contextlib
import sys
import time

import modelos  # noqa: F401
import numpy as np
import torch
from PIL import Image

import generar3d as g
from preparar_imagen import preparar


def lados(malla):
    v = np.asarray(malla.vertices)
    return (np.percentile(v, 98, axis=0) - np.percentile(v, 2, axis=0)).round(2)


def atencion_basica():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False,
                                              enable_math=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|",
          torch.cuda.get_device_name(0), "| capacidad", torch.cuda.get_device_capability(0))
    imagen, notas = preparar(Image.open(sys.argv[1]))
    print(*notas)

    p = g.cargar_pipeline("cuda")
    vae_flash = p.vae
    if g._vae_normal is not None:
        p.vae = g._vae_normal  # decodificador normal: se compara la forma, no FlashVDM
    original = p.encode_cond
    embeddings = {}

    def capturar(nombre):
        def encode(*args, **kwargs):
            cond = original(*args, **kwargs)
            embeddings[nombre] = cond["main"].detach().float().cpu()
            return cond
        return encode

    def encode_cpu(image, additional_cond_inputs=None, **_):
        t = time.time()
        p.conditioner.to("cpu", torch.float32)
        extra = {k: v.to("cpu", torch.float32) if isinstance(v, torch.Tensor) else v
                 for k, v in (additional_cond_inputs or {}).items()}
        with torch.no_grad():
            cond = p.conditioner(image=image.to("cpu", torch.float32), **extra)
        embeddings["cpu"] = cond["main"].detach().float()
        print(f"   (codificador en CPU fp32: {time.time() - t:.0f} s)")
        return {k: v.to("cuda", p.dtype) if isinstance(v, torch.Tensor) else v
                for k, v in cond.items()}

    def generar(nombre, contexto=contextlib.nullcontext(), semilla=0):
        torch.cuda.empty_cache()
        t = time.time()
        try:
            with contexto:
                lat = p(image=imagen, num_inference_steps=5, generator=torch.manual_seed(semilla),
                        output_type="latent", enable_pbar=False)
            malla = p._export(lat.to(torch.float16), "trimesh", octree_resolution=128,
                              num_chunks=20000, enable_pbar=False)[0]
        except Exception as e:  # noqa: BLE001
            print(f"{nombre}: ERROR {type(e).__name__}: {str(e)[:150]}")
            return
        if malla is None:
            print(f"{nombre}: sin superficie")
            return
        ruta = "diag_" + nombre.split(".")[0] + ".glb"
        malla.export(ruta)
        print(f"{nombre}: grosor {g.grosor(malla):.0%}, lados xyz {lados(malla)}, "
              f"{time.time() - t:.0f} s -> {ruta}")

    p.encode_cond = capturar("gpu")
    generar("1. como la app")
    generar("1b. como la app, semilla 1", semilla=1)
    p.encode_cond = capturar("gpu_basica")
    generar("2. atención básica", atencion_basica())

    p.encode_cond = encode_cpu
    generar("3. codificador fp32")
    p.model.to(torch.float32)
    p.dtype = torch.float32
    generar("4. codificador y DiT fp32")

    print()
    ref = embeddings.get("cpu")
    for nombre in ("gpu", "gpu_basica"):
        a = embeddings.get(nombre)
        if a is None or ref is None:
            continue
        a, b = a.flatten(), ref.flatten()
        print(f"Codificador {nombre}: nan={int(torch.isnan(a).sum())} inf={int(torch.isinf(a).sum())} "
              f"max={float(a.nan_to_num().abs().max()):.1f} (CPU fp32 max={float(b.abs().max()):.1f}), "
              f"similitud con fp32 {float(torch.nn.functional.cosine_similarity(a.nan_to_num(), b, dim=0)):.4f}")
    p.vae = vae_flash


if __name__ == "__main__":
    main()
