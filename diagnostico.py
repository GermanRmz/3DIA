"""Diagnóstico de modelos planos.

Genera la forma de tu imagen en tu GPU y la decodifica de 4 maneras, midiendo el
grosor (lado menor / lado mayor) de cada resultado:

  A. FlashVDM en fp32 (lo nuevo)       B. FlashVDM en fp16 (lo de antes)
  C. Decodificador normal en fp32      D. Decodificador normal en fp16

Uso:
    python diagnostico.py figura.png
y pega la salida completa. Tarda unos minutos.
"""
import sys
import time

import modelos  # noqa: F401
import numpy as np
import torch
from PIL import Image

import generar3d as g
from preparar_imagen import preparar


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|",
          torch.cuda.get_device_name(0), "| capacidad", torch.cuda.get_device_capability(0))
    imagen, notas = preparar(Image.open(sys.argv[1]))
    print(*notas)
    p = g.cargar_pipeline("cuda")
    rapido, normal = p.vae, g._vae_normal

    for semilla in (0, 1):
        latentes = p(image=imagen, num_inference_steps=5, generator=torch.manual_seed(semilla),
                     output_type="latent", enable_pbar=False)
        print(f"\nSemilla {semilla}: latentes finitos={bool(torch.isfinite(latentes).all())}, "
              f"VRAM usada {torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB")
        for nombre, vae, tipo, trozos in (("A. FlashVDM fp32", rapido, torch.float32, 20000),
                                          ("B. FlashVDM fp16", rapido, torch.float16, 20000),
                                          ("C. normal fp32", normal, torch.float32, 20000),
                                          ("D. normal fp16", normal, torch.float16, 20000)):
            vae.to("cuda", tipo)
            p.vae = vae
            torch.cuda.empty_cache()
            t = time.time()
            try:
                malla = p._export(latentes.to(tipo), "trimesh", octree_resolution=256,
                                  num_chunks=trozos, enable_pbar=False)[0]
            except Exception as e:  # noqa: BLE001
                print(f"  {nombre}: ERROR {type(e).__name__}: {str(e)[:120]}")
                continue
            finally:
                vae.to(torch.float32)
                if vae is normal and g._gpu_pequena:
                    vae.to("cpu")
            if malla is None:
                print(f"  {nombre}: sin superficie")
                continue
            v = np.asarray(malla.vertices)
            lados = (np.percentile(v, 98, axis=0) - np.percentile(v, 2, axis=0)).round(2)
            ruta = f"diag_s{semilla}_{nombre[0]}.glb"
            malla.export(ruta)
            print(f"  {nombre}: grosor {g.grosor(malla):.0%}, lados xyz {lados}, "
                  f"{len(malla.faces)} caras, {time.time() - t:.0f} s -> {ruta}")
    p.vae = rapido


if __name__ == "__main__":
    main()
