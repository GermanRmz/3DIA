"""Comprueba el fallo de ROCm con capas Linear grandes y que el parche lo corrige.

Reproduce la capa que fallaba en el decodificador de textura (928624 x 128 -> 512)
en la GPU, sin parche y con parche, comparando con la CPU. Tarda segundos.
Uso (desde C:\\AMD-AI\\Pixal3D-ROCm):
    python_embeded\\python.exe prueba_linear_grande.py
"""
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

raiz = os.path.dirname(os.path.abspath(__file__))
ruta = os.path.join(raiz, "ComfyUI", "custom_nodes", "amd_parche_rocm.py")
if not os.path.exists(ruta):
    ruta = os.path.join(raiz, "amd_parche_rocm.py")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("dispositivo:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")

casos = [(928624, 128, 512), (600000, 512, 512), (3717012, 64, 64), (3717012, 64, 6), (300000, 1024, 1024),
         (524288, 32, 32), (524289, 32, 32), (2000000, 3, 3)]
lin_orig = F.linear
spec = importlib.util.spec_from_file_location("amd_parche_rocm", ruta)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.instalar()
lin_parche = F.linear

malos = {"SIN parche": 0, "CON parche": 0}
for i, (m, k, n) in enumerate(casos):
    g = torch.Generator().manual_seed(i)
    x = torch.randn(m, k, generator=g)
    w = torch.randn(n, k, generator=g) / k ** 0.5
    b = torch.randn(n, generator=g)
    ref = lin_orig(x, w, b)
    xd, wd, bd = x.to(dev), w.to(dev), b.to(dev)
    del x
    print(f"\n{m}x{k} -> {n} ({m / 2**19:.2f} x 2^19 filas)")
    for titulo, fn in (("SIN parche", lin_orig), ("CON parche", lin_parche)):
        y = fn(xd, wd, bd).cpu()
        err = (y - ref).abs().max(1).values > 1e-3 * ref.abs().max()
        del y
        frac = float(err.float().mean())
        primera = int(err.nonzero()[0]) if err.any() else -1
        malos[titulo] += frac > 0
        print(f"  [{'OK' if frac == 0 else 'FALLA'}] {titulo}: filas mal {frac:.2%}"
              + (f", desde la fila {primera}" if primera >= 0 else ""))
    del xd, ref
    if dev.type == "cuda":
        torch.cuda.empty_cache()

sin, con = malos["SIN parche"], malos["CON parche"]
print("\nRESUMEN:", "el parche corrige el fallo" if sin and not con else
      ("no se reprodujo el fallo" if not sin else "el parche NO basta"))
