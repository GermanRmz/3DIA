"""Compara la primera etapa de TRELLIS.2 (estructura dispersa) en la GPU AMD contra la CPU.

Carga el modelo de flujo de la estructura dispersa y su decodificador, ejecuta una
pasada con la misma entrada en la CPU (float32, referencia) y en la GPU, y compara
la salida de cada bloque para encontrar el primero que se desvia.
Uso (desde C:\\AMD-AI\\Pixal3D-ROCm, con ComfyUI cerrado; tarda unos minutos):
    python_embeded\\python.exe prueba_ss_flow.py
"""
import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

raiz = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(raiz, "ComfyUI", "custom_nodes", "ComfyUI-Trellis2"))
from trellis2 import models  # noqa: E402

carpeta = os.path.join(raiz, "ComfyUI", "models", "microsoft", "TRELLIS.2-4B")
with open(os.path.join(carpeta, "pipeline.json")) as f:
    cfg = json.load(f)
cfg = cfg.get("args", cfg)
nombre_flujo = cfg["models"]["sparse_structure_flow_model"]
print("modelo:", nombre_flujo)


def cos(a, b):
    a, b = a.float().cpu().flatten(), b.float().cpu().flatten()
    if not torch.isfinite(a).all():
        return float("nan")
    return F.cosine_similarity(a, b, dim=0).item()


def capturar(modelo, almacen):
    hooks = []
    for i, bloque in enumerate(modelo.blocks):
        hooks.append(bloque.register_forward_hook(
            lambda m, entrada, salida, i=i: almacen.__setitem__(i, salida.detach().float().cpu())))
    return hooks


flujo = models.from_pretrained(os.path.join(carpeta, nombre_flujo)).eval()
print("dtype del modelo:", getattr(flujo, "dtype", None), "| resolucion", flujo.resolution,
      "| canales", flujo.in_channels, "| bloques", len(flujo.blocks), "| pe_mode", flujo.pe_mode)
if getattr(flujo, "rope_phases", None) is not None:
    print("rope_phases:", flujo.rope_phases.dtype, tuple(flujo.rope_phases.shape))

torch.manual_seed(0)
R, C = flujo.resolution, flujo.in_channels
x = torch.randn(1, C, R, R, R)
t = torch.tensor([500.0])
cond = F.layer_norm(torch.randn(1, 1029, 1024), (1024,))

# Referencia: CPU en float32
ref_cpu = copy.deepcopy(flujo)
ref_cpu.convert_to(torch.float32)
bloques_cpu = {}
capturar(ref_cpu, bloques_cpu)
inicio = time.time()
with torch.no_grad():
    out_cpu = ref_cpu(x, t, cond)
print(f"CPU float32 terminada en {time.time() - inicio:.0f} s")
del ref_cpu

for etiqueta, convertir in (("GPU dtype original", False), ("GPU float32", True)):
    g = copy.deepcopy(flujo)
    if convertir:
        g.convert_to(torch.float32)
    g = g.cuda()
    bloques_gpu = {}
    capturar(g, bloques_gpu)
    with torch.no_grad():
        out_gpu = g(x.cuda(), t.cuda(), cond.cuda())
    print(f"\n== {etiqueta} ==")
    print(f"salida final: similitud {cos(out_gpu, out_cpu):.5f}")
    primero = None
    for i in sorted(bloques_cpu):
        c = cos(bloques_gpu[i], bloques_cpu[i])
        if primero is None and not (c > 0.99):
            primero = i
        if i < 3 or i == len(bloques_cpu) - 1 or (primero == i):
            print(f"  bloque {i:2d}: similitud {c:.5f}")
    print("  primer bloque desviado:", primero if primero is not None else "ninguno")
    del g
    torch.cuda.empty_cache()

# Decodificador de la estructura dispersa
ruta_dec = os.path.join(raiz, "ComfyUI", "models", "microsoft", "TRELLIS-image-large", "ckpts",
                        "ss_dec_conv3d_16l8_fp16")
if os.path.exists(ruta_dec + ".json"):
    dec = models.from_pretrained(ruta_dec).eval()
    z = torch.randn(1, C, R, R, R)
    dec_cpu = copy.deepcopy(dec)
    if hasattr(dec_cpu, "convert_to_fp32"):
        dec_cpu.convert_to_fp32()
    dec_cpu = dec_cpu.float()
    with torch.no_grad():
        d_cpu = dec_cpu(z)
        d_gpu = dec.cuda()(z.cuda())
    print(f"\n== decodificador ==\nsimilitud {cos(d_gpu, d_cpu):.5f}, "
          f"ocupacion CPU {(d_cpu > 0).float().mean().item():.4f}, GPU {(d_gpu > 0).float().mean().item():.4f}")
print("Fin.")
