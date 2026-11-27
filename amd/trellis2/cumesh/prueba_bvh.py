"""Prueba del BVH (trazado de rayos cuBQL) y de xatlas de cumesh en la GPU AMD.
Uso: C:\\AMD-AI\\Pixal3D-ROCm\\python_embeded\\python.exe prueba_bvh.py
Resultado esperado: el rayo choca en (0.3, 0.3, 0), cara 11, profundidad 1, y "TODO OK"."""
import torch
from cumesh import cuBVH
import cumesh._xatlas  # noqa: F401
print(">> xatlas carga OK", flush=True)

n = 4
xs = torch.linspace(0, 1, n + 1)
gy, gx = torch.meshgrid(xs, xs, indexing="ij")
v = torch.stack([gx.flatten(), gy.flatten(), torch.zeros((n + 1) ** 2)], 1).cuda()
caras = []
for i in range(n):
    for j in range(n):
        a = i * (n + 1) + j
        b, c, d = a + 1, a + n + 1, a + n + 2
        caras += [[a, b, d], [a, d, c]]
f = torch.tensor(caras, dtype=torch.int32, device="cuda")
print(">> plano:", v.shape[0], "vertices,", f.shape[0], "triangulos", flush=True)

bvh = cuBVH(v, f)
print(">> BVH construido", flush=True)
origen = torch.tensor([[0.3, 0.3, -1.0]], device="cuda")
direccion = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
r = bvh.ray_trace(origen, direccion)
print(">> ray_trace:", r, flush=True)
print("TODO OK", flush=True)
