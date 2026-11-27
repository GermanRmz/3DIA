"""Prueba paso a paso de cumesh en la GPU AMD (cada paso imprime antes de ejecutarse).
Uso: C:\\AMD-AI\\Pixal3D-ROCm\\python_embeded\\python.exe prueba_cumesh.py
El BVH se prueba aparte con prueba_bvh.py (necesita mas de 8 triangulos)."""
import os
os.environ["HIP_LAUNCH_BLOCKING"] = "1"   # que el error salte en el paso exacto
import torch  # noqa: E402
from cumesh import CuMesh  # noqa: E402


def paso(t):
    print(">>", t, flush=True)


print(torch.cuda.get_device_name(0), torch.version.hip, flush=True)

paso("1 tetraedro: init")
v = torch.tensor([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device="cuda")
f = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.int32, device="cuda")
m = CuMesh()
m.init(v, f)
paso("2 normales de caras")
m.compute_face_normals()
paso("3 normales de vertices")
m.compute_vertex_normals()
paso("4 leer")
m.read()
m.read_face_normals()
m.read_vertex_normals()

paso("5 malla grande (4 M vertices)")
nv, nf = 4_091_147, 7_921_414
gv = torch.arange(nv * 3, dtype=torch.float32, device="cuda").reshape(-1, 3) / nv
gf = (torch.arange(nf * 3, dtype=torch.int32, device="cuda").reshape(-1, 3) % nv)
g = CuMesh()
g.init(gv, gf)
g.read()
del g, gv, gf

paso("6 cuadrado abierto: init")
ov = torch.tensor([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], device="cuda")
of = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device="cuda")
o = CuMesh()
o.init(ov, of)
for nombre in ["get_edges", "get_boundary_info", "get_vertex_edge_adjacency",
               "get_vertex_boundary_adjacency", "get_manifold_boundary_adjacency",
               "read_manifold_boundary_adjacency", "get_boundary_connected_components",
               "get_boundary_loops"]:
    paso("7 " + nombre)
    getattr(o, nombre)()
paso("8 fill_holes")
o.fill_holes(max_hole_perimeter=10.0)
print("TODO OK (pasos 1-8)", flush=True)
