"""Analiza las entradas reales del horneado guardadas por amd_volcado_textura.py.

Lee ComfyUI/output/amd_horneado.npz y:
  1. revisa los colores de los voxeles (rango, NaN, si son suaves o ruido)
  2. guarda vistas de los voxeles coloreados (amd_voxeles_*.png) para verlas
  3. mide la calidad del desenrollado UV real (solapes, pliegues, islas)
  4. hornea la textura en GPU y en CPU y compara (amd_base_gpu.png / amd_base_cpu.png)
Uso (desde C:\\AMD-AI\\Pixal3D-ROCm, con ComfyUI cerrado):
    python_embeded\\python.exe prueba_horneado_real.py
"""
import os
import sys
import time
import types

import numpy as np

raiz = os.path.dirname(os.path.abspath(__file__))
carpeta_comfy = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(raiz, "ComfyUI")
TEX = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
sys.path.insert(0, carpeta_comfy)
os.chdir(carpeta_comfy)
sys.argv = [sys.argv[0]]

import torch  # noqa: E402

if not torch.cuda.is_available():
    import comfy.options  # noqa: E402
    comfy.options.enable_args_parsing()
    sys.argv.append("--cpu")
if "server" not in sys.modules:
    _srv = types.ModuleType("server")
    _srv.PromptServer = type("PromptServer", (), {"instance": None})
    sys.modules["server"] = _srv

import comfy.model_management as mm  # noqa: E402
import comfy_extras.nodes_mesh_postprocess as mp  # noqa: E402
from PIL import Image  # noqa: E402

GPU = mm.get_torch_device()
CPU = torch.device("cpu")
_original = mm.get_torch_device
salida = os.path.join(carpeta_comfy, "output")
ruta = os.path.join(salida, "amd_horneado.npz")
if not os.path.exists(ruta):
    print("No existe", ruta, "- ejecuta antes el workflow con amd_volcado_textura.py instalado")
    sys.exit(1)


def usar(dev):
    mm.get_torch_device = (lambda: dev) if dev.type == "cpu" else _original


def guardar(img, nombre):
    img = np.nan_to_num(np.clip(img, 0, 1))
    Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(salida, nombre))
    print("  guardado output\\" + nombre)


z = np.load(ruta)
V, Fc, UV = z["vertices"].astype(np.float32), z["faces"].astype(np.int64), z["uvs"].astype(np.float32)
VC, COL = z["voxel_coords"], z["voxel_colors"].astype(np.float32)
R = int(z["resolution"])
print(f"malla: {len(V)} vertices, {len(Fc)} caras, UV {UV.shape}; voxeles {VC.shape}, colores {COL.shape}, resolucion {R}")
if VC.shape[1] == 4:
    VC = VC[VC[:, 0] == 0][:, 1:] if (VC[:, 0] == 0).any() else VC[:, 1:]
    COL = z["voxel_colors"].astype(np.float32)[: len(VC)]
VC = VC.astype(np.int64)

# ------------------------------------------------------------------ 1. colores de voxeles
print("\n1. colores de los voxeles")
print("  finitos:", bool(np.isfinite(COL).all()))
for c in range(COL.shape[1]):
    x = COL[:, c]
    print(f"  canal {c}: min {np.nanmin(x):.3f}  media {np.nanmean(x):.3f}  max {np.nanmax(x):.3f}")
clave = (VC[:, 0] * R + VC[:, 1]) * R + VC[:, 2]
orden = np.argsort(clave)
sk = clave[orden]
difs = []
for eje in range(3):
    paso = (R * R, R, 1)[eje]
    q = clave + paso
    pos = np.clip(np.searchsorted(sk, q), 0, len(sk) - 1)
    ok = sk[pos] == q
    j = orden[pos[ok]]
    difs.append(np.abs(COL[ok, :3] - COL[j, :3]).mean(1))
difs = np.concatenate(difs)
print(f"  diferencia con el vecino: mediana {np.median(difs):.3f}, p90 {np.percentile(difs, 90):.3f}"
      "  (textura real suele dar <0.05; ruido ~0.3)")

# ------------------------------------------------------------------ 2. vistas de voxeles
print("\n2. vistas de los voxeles coloreados")
S = min(R, 1024)
for eje, nombre in ((0, "x"), (1, "y"), (2, "z")):
    u, v_ = [a for a in range(3) if a != eje]
    pu = VC[:, u] * S // R
    pv = (R - 1 - VC[:, v_]) * S // R
    o = np.lexsort((VC[:, eje], pv, pu))
    pix = pu[o] * S + pv[o]
    ult = np.r_[pix[1:] != pix[:-1], True]
    img = np.zeros((S, S, 3), np.float32)
    sel = o[ult]
    img[pv[sel], pu[sel]] = COL[sel, :3]
    guardar(img, f"amd_voxeles_{nombre}.png")

# ------------------------------------------------------------------ 3. calidad UV
print("\n3. desenrollado UV real")
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
uvn = mp._normalize_uvs_to_unit(UV.copy(), True)
a, b, c = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
area3 = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
ua, ub, uc = uvn[Fc[:, 0]], uvn[Fc[:, 1]], uvn[Fc[:, 2]]
sarea = 0.5 * ((ub[:, 0] - ua[:, 0]) * (uc[:, 1] - ua[:, 1]) - (uc[:, 0] - ua[:, 0]) * (ub[:, 1] - ua[:, 1]))
nv, nf = len(uvn), len(Fc)
g = coo_matrix((np.ones(3 * nf), (np.repeat(np.arange(nf), 3), Fc.reshape(-1) + nf)), shape=(nf + nv, nf + nv))
_, lab = connected_components(g, directed=False)
isla = lab[:nf]
_, isla = np.unique(isla, return_inverse=True)
ni = isla.max() + 1
pos = np.bincount(isla, weights=(sarea > 0).astype(np.float64), minlength=ni)
tot = np.bincount(isla, minlength=ni)
inv = ((sarea > 0) != (pos[isla] * 2 >= tot[isla])) & (np.abs(sarea) > 0)
caras_isla = np.bincount(isla, minlength=ni)
print(f"  islas: {ni}  (con <=2 caras: {(caras_isla <= 2).sum()})")
print(f"  caras con area UV 0: {(np.abs(sarea) == 0).mean():.2%}")
print(f"  caras invertidas dentro de su isla: {inv.mean():.2%}")
ratio = np.abs(sarea) / np.maximum(area3, 1e-20)
med = np.median(ratio[ratio > 0])
print(f"  estiramiento extremo (x10): {((ratio < med / 10) | (ratio > med * 10)).mean():.2%}")
usar(CPU)
_, _, cov = mp._rasterize_uv_barycentric(Fc, uvn, 1024)
print(f"  solape (area UV / texels cubiertos, 1.0 = ninguno): {np.abs(sarea).sum() * 1024 * 1024 / max(int(cov.sum()), 1):.3f}")
print(f"  texels por isla (media) a {TEX}px: {np.abs(sarea).sum() * TEX * TEX / ni:.1f}")

# ------------------------------------------------------------------ 4. horneado GPU vs CPU
print(f"\n4. horneado a {TEX}px en GPU y en CPU")
vc_t, col_t = torch.from_numpy(VC), torch.from_numpy(COL)
res = {}
for nombre, dev in (("gpu", GPU), ("cpu", CPU)):
    if nombre == "gpu" and GPU.type != "cuda":
        continue
    usar(dev)
    t0 = time.time()
    pm, mk = mp._bake_position_map(V, Fc.astype(np.uint32), uvn, TEX)
    at = mp._sample_voxel_attrs_per_texel(pm, mk, vc_t, col_t, R)
    print(f"  {nombre}: {time.time() - t0:.1f}s, texels cubiertos {mk.mean():.1%}")
    res[nombre] = (pm, mk, at)
    guardar(np.where(mk[..., None], at[..., :3], 0), f"amd_base_{nombre}.png")
if len(res) == 2:
    (pg, mg, ag), (pc, mc, ac) = res["gpu"], res["cpu"]
    print(f"  mascara igual: {(mg == mc).mean():.4%}")
    m = mg & mc
    print(f"  posiciones: diferencia maxima {np.abs(pg[m] - pc[m]).max():.2e}")
    d = np.abs(ag[m] - ac[m])
    print(f"  colores: diferencia media {d.mean():.4f}, maxima {d.max():.4f}")
usar(GPU)
print("\nListo. Mandame esta salida y las imagenes output\\amd_voxeles_*.png y output\\amd_base_*.png")
