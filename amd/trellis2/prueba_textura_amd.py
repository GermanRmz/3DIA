"""Compara en la GPU AMD contra la CPU las etapas de textura de ComfyUI (mesh3d).

Usa una malla sintetica (esfera deformada) y prueba, por separado:
  1. matmul por lotes en fp64/fp32 (lo que usa el LSCM del desenrollado UV)
  2. desenrollado UV completo (_uv_unwrap) en GPU y en CPU, con medidas de calidad
     (solapes, triangulos invertidos, estiramiento)
  3. rasterizado UV / mapa de posiciones
  4. muestreo de colores de voxeles por texel
  5. relleno de costuras (JFA)
  6. oclusion ambiental (BVH + rayos)
Uso (desde C:\\AMD-AI\\Pixal3D-ROCm, con ComfyUI cerrado; tarda 1-3 minutos):
    python_embeded\\python.exe prueba_textura_amd.py
"""
import os
import sys
import time
import types

import numpy as np

raiz = os.path.dirname(os.path.abspath(__file__))
carpeta_comfy = os.path.join(raiz, "ComfyUI")
if len(sys.argv) > 1:
    carpeta_comfy = os.path.abspath(sys.argv[1])
sys.path.insert(0, carpeta_comfy)
os.chdir(carpeta_comfy)
sys.argv = [sys.argv[0]]

import torch  # noqa: E402

if not torch.cuda.is_available():
    import comfy.options  # noqa: E402
    comfy.options.enable_args_parsing()
    sys.argv.append("--cpu")

# server.PromptServer solo se usa para mensajes de progreso en la interfaz.
if "server" not in sys.modules:
    _srv = types.ModuleType("server")

    class _PromptServer:
        instance = None

    _srv.PromptServer = _PromptServer
    sys.modules["server"] = _srv

import comfy.model_management as mm  # noqa: E402
import comfy_extras.nodes_mesh_postprocess as mp  # noqa: E402

GPU = mm.get_torch_device()
CPU = torch.device("cpu")
_original = mm.get_torch_device
print("dispositivo:", GPU, torch.cuda.get_device_name(GPU) if GPU.type == "cuda" else "")
if GPU.type != "cuda":
    print("AVISO: no hay GPU; las dos pasadas seran en CPU")

problemas = []


def usar(dev):
    mm.get_torch_device = (lambda: dev) if dev.type == "cpu" else _original


def informe(nombre, ok, detalle):
    print(f"  [{'OK' if ok else 'FALLA'}] {nombre}: {detalle}")
    if not ok:
        problemas.append(nombre)


# ---------------------------------------------------------------- malla de prueba
def icosfera(sub):
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
         [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
         [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    v = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    for _ in range(sub):
        cache, nf = {}, []

        def medio(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                p = v[a] + v[b]
                v.append(p / np.linalg.norm(p))
                cache[k] = len(v) - 1
            return cache[k]

        for a, b, c in f:
            ab, bc, ca = medio(a, b), medio(b, c), medio(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return np.array(v), np.array(f, dtype=np.int64)


v, f = icosfera(6)                                   # ~82k caras
d = v.copy()
r = 1 + 0.15 * np.sin(5 * d[:, 0]) * np.cos(4 * d[:, 1]) + 0.1 * np.sin(7 * d[:, 2])
v = (d * r[:, None] * np.array([0.42, 0.30, 0.22])).astype(np.float32)   # dentro de [-0.5,0.5]
print(f"malla: {len(v)} vertices, {len(f)} caras")


# ---------------------------------------------------------------- 1. matmul por lotes
print("\n1. matmul por lotes (LSCM)")
if GPU.type == "cuda":
    g = torch.Generator().manual_seed(0)
    for dt in (torch.float64, torch.float32):
        A = torch.randn(48, 300, 160, generator=g, dtype=dt)
        ref = A.transpose(1, 2) @ A
        res = (A.to(GPU).transpose(1, 2) @ A.to(GPU)).cpu()
        err = ((res - ref).abs().max() / ref.abs().max()).item()
        informe(f"bmm {dt}", err < (1e-10 if dt == torch.float64 else 1e-4), f"error relativo {err:.2e}")
    A = torch.randn(48, 160, 160, generator=g, dtype=torch.float64)
    A = A @ A.transpose(1, 2) + 160 * torch.eye(160, dtype=torch.float64)
    b = torch.randn(48, 160, generator=g, dtype=torch.float64)
    ref = torch.linalg.solve(A, b)
    try:
        res = torch.linalg.solve(A.to(GPU), b.to(GPU)).cpu()
        err = ((res - ref).abs().max() / ref.abs().max()).item()
        informe("solve fp64 GPU", err < 1e-8, f"error relativo {err:.2e}")
    except Exception as e:  # ya sabemos que hipBLAS puede fallar aqui
        print(f"  [info] solve fp64 en GPU no disponible ({type(e).__name__}); el parche lo hace en CPU")


# ---------------------------------------------------------------- 2. desenrollado UV
def calidad_uv(nombre, vm, idx, uvs):
    p3 = v[vm]
    a, b, c = p3[idx[:, 0]], p3[idx[:, 1]], p3[idx[:, 2]]
    area3 = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    ua, ub, uc = uvs[idx[:, 0]], uvs[idx[:, 1]], uvs[idx[:, 2]]
    sarea = 0.5 * ((ub[:, 0] - ua[:, 0]) * (uc[:, 1] - ua[:, 1]) - (uc[:, 0] - ua[:, 0]) * (ub[:, 1] - ua[:, 1]))
    fuera = float(((uvs < -1e-4) | (uvs > 1 + 1e-4)).any(1).mean())
    # islas = componentes conexas por vertices UV compartidos; una isla entera espejada
    # es valida, lo que importa son caras invertidas dentro de su isla (pliegues)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nv, nfc = len(uvs), len(idx)
    filas = np.repeat(np.arange(nfc), 3)
    g = coo_matrix((np.ones(3 * nfc), (filas, idx.reshape(-1) + nfc)), shape=(nfc + nv, nfc + nv))
    _, lab = connected_components(g, directed=False)
    isla = lab[:nfc]
    pos = np.bincount(isla, weights=(sarea > 0).astype(np.float64), minlength=lab.max() + 1)
    tot = np.bincount(isla, minlength=lab.max() + 1)
    mayor_pos = pos[isla] * 2 >= tot[isla]
    neg = float(((sarea > 0) != mayor_pos)[np.abs(sarea) > 0].mean())
    print(f"  islas UV: {len(np.unique(isla))}")
    ratio = np.abs(sarea) / np.maximum(area3, 1e-20)
    med = np.median(ratio[ratio > 0])
    estir = float(((ratio < med / 10) | (ratio > med * 10)).mean())
    usar(CPU)
    res = 1024
    _, _, cov = mp._rasterize_uv_barycentric(idx, uvs, res)
    cubiertos = int(cov.sum())
    solape = float(np.abs(sarea).sum() * res * res / max(cubiertos, 1))
    print(f"  {nombre}: {len(idx)} caras, {len(uvs)} vertices UV")
    informe(f"{nombre} UV dentro de [0,1]", fuera < 1e-3, f"{fuera:.2%} de vertices fuera")
    informe(f"{nombre} caras invertidas dentro de su isla", neg < 0.02, f"{neg:.2%}")
    informe(f"{nombre} estiramiento extremo", estir < 0.02, f"{estir:.2%} de caras")
    informe(f"{nombre} solape (area UV / texels cubiertos)", solape < 1.25, f"{solape:.3f} (1.0 = sin solapes)")
    return cov


resultados = {}
for nombre, dev in (("CPU", CPU), ("GPU", GPU)):
    print(f"\n2. desenrollado UV en {nombre}")
    usar(dev)
    t0 = time.time()
    try:
        vm, idx, uvs = mp._uv_unwrap(torch.from_numpy(v).to(dev), torch.from_numpy(f).to(dev),
                                     "pec", 1024, 1, 0.0)
        uvs = uvs.copy()
        uvs[:, 1] = 1.0 - uvs[:, 1]
        print(f"  tiempo {time.time() - t0:.1f}s")
        resultados[nombre] = (vm, idx, uvs)
        calidad_uv(nombre, vm, idx, uvs)
    except Exception as e:
        informe(f"desenrollado {nombre}", False, f"{type(e).__name__}: {e}")
    if GPU.type != "cuda":
        break

if "CPU" not in resultados:
    print("\nNo se pudo desenrollar en CPU; no puedo seguir.")
    sys.exit(1)
vm, idx, uvs = resultados["CPU"]
vv = v[vm]


# ---------------------------------------------------------------- 3. rasterizado UV
print("\n3. rasterizado UV / mapa de posiciones (mismos UV en GPU y CPU)")
res = 1024
mapas = {}
for nombre, dev in (("CPU", CPU), ("GPU", GPU)):
    usar(dev)
    mapas[nombre] = mp._bake_position_map(vv, idx.astype(np.uint32), uvs, res)
pc, mc = mapas["CPU"]
pg, mg = mapas["GPU"]
igual = float((mc == mg).mean())
informe("mascara de cobertura", igual > 0.999, f"{igual:.4%} de texels iguales, cubiertos {mc.mean():.1%}")
ambos = mc & mg
err = float(np.abs(pc[ambos] - pg[ambos]).max()) if ambos.any() else 0.0
informe("posiciones", err < 1e-3, f"diferencia maxima {err:.2e}")
fin = float(np.isfinite(pg).all())
informe("posiciones finitas (GPU)", fin == 1.0, "sin NaN/inf" if fin == 1.0 else "hay NaN/inf")


# ---------------------------------------------------------------- 4. muestreo de voxeles
print("\n4. muestreo de colores de voxeles")
R = 256
rng = np.random.default_rng(0)
w = rng.dirichlet([1, 1, 1], size=600000).astype(np.float32)
pts = (w[:, :, None] * vv[idx[rng.integers(0, len(idx), 600000)]]).sum(1)
coords = np.unique(np.clip(np.floor((pts + 0.5) * R), 0, R - 1).astype(np.int64), axis=0)


def color_de(p):                                       # funcion suave conocida
    return np.stack([0.5 + 0.5 * np.sin(6 * p[..., 0]), 0.5 + 0.5 * np.cos(5 * p[..., 1]),
                     0.5 + 0.5 * np.sin(4 * p[..., 2] + 1), np.full(p.shape[:-1], 0.3),
                     np.full(p.shape[:-1], 0.7), np.ones(p.shape[:-1])], -1).astype(np.float32)


centros = (coords.astype(np.float32) + 0.5) / R - 0.5
colores = torch.from_numpy(color_de(centros))
coords_t = torch.from_numpy(coords)
mues = {}
for nombre, dev in (("CPU", CPU), ("GPU", GPU)):
    usar(dev)
    mues[nombre] = mp._sample_voxel_attrs_per_texel(pc, mc, coords_t, colores, R)
err = float(np.abs(mues["CPU"][mc] - mues["GPU"][mc]).max())
informe("muestreo GPU vs CPU", err < 1e-3, f"diferencia maxima {err:.2e}")
esperado = color_de(pc[mc])
for nombre in ("CPU", "GPU"):
    e = np.abs(mues[nombre][mc][:, :3] - esperado[:, :3])
    informe(f"muestreo {nombre} vs color esperado", float(np.percentile(e, 99)) < 0.1,
            f"error medio {e.mean():.3f}, p99 {np.percentile(e, 99):.3f}")


# ---------------------------------------------------------------- 5. relleno de costuras
print("\n5. relleno de costuras (JFA)")
img = mues["CPU"][..., :3]
rell = {}
for nombre, dev in (("CPU", CPU), ("GPU", GPU)):
    usar(dev)
    rell[nombre] = mp._seam_fill(img, mc)
err = float(np.abs(rell["CPU"] - rell["GPU"]).max())
informe("JFA GPU vs CPU", err < 1e-4, f"diferencia maxima {err:.2e}")
err = float(np.abs(rell["GPU"][mc] - img[mc]).max())
informe("JFA conserva texels cubiertos", err < 1e-6, f"diferencia maxima {err:.2e}")


# ---------------------------------------------------------------- 6. oclusion ambiental
print("\n6. oclusion ambiental (BVH)")
torch.manual_seed(0)
normales = torch.from_numpy(vv / np.linalg.norm(vv, axis=1, keepdims=True))   # aprox. para la esfera
ao = {}
for nombre, dev in (("CPU", CPU), ("GPU", GPU)):
    usar(dev)
    t0 = time.time()
    ao[nombre] = mp._bake_ambient_occlusion(torch.from_numpy(vv), torch.from_numpy(idx), vv, idx, uvs,
                                            normales, 256, num_samples=32, max_distance=0.5)
    print(f"  {nombre}: {time.time() - t0:.1f}s, AO media {ao[nombre][..., 0].mean():.3f}")
mk = mc[::4, ::4]
a_c, a_g = ao["CPU"][..., 0][mk], ao["GPU"][..., 0][mk]
informe("AO GPU vs CPU (media)", abs(a_c.mean() - a_g.mean()) < 0.03, f"CPU {a_c.mean():.3f} / GPU {a_g.mean():.3f}")
informe("AO GPU sin ruido", float((a_g < 0.5).mean()) < 0.05, f"{(a_g < 0.5).mean():.2%} de texels oscuros (forma casi convexa)")

usar(GPU)
print("\nRESUMEN:", "todo OK" if not problemas else "FALLAN: " + ", ".join(problemas))
