"""Comprueba en la GPU AMD que las operaciones de TRELLIS.2 dan resultados correctos.

Compara cada operacion (en float16/bfloat16, como la usa el modelo) contra una
referencia en float32. Uso:
    python_embeded\\python.exe prueba_numerica_amd.py
"""
import math

import torch
import torch.nn.functional as F

dev = "cuda"
torch.manual_seed(0)
print(torch.cuda.get_device_name(0), torch.__version__)


def informe(nombre, salida, ref):
    salida = salida.float()
    malos = (~torch.isfinite(salida)).sum().item()
    err = (salida - ref).abs().max().item() if malos == 0 else float("nan")
    escala = ref.abs().max().item()
    estado = "OK" if malos == 0 and err <= 0.02 * max(escala, 1e-6) + 1e-3 else "MAL"
    print(f"[{estado}] {nombre}: error max {err:.4g} (escala {escala:.3g}), no finitos {malos}")


def ref_atencion(q, k, v, mascara=None):
    s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mascara is not None:
        s = s.masked_fill(~mascara, float("-inf"))
    return torch.softmax(s, -1) @ v.float()


# 1) Atencion densa (transformers de flujo y DINOv3)
for dt in (torch.float16, torch.bfloat16):
    for L in (1024, 4096):
        q, k, v = (torch.randn(1, 16, L, 64, device=dev, dtype=dt) for _ in range(3))
        informe(f"SDPA {str(dt)[6:]} L={L}", F.scaled_dot_product_attention(q, k, v), ref_atencion(q, k, v))

# 2) Atencion con mascara (ruta SDPA por ventanas del parche)
q, k, v = (torch.randn(64, 16, 512, 64, device=dev, dtype=torch.float16) for _ in range(3))
lens = torch.randint(1, 513, (64,), device=dev)
m = (torch.arange(512, device=dev)[None, :] < lens[:, None])[:, None, None, :]
informe("SDPA float16 con mascara", F.scaled_dot_product_attention(q, k, v, attn_mask=m), ref_atencion(q, k, v, m))

# 3) Convolucion dispersa de flex_gemm (explicit_gemm), como en los VAE
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d

flex_gemm.ops.spconv.set_algorithm("explicit_gemm")
R = 64
ocupado = torch.rand(R, R, R, device=dev) < 0.15
xyz = torch.nonzero(ocupado).int()
coords = torch.cat([torch.zeros(xyz.shape[0], 1, dtype=torch.int32, device=dev), xyz], 1).contiguous()
N = coords.shape[0]
for Ci, Co in ((64, 64), (512, 512)):
    for dt in (torch.float16, torch.float32):
        feats = (torch.randn(N, Ci, device=dev) * 2).to(dt).contiguous()
        w = (torch.randn(Co, 3, 3, 3, Ci, device=dev) / math.sqrt(27 * Ci)).to(dt)
        b = torch.randn(Co, device=dev).to(dt)
        out, _ = sparse_submanifold_conv3d(feats, coords, torch.Size([1, Ci, R, R, R]), w, b)
        # referencia: rejilla densa de indices (se prueban las dos orientaciones del nucleo)
        idx = torch.full((R + 2, R + 2, R + 2), -1, dtype=torch.long, device=dev)
        idx[xyz[:, 0] + 1, xyz[:, 1] + 1, xyz[:, 2] + 1] = torch.arange(N, device=dev)
        f32, w32 = feats.float(), w.float()
        refs = []
        for invertir in (False, True):
            ref = b.float().expand(N, Co).clone()
            for a in range(3):
                for c in range(3):
                    for e in range(3):
                        vec = idx[xyz[:, 0] + a, xyz[:, 1] + c, xyz[:, 2] + e]
                        ok = vec >= 0
                        wa = w32[:, 2 - a, 2 - c, 2 - e, :] if invertir else w32[:, a, c, e, :]
                        ref[ok] += f32[vec[ok]] @ wa.t()
            refs.append(ref)
        ref = min(refs, key=lambda r: (out.float() - r).abs().nan_to_num(1e9).max().item())
        informe(f"flex_gemm conv {Ci}->{Co} {str(dt)[6:]} ({N} voxeles)", out, ref)

# 4) Normalizaciones y GEMM en float16
x = torch.randn(200000, 512, device=dev, dtype=torch.float16) * 4
informe("LayerNorm float16", F.layer_norm(x, (512,)), F.layer_norm(x.float(), (512,)))
xg = torch.randn(4, 512, 32, 32, 32, device=dev, dtype=torch.float16)
informe("GroupNorm float16", F.group_norm(xg, 32), F.group_norm(xg.float(), 32))
a = torch.randn(8192, 4096, device=dev, dtype=torch.float16)
bm = torch.randn(4096, 1024, device=dev, dtype=torch.float16) / 64
informe("matmul float16", a @ bm, a.float() @ bm.float())

# 5) Convolucion 3D densa (decodificador de estructura dispersa)
for cudnn in (True, False):
    torch.backends.cudnn.enabled = cudnn
    xc = torch.randn(1, 128, 32, 32, 32, device=dev, dtype=torch.float16)
    wc = torch.randn(128, 128, 3, 3, 3, device=dev, dtype=torch.float16) / 60
    informe(f"conv3d float16 (cudnn/MIOpen {'si' if cudnn else 'no'})",
            F.conv3d(xc, wc, padding=1), F.conv3d(xc.float(), wc.float(), padding=1))
print("Fin.")
