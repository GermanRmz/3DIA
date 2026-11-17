"""Sustituto de ``nvdiffrast.torch`` en PyTorch puro, para GPUs AMD (ROCm).

nvdiffrast solo funciona en NVIDIA. ComfyUI-Trellis2 lo usa para rasterizar
mallas: sobre todo para "hornear" la textura en el mapa UV y para vistas
previas. Este modulo implementa la parte de su API que usa el nodo:

- RasterizeCudaContext / RasterizeGLContext
- rasterize, interpolate, texture, antialias

Convenciones iguales a nvdiffrast: la fila 0 de la imagen es y = -1 (abajo),
los centros de pixel estan en (i + 0.5) / W, ``rast[..., 3]`` es id de
triangulo + 1 (0 = vacio) y ``attr = u*a0 + v*a1 + (1-u-v)*a2``.

Limitaciones: sin recorte contra el plano cercano (los triangulos con algun
w <= 0 se descartan), sin derivadas (``rast_db``, ``dattr`` van a cero),
``antialias`` no hace nada, ``texture`` es bilineal sin mipmaps y no hay
``DepthPeeler``.
"""
import os

import torch
import torch.nn.functional as F

__amd_fallback__ = True

# Pares (triangulo, pixel candidato) por bloque. Mas alto = mas rapido y mas memoria.
_PARES_POR_BLOQUE = int(os.environ.get("AMD_RAST_PARES", "8000000"))
_INT64_MAX = torch.iinfo(torch.int64).max


class RasterizeCudaContext:
    def __init__(self, device=None, **kwargs):
        self.device = device


RasterizeGLContext = RasterizeCudaContext


class DepthPeeler:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DepthPeeler no esta disponible en el sustituto AMD de nvdiffrast")


def _clave_profundidad(z):
    """float32 -> int64 que conserva el orden (para quedarse con el mas cercano)."""
    i = z.contiguous().view(torch.int32)
    i = torch.where(i < 0, i ^ 0x7FFFFFFF, i)
    return i.to(torch.int64) << 32


def _rasterizar_una(pos, tri, alto, ancho):
    dispositivo = pos.device
    pos = pos.float()
    tri = tri.long()
    salida = torch.zeros(alto, ancho, 4, dtype=torch.float32, device=dispositivo)
    if tri.shape[0] == 0:
        return salida

    v = pos[tri]                                   # [F, 3, 4]
    w = v[..., 3]
    valido = (w > 1e-8).all(dim=1)
    w_seguro = torch.where(w.abs() < 1e-8, torch.full_like(w, 1e-8), w)
    sx = (v[..., 0] / w_seguro + 1) * 0.5 * ancho  # [F, 3] en pixeles
    sy = (v[..., 1] / w_seguro + 1) * 0.5 * alto
    sz = v[..., 2] / w_seguro

    area = (sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0])
    valido &= area.abs() > 1e-12
    valido &= ~((sz < -1).all(dim=1) | (sz > 1).all(dim=1))

    x0 = torch.ceil(sx.min(dim=1).values - 0.5).clamp(0, ancho - 1).long()
    x1 = torch.floor(sx.max(dim=1).values - 0.5).clamp(-1, ancho - 1).long()
    y0 = torch.ceil(sy.min(dim=1).values - 0.5).clamp(0, alto - 1).long()
    y1 = torch.floor(sy.max(dim=1).values - 0.5).clamp(-1, alto - 1).long()
    valido &= (x1 >= x0) & (y1 >= y0)

    ids = torch.nonzero(valido).squeeze(1)
    if ids.numel() == 0:
        return salida
    anchos_bb = (x1 - x0 + 1)[ids]
    cuentas = anchos_bb * (y1 - y0 + 1)[ids]

    mejor = torch.full((alto * ancho,), _INT64_MAX, dtype=torch.int64, device=dispositivo)
    acumulado = torch.cumsum(cuentas, 0)
    inicio = 0
    n = ids.numel()
    while inicio < n:
        base = acumulado[inicio - 1] if inicio > 0 else torch.zeros((), dtype=torch.int64, device=dispositivo)
        fin = int(torch.searchsorted(acumulado, base + _PARES_POR_BLOQUE, right=True).item())
        fin = max(fin, inicio + 1)
        t = ids[inicio:fin]
        c = cuentas[inicio:fin]
        total = int(c.sum().item())
        rep = torch.repeat_interleave(torch.arange(t.numel(), device=dispositivo), c)
        desplaz = torch.arange(total, device=dispositivo) - torch.repeat_interleave(torch.cumsum(c, 0) - c, c)
        bw = anchos_bb[inicio:fin][rep]
        tr = t[rep]
        px = x0[tr] + desplaz % bw
        py = y0[tr] + desplaz // bw
        cx = px.float() + 0.5
        cy = py.float() + 0.5

        X, Y, Z = sx[tr], sy[tr], sz[tr]
        a = area[tr]
        b0 = ((X[:, 1] - cx) * (Y[:, 2] - cy) - (X[:, 2] - cx) * (Y[:, 1] - cy)) / a
        b1 = ((X[:, 2] - cx) * (Y[:, 0] - cy) - (X[:, 0] - cx) * (Y[:, 2] - cy)) / a
        b2 = 1 - b0 - b1
        prof = b0 * Z[:, 0] + b1 * Z[:, 1] + b2 * Z[:, 2]
        dentro = (b0 >= 0) & (b1 >= 0) & (b2 >= 0) & (prof >= -1) & (prof <= 1)

        clave = _clave_profundidad(prof[dentro]) + tr[dentro]
        mejor.scatter_reduce_(0, (py * ancho + px)[dentro], clave, reduce="amin")
        del rep, desplaz, bw, tr, px, py, cx, cy, X, Y, Z, a, b0, b1, b2, prof, dentro, clave
        inicio = fin

    pix = torch.nonzero(mejor != _INT64_MAX).squeeze(1)
    if pix.numel() == 0:
        return salida
    t = (mejor[pix] & 0xFFFFFFFF).long()
    cx = (pix % ancho).float() + 0.5
    cy = torch.div(pix, ancho, rounding_mode="floor").float() + 0.5
    X, Y, Z, W = sx[t], sy[t], sz[t], w[t]
    a = area[t]
    b0 = ((X[:, 1] - cx) * (Y[:, 2] - cy) - (X[:, 2] - cx) * (Y[:, 1] - cy)) / a
    b1 = ((X[:, 2] - cx) * (Y[:, 0] - cy) - (X[:, 0] - cx) * (Y[:, 2] - cy)) / a
    b2 = 1 - b0 - b1
    prof = b0 * Z[:, 0] + b1 * Z[:, 1] + b2 * Z[:, 2]
    # Baricentricas con correccion de perspectiva, como nvdiffrast.
    q0, q1, q2 = b0 / W[:, 0], b1 / W[:, 1], b2 / W[:, 2]
    s = q0 + q1 + q2
    plano = salida.view(-1, 4)
    plano[pix, 0] = q0 / s
    plano[pix, 1] = q1 / s
    plano[pix, 2] = prof
    plano[pix, 3] = (t + 1).float()
    return salida


@torch.no_grad()
def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    if ranges is not None:
        raise NotImplementedError("rasterize con 'ranges' no esta soportado en el sustituto AMD")
    alto, ancho = int(resolution[0]), int(resolution[1])
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
    rast = torch.stack([_rasterizar_una(pos[b], tri, alto, ancho) for b in range(pos.shape[0])])
    return rast, torch.zeros_like(rast)


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    if attr.dim() == 2:
        attr = attr.unsqueeze(0)
    lotes, alto, ancho = rast.shape[:3]
    canales = attr.shape[-1]
    tri = tri.long()
    salidas = []
    for b in range(lotes):
        a = attr[b if attr.shape[0] > 1 else 0]
        r = rast[b].reshape(-1, 4)
        tid = r[:, 3].long() - 1
        mascara = tid >= 0
        out = torch.zeros(alto * ancho, canales, dtype=attr.dtype, device=attr.device)
        if mascara.any():
            vt = tri[tid[mascara]]
            u = r[mascara, 0:1].to(attr.dtype)
            v = r[mascara, 1:2].to(attr.dtype)
            out[mascara] = u * a[vt[:, 0]] + v * a[vt[:, 1]] + (1 - u - v) * a[vt[:, 2]]
        salidas.append(out.view(alto, ancho, canales))
    out = torch.stack(salidas)
    if diff_attrs is not None:
        dattr = torch.zeros(lotes, alto, ancho, 2 * canales, dtype=attr.dtype, device=attr.device)
    else:
        dattr = torch.zeros(lotes, alto, ancho, 0, dtype=attr.dtype, device=attr.device)
    return out, dattr


def texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None, filter_mode="auto",
            boundary_mode="wrap", max_mip_level=None):
    if boundary_mode == "cube":
        raise NotImplementedError("texturas cubicas no soportadas en el sustituto AMD")
    if tex.dim() == 3:
        tex = tex.unsqueeze(0)
    lotes = uv.shape[0]
    if tex.shape[0] == 1 and lotes > 1:
        tex = tex.expand(lotes, -1, -1, -1)
    if boundary_mode == "wrap":
        uv = uv - torch.floor(uv)
    else:
        uv = uv.clamp(0, 1)
    modo = "nearest" if filter_mode == "nearest" else "bilinear"
    rejilla = (uv * 2 - 1).to(tex.dtype)
    out = F.grid_sample(tex.permute(0, 3, 1, 2), rejilla, mode=modo,
                        padding_mode="border", align_corners=False)
    return out.permute(0, 2, 3, 1)


def antialias(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
    return color
