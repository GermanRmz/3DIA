"""Rasterizador de triángulos en PyTorch puro.

Sustituye al módulo compilado `custom_rasterizer` de Hunyuan3D-2 (CUDA), que
hay que compilar con nvcc y da problemas con compiladores nuevos. Sigue las
mismas reglas que su kernel: centro de píxel en +0.5, prueba de profundidad
con el valor más cercano, mismo desempate y baricéntricas con corrección de
perspectiva. Funciona en cualquier dispositivo de PyTorch (GPU o CPU).

Interfaz igual que `custom_rasterizer`: `rasterize(pos, tri, resolution)`
devuelve (findices, barycentric); findices = índice de cara + 1 (0 = vacío).
"""
import torch

MAXINT = 2147483647
_PARES_POR_TROZO = 1 << 21  # pares (cara, píxel) evaluados a la vez


def _pantalla(V, ancho, alto):
    w = V[:, 3]
    x = (V[:, 0] / w * 0.5 + 0.5) * (ancho - 1) + 0.5
    y = (0.5 + 0.5 * V[:, 1] / w) * (alto - 1) + 0.5
    z = V[:, 2] / w * 0.49999 + 0.5
    return torch.stack([x, y, z], dim=1)


def _area2(a, b, c):
    return (c[..., 0] - a[..., 0]) * (b[..., 1] - a[..., 1]) - (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])


def _baricentricas(a, b, c, p):
    area = _area2(a, b, c)
    inv = 1.0 / torch.where(area == 0, torch.ones_like(area), area)
    beta = _area2(a, p, c) * inv
    gamma = _area2(a, b, p) * inv
    bary = torch.stack([1.0 - beta - gamma, beta, gamma], dim=-1)
    return torch.where((area == 0)[..., None], torch.full_like(bary, -1.0), bary)


@torch.no_grad()
def rasterize(pos, tri, resolution, clamp_depth=None, use_depth_prior=0):
    """pos: (1, N, 4) en coordenadas de recorte; tri: (M, 3); resolution: (alto, ancho)."""
    if use_depth_prior:
        raise NotImplementedError("use_depth_prior no está soportado")
    alto, ancho = int(resolution[0]), int(resolution[1])
    V = pos[0].float()
    F = tri.long()
    dev = V.device
    P = _pantalla(V, ancho, alto)
    t0, t1, t2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]

    xy = torch.stack([t0[:, :2], t1[:, :2], t2[:, :2]], dim=1)
    mn, mx = xy.min(dim=1).values, xy.max(dim=1).values
    # Igual que el kernel: px desde int(x_min) mientras px < x_max + 1.
    x0 = torch.trunc(mn[:, 0]).long().clamp(min=0)
    y0 = torch.trunc(mn[:, 1]).long().clamp(min=0)
    x1 = (torch.ceil(mx[:, 0] + 1) - 1).long().clamp(max=ancho - 1)
    y1 = (torch.ceil(mx[:, 1] + 1) - 1).long().clamp(max=alto - 1)
    bw, bh = (x1 - x0 + 1).clamp(min=0), (y1 - y0 + 1).clamp(min=0)
    validas = (bw > 0) & (bh > 0) & (_area2(t0, t1, t2) != 0)
    caras = torch.nonzero(validas).squeeze(1)

    zbuffer = torch.full((alto * ancho,), MAXINT * MAXINT + (MAXINT - 1), dtype=torch.int64, device=dev)
    cuenta = (bw * bh)[caras]
    acumulado = torch.cumsum(cuenta, 0)
    inicio = 0
    while inicio < len(caras):
        # Grupo de caras cuyo total de píxeles cabe en un trozo (al menos una).
        base = acumulado[inicio - 1] if inicio > 0 else 0
        fin = int(torch.searchsorted(acumulado, base + _PARES_POR_TROZO, right=True))
        fin = max(fin, inicio + 1)
        f = caras[inicio:fin]
        n = cuenta[inicio:fin]
        cara = torch.repeat_interleave(f, n)
        desplaz = torch.arange(int(n.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
        ancho_caja = bw[cara]
        px = x0[cara] + desplaz % ancho_caja
        py = y0[cara] + desplaz // ancho_caja
        p = torch.stack([px.float() + 0.5, py.float() + 0.5], dim=1)
        a, b, c = t0[cara], t1[cara], t2[cara]
        bary = _baricentricas(a[:, :2], b[:, :2], c[:, :2], p)
        dentro = ((bary >= 0) & (bary <= 1)).all(dim=1)
        prof = bary[:, 0] * a[:, 2] + bary[:, 1] * b[:, 2] + bary[:, 2] * c[:, 2]
        dentro &= prof >= 0
        token = (prof[dentro] * (2 << 17)).long() * MAXINT + (cara[dentro] + 1)
        zbuffer.scatter_reduce_(0, (py * ancho + px)[dentro], token, reduce="amin")
        inicio = fin

    f = zbuffer % MAXINT
    vacio = f == (MAXINT - 1)
    findices = torch.where(vacio, torch.zeros_like(f), f)
    # Baricéntricas finales en 2D con corrección de perspectiva (como el kernel).
    idx = (findices - 1).clamp(min=0)
    pix = torch.arange(alto * ancho, device=dev)
    p = torch.stack([(pix % ancho).float() + 0.5, (pix // ancho).float() + 0.5], dim=1)
    Fi = F[idx]
    bary = _baricentricas(P[Fi[:, 0], :2], P[Fi[:, 1], :2], P[Fi[:, 2], :2], p)
    bary = bary / V[Fi, 3]
    bary = bary / bary.sum(dim=1, keepdim=True)
    bary = torch.where(vacio[:, None], torch.zeros_like(bary), bary)
    return findices.int().view(alto, ancho), bary.view(alto, ancho, 3)


def interpolate(col, findices, barycentric, tri):
    """Igual que custom_rasterizer.interpolate."""
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
