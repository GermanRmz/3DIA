"""Superresolución x4 con IA (Real-ESRGAN) para ganar detalle.

Se usa en dos sitios:
- Imágenes de entrada pequeñas: una foto de 200 px estirada a lo que usan los
  modelos (~512 px) queda borrosa; ampliada con IA conserva bordes y detalles,
  y la forma y la textura salen más definidas.
- Vistas de la textura: Hunyuan3D-Paint pinta 6 vistas de 512 px y las estira
  a la resolución de la textura (2048) con un reescalado simple, que la deja
  borrosa. Ampliarlas con IA da una textura más nítida (es lo que hace
  Hunyuan3D 2.1).

El modelo pesa 67 MB, se descarga la primera vez y procesa la imagen por trozos,
así que cabe en GPUs pequeñas (y funciona, más lento, en CPU).
"""
import numpy as np
import torch
from PIL import Image

REPO = "Comfy-Org/Real-ESRGAN_repackaged"
ARCHIVO = "RealESRGAN_x4plus.safetensors"
ESCALA = 4

_modelo = None


def _cargar(dispositivo: str):
    global _modelo
    if _modelo is None:
        import spandrel
        from huggingface_hub import hf_hub_download

        import modelos  # noqa: F401  (timeouts de descarga)

        ruta = hf_hub_download(REPO, ARCHIVO)
        _modelo = spandrel.ModelLoader().load_from_file(ruta).eval()
    _modelo.to(torch.device(dispositivo))
    if dispositivo == "cuda" and _modelo.supports_half:
        _modelo.half()
    else:
        _modelo.float()
    return _modelo


@torch.inference_mode()
def ampliar(imagen: Image.Image, dispositivo: str = "auto", trozo: int = 256,
            solape: int = 16) -> Image.Image:
    """Devuelve `imagen` ampliada x4. El canal alfa (si lo hay) se amplía con un
    reescalado de calidad (la IA solo trabaja el color)."""
    if dispositivo == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = _cargar(dispositivo)
    tipo = next(modelo.model.parameters()).dtype
    rgb = np.asarray(imagen.convert("RGB"), np.float32) / 255.0
    alto, ancho = rgb.shape[:2]
    entrada = torch.from_numpy(rgb).permute(2, 0, 1)[None]
    salida = torch.zeros(1, 3, alto * ESCALA, ancho * ESCALA)
    # Por trozos con solape: cada trozo se procesa con un margen que luego se
    # descarta, para que no se noten las costuras.
    for y in range(0, alto, trozo):
        for x in range(0, ancho, trozo):
            y0, x0 = max(0, y - solape), max(0, x - solape)
            y1, x1 = min(alto, y + trozo + solape), min(ancho, x + trozo + solape)
            parte = entrada[:, :, y0:y1, x0:x1].to(dispositivo, tipo)
            res = modelo(parte).float().cpu()
            yy, xx = (y - y0) * ESCALA, (x - x0) * ESCALA
            h, w = min(trozo, alto - y) * ESCALA, min(trozo, ancho - x) * ESCALA
            salida[:, :, y * ESCALA:y * ESCALA + h, x * ESCALA:x * ESCALA + w] = res[:, :, yy:yy + h, xx:xx + w]
    color = Image.fromarray((salida[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8))
    if imagen.mode == "RGBA":
        alfa = imagen.getchannel("A").resize(color.size, Image.LANCZOS)
        color.putalpha(alfa)
    return color


def liberar():
    """Saca el modelo de la GPU (ocupa poco, pero deja sitio a los demás)."""
    if _modelo is not None:
        _modelo.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
