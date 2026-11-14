"""Análisis y preparación de la imagen antes de generar el 3D.

Son dos pasos independientes (cada uno se puede desactivar):

Quitar el fondo (`fondo`):
  - "auto": respeta la transparencia si la imagen ya la trae; si no, usa IA.
  - "ia": modelo de segmentación BiRefNet (rembg, "birefnet-general-lite"). A
    diferencia de los métodos por color, distingue partes blancas del objeto
    (una cara, una barriga) de un fondo blanco, y el fondo encerrado entre
    brazo y cuerpo.
  - "blanco": relleno por inundación del fondo claro desde los bordes (rápido,
    sin IA; no quita el fondo encerrado por el objeto).
  - "mantener": no se toca.

Ajustar la imagen (`ajustar`):
  - Corrige la rotación de fotos de móvil (EXIF) y reduce las muy grandes.
  - Quita manchas sueltas de la máscara y la semitransparencia del interior.
  - Recorta alrededor del objeto con margen y lo deja cuadrado.
  - Avisa si el objeto está cortado por el borde.
  - Si el objeto es pequeño (menos de 768 px), lo amplía x4 con IA
    (superres.py, Real-ESRGAN): más detalle en la forma y en la textura.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

LADO_MAXIMO = 1536      # las fotos más grandes se reducen (los modelos usan ~512)
LADO_MINIMO_OBJETO = 300
LADO_AMPLIAR = 768      # por debajo, el objeto se amplía con IA
MODELO_IA = "birefnet-general-lite"

_sesion_ia = None


def _fondo_por_ia(imagen: Image.Image) -> np.ndarray:
    global _sesion_ia
    from rembg import new_session, remove

    if _sesion_ia is None:
        import onnxruntime as ort

        # Sin "arena" de memoria: la RAM usada al segmentar se libera al terminar
        # en vez de quedarse reservada (el pico ronda los 7 GB).
        opciones = ort.SessionOptions()
        opciones.enable_cpu_mem_arena = False
        opciones.enable_mem_pattern = False
        _sesion_ia = new_session(MODELO_IA, sess_opts=opciones)  # la primera vez descarga ~220 MB
    return np.array(remove(imagen.convert("RGB"), session=_sesion_ia, only_mask=True))


def _fondo_por_inundacion(rgb: np.ndarray, umbral: int = 235) -> np.ndarray:
    """Alfa (uint8) quitando el fondo claro conectado a los bordes."""
    alto, ancho = rgb.shape[:2]
    fondo = (rgb.min(axis=2) > umbral).astype(np.uint8)
    mascara = np.zeros((alto + 2, ancho + 2), np.uint8)
    borde = [(x, 0) for x in range(0, ancho, 8)] + [(x, alto - 1) for x in range(0, ancho, 8)] + \
            [(0, y) for y in range(0, alto, 8)] + [(ancho - 1, y) for y in range(0, alto, 8)]
    for x, y in borde:
        if fondo[y, x] == 1:
            cv2.floodFill(fondo, mascara, (x, y), 2)
    alfa = np.where(fondo == 2, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(alfa, (3, 3), 0)


def _limpiar(alfa: np.ndarray) -> np.ndarray:
    """Quita manchas sueltas (componentes de menos del 2 % del mayor) y deja
    opaco el interior del objeto; solo el contorno conserva su suavizado."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats((alfa > 127).astype(np.uint8), 8)
    alfa = alfa.copy()
    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        alfa[np.isin(etiquetas, np.nonzero(areas < 0.02 * areas.max())[0] + 1)] = 0
    interior = cv2.erode((alfa > 127).astype(np.uint8), np.ones((7, 7), np.uint8))
    alfa[interior > 0] = 255
    return alfa


def _recortar_cuadrado(rgba: np.ndarray, margen: float = 0.12) -> np.ndarray:
    ys, xs = np.nonzero(rgba[:, :, 3] > 127)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lado = int(max(y1 - y0, x1 - x0) * (1 + 2 * margen))
    salida = np.zeros((lado, lado, 4), np.uint8)
    salida[:, :, :3] = 255
    oy, ox = (lado - (y1 - y0)) // 2, (lado - (x1 - x0)) // 2
    salida[oy:oy + y1 - y0, ox:ox + x1 - x0] = rgba[y0:y1, x0:x1]
    return salida


def quitar_fondo(imagen: Image.Image, fondo: str = "auto") -> tuple[np.ndarray, str]:
    """Devuelve (alfa uint8, nota) según el método elegido."""
    rgba = np.array(imagen.convert("RGBA"))
    trae_alfa = imagen.mode in ("RGBA", "LA", "P") and rgba[:, :, 3].min() < 250
    if fondo == "mantener" or (fondo == "auto" and trae_alfa):
        return rgba[:, :, 3], "Fondo: se usa la transparencia de la imagen (o no se toca)."
    if fondo == "blanco":
        return _fondo_por_inundacion(np.ascontiguousarray(rgba[:, :, :3])), \
            "Fondo: claro, quitado por relleno desde los bordes."
    return _fondo_por_ia(imagen), "Fondo: quitado con IA."


def preparar(imagen: Image.Image, fondo: str = "auto", ajustar: bool = True,
             superres: bool = True) -> tuple[Image.Image, list[str]]:
    """Devuelve (imagen RGBA lista para el modelo, lista de notas y avisos). Con
    `ajustar` y `superres`, un objeto pequeño se amplía con IA."""
    if ajustar:
        imagen = ImageOps.exif_transpose(imagen)
        if max(imagen.size) > LADO_MAXIMO:
            imagen.thumbnail((LADO_MAXIMO, LADO_MAXIMO), Image.LANCZOS)

    alfa, nota = quitar_fondo(imagen, fondo)
    notas = [nota]
    if ajustar:
        alfa = _limpiar(alfa)
    if (alfa > 127).mean() < 0.005:
        raise RuntimeError("No se encontró ningún objeto en la imagen. Usa una foto con un solo "
                           "objeto bien visible.")
    rgba = np.dstack([np.array(imagen.convert("RGB")), alfa])
    if not ajustar:
        return Image.fromarray(rgba, "RGBA"), notas

    bordes = np.concatenate([alfa[0], alfa[-1], alfa[:, 0], alfa[:, -1]])
    if (bordes > 127).mean() > 0.03:
        notas.append("Aviso: el objeto toca el borde de la foto (¿está cortado?). "
                     "El 3D sale mejor si se ve completo.")
    resultado = Image.fromarray(_recortar_cuadrado(rgba), "RGBA")
    lado_objeto = int(resultado.width / 1.24)
    if superres and resultado.width < LADO_AMPLIAR:
        try:
            import superres as sr

            antes = resultado.width
            resultado = sr.ampliar(resultado)
            sr.liberar()
            if resultado.width > LADO_MAXIMO:
                resultado.thumbnail((LADO_MAXIMO, LADO_MAXIMO), Image.LANCZOS)
            notas.append(f"Imagen pequeña ({antes} px): ampliada con IA a {resultado.width} px "
                         "para ganar detalle.")
        except Exception as e:  # sin red la primera vez, etc.: se sigue sin ampliar
            notas.append(f"No se pudo ampliar la imagen con IA ({e}).")
    if lado_objeto < LADO_MINIMO_OBJETO:
        notas.append(f"Aviso: el objeto mide solo ~{lado_objeto} px en la foto original; con una "
                     "imagen de más resolución saldrán más detalles reales.")
    return resultado, notas


def liberar():
    """Descarga el modelo de quitar fondos (se recarga al volver a usarlo)."""
    global _sesion_ia
    _sesion_ia = None
