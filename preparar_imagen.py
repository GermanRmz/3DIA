"""Análisis y preparación de la imagen antes de generar el 3D.

1. Corrige la rotación de las fotos de móvil (EXIF) y reduce las enormes.
2. Quita el fondo:
   - si la imagen ya trae transparencia, la respeta;
   - si el fondo es claro y uniforme, usa relleno por inundación desde los
     bordes (bordes precisos) y la IA para los huecos de fondo encerrados;
   - si no, usa un modelo de IA de segmentación (rembg, isnet-general-use).
3. Limpia la máscara: quita manchas sueltas.
4. Recorta alrededor del objeto con margen y lo deja cuadrado.
5. Avisa de problemas: objeto cortado por el borde, objeto muy pequeño...
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

LADO_MAXIMO = 1536      # las fotos más grandes se reducen (los modelos usan ~512)
LADO_MINIMO_OBJETO = 300
MODELO_IA = "isnet-general-use"

_sesion_ia = None


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


def _fondo_por_ia(imagen: Image.Image) -> np.ndarray:
    global _sesion_ia
    from rembg import new_session, remove

    if _sesion_ia is None:
        _sesion_ia = new_session(MODELO_IA)  # la primera vez descarga ~180 MB
    return np.array(remove(imagen.convert("RGB"), session=_sesion_ia, only_mask=True))


def _fondo_claro_uniforme(rgb: np.ndarray) -> bool:
    borde = np.concatenate([rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3),
                            rgb[:, :3].reshape(-1, 3), rgb[:, -3:].reshape(-1, 3)])
    return (borde.min(axis=1) > 235).mean() > 0.9


def _limpiar(alfa: np.ndarray) -> np.ndarray:
    """Quita manchas sueltas: componentes de menos del 2 % del mayor."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats((alfa > 127).astype(np.uint8), 8)
    if n <= 2:
        return alfa
    areas = stats[1:, cv2.CC_STAT_AREA]
    pequenas = np.nonzero(areas < 0.02 * areas.max())[0] + 1
    alfa = alfa.copy()
    alfa[np.isin(etiquetas, pequenas)] = 0
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


def preparar(imagen: Image.Image, fondo: str = "auto") -> tuple[Image.Image, list[str]]:
    """Devuelve (imagen RGBA lista para el modelo, lista de avisos/notas).

    fondo: "auto", "ia" (siempre IA), "blanco" (relleno por inundación) o
    "mantener" (usar la imagen tal cual, con su transparencia si la tiene).
    """
    notas = []
    imagen = ImageOps.exif_transpose(imagen)
    if max(imagen.size) > LADO_MAXIMO:
        imagen.thumbnail((LADO_MAXIMO, LADO_MAXIMO), Image.LANCZOS)

    rgba = np.array(imagen.convert("RGBA"))
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    trae_alfa = imagen.mode in ("RGBA", "LA", "P") and rgba[:, :, 3].min() < 250

    if fondo == "mantener" or (fondo == "auto" and trae_alfa):
        alfa = rgba[:, :, 3]
        notas.append("Fondo: se usa la transparencia de la imagen.")
    elif fondo == "blanco" or (fondo == "auto" and _fondo_claro_uniforme(rgb)):
        alfa = _fondo_por_inundacion(rgb)
        if fondo == "auto":
            # El relleno no llega al fondo encerrado por el objeto (p. ej. entre
            # un brazo y el cuerpo). Lo que la IA ve como fondo y además es
            # blanco se quita; lo blanco que la IA ve como objeto se conserva.
            ia = _fondo_por_ia(imagen)
            claro = rgb.min(axis=2) > 235
            alfa = np.where((ia < 128) & claro & (alfa > 127), 0, alfa).astype(np.uint8)
        objeto = (alfa > 127).mean()
        if fondo == "auto" and not 0.01 < objeto < 0.98:
            alfa = _fondo_por_ia(imagen)
            notas.append("Fondo: el relleno del fondo claro falló; se quitó con IA.")
        else:
            notas.append("Fondo: claro y uniforme, quitado por relleno.")
    else:
        alfa = _fondo_por_ia(imagen)
        notas.append("Fondo: complejo, quitado con IA.")

    alfa = _limpiar(alfa)
    objeto = (alfa > 127).mean()
    if objeto < 0.005:
        raise RuntimeError("No se encontró ningún objeto en la imagen. Usa una foto con un solo "
                           "objeto bien visible.")

    bordes = np.concatenate([alfa[0], alfa[-1], alfa[:, 0], alfa[:, -1]])
    if (bordes > 127).mean() > 0.03:
        notas.append("Aviso: el objeto toca el borde de la foto (¿está cortado?). "
                     "El 3D sale mejor si se ve completo.")

    resultado = _recortar_cuadrado(np.dstack([rgb, alfa]))
    lado_objeto = int(resultado.shape[0] / 1.24)
    if lado_objeto < LADO_MINIMO_OBJETO:
        notas.append(f"Aviso: el objeto mide solo ~{lado_objeto} px; con una foto de más "
                     "resolución saldrán más detalles.")
    return Image.fromarray(resultado, "RGBA"), notas
