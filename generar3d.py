"""Convierte una imagen en un modelo 3D (.glb) usando Hunyuan3D-2.

Uso:
    python generar3d.py foto.png -o modelo.glb             # forma + textura
    python generar3d.py foto.png -o modelo.glb --sin-textura
"""
import argparse
import time

import cv2
import numpy as np
import torch
from PIL import Image

MODELO = "tencent/Hunyuan3D-2mini"
SUBCARPETA = "hunyuan3d-dit-v2-mini-turbo"

_pipeline = None


def quitar_fondo(imagen: Image.Image, umbral: int = 235) -> Image.Image:
    """Vuelve transparente el fondo claro conectado a los bordes.

    Usa relleno por inundación desde los bordes, así los blancos del interior
    (ojos, parches, etc.) se conservan. Si la imagen ya trae transparencia,
    se devuelve tal cual.
    """
    if imagen.mode == "RGBA" and np.array(imagen)[:, :, 3].min() < 250:
        return imagen
    rgb = np.array(imagen.convert("RGB"))
    alto, ancho = rgb.shape[:2]
    fondo = (rgb.min(axis=2) > umbral).astype(np.uint8)
    mascara = np.zeros((alto + 2, ancho + 2), np.uint8)
    semillas = [(0, 0), (ancho - 1, 0), (0, alto - 1), (ancho - 1, alto - 1),
                (ancho // 2, 0), (ancho // 2, alto - 1), (0, alto // 2), (ancho - 1, alto // 2)]
    for x, y in semillas:
        if fondo[y, x] == 1:
            cv2.floodFill(fondo, mascara, (x, y), 2)
    alfa = np.where(fondo == 2, 0, 255).astype(np.uint8)
    alfa = cv2.GaussianBlur(alfa, (3, 3), 0)
    return Image.fromarray(np.dstack([rgb, alfa]), "RGBA")


def elegir_dispositivo(dispositivo: str = "auto") -> str:
    if dispositivo != "auto":
        return dispositivo
    return "cuda" if torch.cuda.is_available() else "cpu"


def cargar_pipeline(dispositivo: str = "auto"):
    """Carga el modelo una sola vez (la primera vez descarga ~3 GB)."""
    global _pipeline
    if _pipeline is None:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        dispositivo = elegir_dispositivo(dispositivo)
        if dispositivo == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODELO,
            subfolder=SUBCARPETA,
            use_safetensors=True,
            variant="fp16",
            device=dispositivo,
            dtype=torch.float16 if dispositivo == "cuda" else torch.float32,
        )
        if dispositivo == "cuda":
            # FlashVDM: decodificador de volumen jerárquico que solo evalúa
            # la red cerca de la superficie. Es el paso que más tardaba.
            _pipeline.enable_flashvdm(topk_mode="merge")
    return _pipeline


def generar_forma(imagen: Image.Image, pasos: int = 5, resolucion: int = 256,
                  semilla: int = 0, dispositivo: str = "auto"):
    """Devuelve la malla (trimesh) generada a partir de una imagen sin fondo."""
    pipeline = cargar_pipeline(dispositivo)
    # En GPU, trozos grandes = muchas menos llamadas a la red (según la VRAM).
    trozos = 20000
    if str(pipeline.device).startswith("cuda"):
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        trozos = 200000 if vram_gb >= 12 else 80000
    return pipeline(
        image=imagen,
        num_inference_steps=pasos,
        octree_resolution=resolucion,
        num_chunks=trozos,
        generator=torch.manual_seed(semilla),
        output_type="trimesh",
    )[0]


def generar(imagen: Image.Image, salida: str, pasos: int = 5, resolucion: int = 256,
            semilla: int = 0, dispositivo: str = "auto", textura: bool = True,
            **opciones_textura) -> str:
    """Genera el modelo 3D de `imagen` y lo guarda en `salida` (.glb, .obj, .stl).

    Con `textura=True` también lo pinta (necesita GPU); `opciones_textura` se
    pasan a `textura.texturizar` (max_caras, resolucion_textura, pasos_vistas...).
    """
    imagen = quitar_fondo(imagen)
    if textura:
        from textura import comprobar_dependencias
        comprobar_dependencias()  # falla antes de gastar tiempo en la forma

    t = time.time()
    malla = generar_forma(imagen, pasos, resolucion, semilla, dispositivo)
    print(f"Forma: {time.time() - t:.1f} s ({len(malla.faces)} caras)")

    t = time.time()
    if textura:
        from textura import texturizar
        malla = texturizar(malla, imagen, **opciones_textura)
        print(f"Textura: {time.time() - t:.1f} s")
    else:
        from textura import preparar_malla
        malla = preparar_malla(malla)
    malla.export(salida)
    return salida


def main():
    p = argparse.ArgumentParser(description="Imagen -> modelo 3D con Hunyuan3D-2")
    p.add_argument("imagen", help="Imagen de entrada (png, jpg, webp)")
    p.add_argument("-o", "--salida", default="modelo.glb", help="Archivo de salida (.glb, .obj, .stl)")
    p.add_argument("--pasos", type=int, default=5, help="Pasos de difusión de la forma (el modelo turbo usa 5)")
    p.add_argument("--resolucion", type=int, default=256,
                   help="Resolución de la malla; en CPU usa 128-192 para que no tarde tanto")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--dispositivo", default="auto", choices=["auto", "cuda", "cpu"])

    t = p.add_argument_group("textura")
    t.add_argument("--sin-textura", action="store_true", help="Solo la forma, sin color")
    t.add_argument("--resolucion-textura", type=int, default=2048, choices=[1024, 2048],
                   help="1024 es más rápido y gasta menos VRAM")
    t.add_argument("--pasos-textura", type=int, default=15,
                   help="Pasos de las vistas de color (el original usa 30)")
    t.add_argument("--max-caras", type=int, default=40000,
                   help="La malla se reduce a estas caras antes de pintarla")
    t.add_argument("--vram-baja", action="store_true", default=None,
                   help="Mueve los modelos a CPU cuando no se usan (auto si la GPU tiene <16 GB)")
    args = p.parse_args()

    inicio = time.time()
    generar(Image.open(args.imagen), args.salida, args.pasos, args.resolucion,
            args.semilla, args.dispositivo, textura=not args.sin_textura,
            max_caras=args.max_caras, resolucion_textura=args.resolucion_textura,
            pasos_vistas=args.pasos_textura, vram_baja=args.vram_baja)
    print(f"Listo: {args.salida} ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    main()
