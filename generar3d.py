"""Convierte una imagen en un modelo 3D (.glb) usando Hunyuan3D-2.

Uso:
    python generar3d.py foto.png -o modelo.glb
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
    """Carga el modelo una sola vez (la primera vez descarga ~8 GB)."""
    global _pipeline
    if _pipeline is None:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        dispositivo = elegir_dispositivo(dispositivo)
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODELO,
            subfolder=SUBCARPETA,
            use_safetensors=True,
            variant="fp16",
            device=dispositivo,
            dtype=torch.float16 if dispositivo == "cuda" else torch.float32,
        )
    return _pipeline


def generar(imagen: Image.Image, salida: str, pasos: int = 5, resolucion: int = 256,
            semilla: int = 0, dispositivo: str = "auto") -> str:
    """Genera la malla 3D de `imagen` y la guarda en `salida` (.glb, .obj, .stl)."""
    pipeline = cargar_pipeline(dispositivo)
    malla = pipeline(
        image=quitar_fondo(imagen),
        num_inference_steps=pasos,
        octree_resolution=resolucion,
        num_chunks=20000,
        generator=torch.manual_seed(semilla),
        output_type="trimesh",
    )[0]
    malla.export(salida)
    return salida


def main():
    p = argparse.ArgumentParser(description="Imagen -> modelo 3D con Hunyuan3D-2")
    p.add_argument("imagen", help="Imagen de entrada (png, jpg, webp)")
    p.add_argument("-o", "--salida", default="modelo.glb", help="Archivo de salida (.glb, .obj, .stl)")
    p.add_argument("--pasos", type=int, default=5, help="Pasos de difusión (el modelo turbo usa 5)")
    p.add_argument("--resolucion", type=int, default=256,
                   help="Resolución de la malla; en CPU usa 128-192 para que no tarde tanto")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--dispositivo", default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    inicio = time.time()
    generar(Image.open(args.imagen), args.salida, args.pasos, args.resolucion,
            args.semilla, args.dispositivo)
    print(f"Listo: {args.salida} ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    main()
