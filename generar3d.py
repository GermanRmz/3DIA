"""Convierte una imagen en un modelo 3D (.glb) usando Hunyuan3D-2.

Uso:
    python generar3d.py foto.png -o modelo.glb             # forma + textura
    python generar3d.py foto.png -o modelo.glb --sin-textura
"""
import argparse
import os
import time

import modelos  # noqa: F401  (ajusta los timeouts de descarga antes de importar nada más)

import torch
from PIL import Image

from preparar_imagen import preparar

_pipeline = None
_vae_normal = None  # decodificador original, por si FlashVDM falla


def elegir_dispositivo(dispositivo: str = "auto") -> str:
    if dispositivo != "auto":
        return dispositivo
    return "cuda" if torch.cuda.is_available() else "cpu"


def cargar_pipeline(dispositivo: str = "auto"):
    """Carga el modelo una sola vez (la primera vez descarga ~4 GB)."""
    global _pipeline, _vae_normal
    if _pipeline is not None and _pipeline.device.type == "cpu" and _en_gpu_antes:
        _mover_forma("cuda")
    if _pipeline is None:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        dispositivo = elegir_dispositivo(dispositivo)
        en_gpu = dispositivo == "cuda"
        if en_gpu:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        carpeta = modelos.carpeta_forma(con_vae_rapido=en_gpu)
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            carpeta,
            subfolder=modelos.DIT_FORMA,
            use_safetensors=True,
            variant="fp16",
            device=dispositivo,
            dtype=torch.float16 if en_gpu else torch.float32,
        )
        if en_gpu:
            # FlashVDM: decodificador de volumen jerárquico que solo evalúa
            # la red cerca de la superficie. Es el paso que más tardaba.
            from hy3dgen.shapegen.models import ShapeVAE

            _vae_normal = _pipeline.vae
            _pipeline.vae = ShapeVAE.from_pretrained(
                carpeta, subfolder=modelos.VAE_RAPIDO, use_safetensors=True,
                variant="fp16", device=dispositivo, dtype=torch.float16)
            _pipeline.enable_flashvdm(topk_mode="merge", replace_vae=False)
    return _pipeline


_en_gpu_antes = False


def _mover_forma(dispositivo: str):
    _pipeline.to(dispositivo)
    if _vae_normal is not None:
        _vae_normal.to(dispositivo)


def liberar_gpu_forma():
    """Pasa el modelo de forma a la CPU para dejar la VRAM a la textura (GPUs <12 GB)."""
    global _en_gpu_antes
    if _pipeline is None or _pipeline.device.type != "cuda":
        return
    if torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 < 12:
        _mover_forma("cpu")
        _en_gpu_antes = True
        torch.cuda.empty_cache()


def generar_forma(imagen: Image.Image, pasos: int = 5, resolucion: int = 256,
                  semilla: int = 0, dispositivo: str = "auto"):
    """Devuelve la malla (trimesh) generada a partir de una imagen sin fondo."""
    pipeline = cargar_pipeline(dispositivo)
    # En GPU, trozos grandes = menos llamadas a la red, pero más VRAM.
    trozos = 20000
    if str(pipeline.device).startswith("cuda"):
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        trozos = 200000 if vram_gb >= 12 else 60000 if vram_gb >= 8 else 20000

    latentes = pipeline(
        image=imagen,
        num_inference_steps=pasos,
        octree_resolution=resolucion,
        num_chunks=trozos,
        generator=torch.manual_seed(semilla),
        output_type="latent",
    )
    if not torch.isfinite(latentes).all():
        raise RuntimeError("La difusión dio valores inválidos (NaN). Prueba con --dispositivo cpu "
                           "para comprobar si es un problema de la GPU.")

    def decodificar():
        return pipeline._export(latentes, "trimesh", octree_resolution=resolucion,
                                num_chunks=trozos)[0]

    global _vae_normal
    malla = decodificar()
    if malla is None and _vae_normal is not None:
        print("FlashVDM no encontró superficie; se repite con el decodificador normal "
              "(más lento) y se usará ese a partir de ahora.")
        pipeline.vae, _vae_normal = _vae_normal, None
        torch.cuda.empty_cache()
        malla = decodificar()
    if malla is None:
        raise RuntimeError("El modelo no generó ninguna superficie. Prueba otra imagen "
                           "(un objeto centrado sobre fondo liso) u otra --semilla.")
    return malla


def generar(imagen: Image.Image, salida: str, pasos: int = 5, resolucion: int = 256,
            semilla: int = 0, dispositivo: str = "auto", textura: bool = True,
            fondo: str = "auto", **opciones_textura) -> str:
    """Genera el modelo 3D de `imagen` y lo guarda en `salida` (.glb, .obj, .stl).

    Con `textura=True` también lo pinta (necesita GPU); `opciones_textura` se
    pasan a `textura.texturizar` (max_caras, resolucion_textura...).
    La imagen ya preparada se guarda junto a la salida como *_entrada.png.
    """
    imagen, notas = preparar(imagen, fondo)
    for nota in notas:
        print(nota)
    imagen.save(os.path.splitext(salida)[0] + "_entrada.png")
    if textura:
        from textura import comprobar_dependencias
        comprobar_dependencias()  # falla antes de gastar tiempo en la forma

    t = time.time()
    malla = generar_forma(imagen, pasos, resolucion, semilla, dispositivo)
    print(f"Forma: {time.time() - t:.1f} s ({len(malla.faces)} caras)")

    t = time.time()
    if textura:
        from textura import texturizar
        liberar_gpu_forma()
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
    p.add_argument("--fondo", default="auto", choices=["auto", "ia", "blanco", "mantener"],
                   help="Cómo quitar el fondo: auto (decide solo), ia (fotos con fondo "
                        "complejo), blanco (fondo liso claro) o mantener (no tocarlo)")

    t = p.add_argument_group("textura")
    t.add_argument("--sin-textura", action="store_true", help="Solo la forma, sin color")
    t.add_argument("--resolucion-textura", type=int, default=2048, choices=[1024, 2048],
                   help="1024 es más rápido y gasta menos VRAM")
    t.add_argument("--max-caras", type=int, default=40000,
                   help="La malla se reduce a estas caras antes de pintarla")
    t.add_argument("--vram-baja", action="store_true", default=None,
                   help="Mueve los modelos a CPU cuando no se usan (auto si la GPU tiene <16 GB)")
    args = p.parse_args()

    inicio = time.time()
    generar(Image.open(args.imagen), args.salida, args.pasos, args.resolucion,
            args.semilla, args.dispositivo, textura=not args.sin_textura, fondo=args.fondo,
            max_caras=args.max_caras, resolucion_textura=args.resolucion_textura,
            vram_baja=args.vram_baja)
    print(f"Listo: {args.salida} ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    main()
