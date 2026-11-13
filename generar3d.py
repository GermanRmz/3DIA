"""Convierte una imagen en un modelo 3D (.glb) usando Hunyuan3D-2.

Uso:
    python generar3d.py foto.png -o modelo.glb             # forma + textura
    python generar3d.py foto.png -o modelo.glb --sin-textura
"""
import argparse
import os
import time

import modelos  # noqa: F401  (ajusta los timeouts de descarga antes de importar nada más)

import numpy as np
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

            # Los decodificadores van en fp32: en fp16 algunas GPUs (p. ej. RTX 20xx)
            # pierden la superficie de ciertas imágenes y el modelo sale plano.
            _vae_normal = _pipeline.vae.to(torch.float32)
            _pipeline.vae = ShapeVAE.from_pretrained(
                carpeta, subfolder=modelos.VAE_RAPIDO, use_safetensors=True,
                variant="fp16", device=dispositivo, dtype=torch.float32)
            _pipeline.enable_flashvdm(topk_mode="merge", replace_vae=False)
            if torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 < 8:
                _codificador_bajo_demanda(_pipeline)
    return _pipeline


def _codificador_bajo_demanda(pipeline):
    """En GPUs de menos de 8 GB, el codificador de imagen (DINOv2, 2.3 GB) solo
    está en la GPU mientras codifica; luego pasa a la CPU y deja sitio al resto."""
    original = pipeline.encode_cond

    def encode_cond(*args, **kwargs):
        pipeline.conditioner.to(pipeline.device)
        try:
            return original(*args, **kwargs)
        finally:
            pipeline.conditioner.to("cpu")
            torch.cuda.empty_cache()

    pipeline.encode_cond = encode_cond


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


GROSOR_MINIMO = 0.25  # lado menor / lado mayor; por debajo el modelo se considera "plano"


def grosor(malla) -> float:
    """Lado menor / lado mayor de la caja del modelo (robusto a trozos sueltos)."""
    v = np.asarray(malla.vertices)
    lados = np.percentile(v, 98, axis=0) - np.percentile(v, 2, axis=0)
    return float(lados.min() / max(lados.max(), 1e-9))


def _una_forma(pipeline, imagen, pasos, resolucion, semilla, trozos):
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
        tipo = next(pipeline.vae.parameters()).dtype
        return pipeline._export(latentes.to(tipo), "trimesh", octree_resolution=resolucion,
                                num_chunks=trozos)[0]

    if _vae_normal is None:
        return decodificar()
    # FlashVDM falla con algunas imágenes (no ve la superficie en su malla
    # gruesa); en ese caso esta imagen se decodifica con el método normal.
    try:
        malla = decodificar()
    except (RuntimeError, IndexError):
        malla = None
    if malla is None:
        print("FlashVDM no encontró la superficie; se usa el decodificador normal "
              "para esta imagen (más lento).")
        vae_rapido, pipeline.vae = pipeline.vae, _vae_normal
        torch.cuda.empty_cache()
        try:
            malla = decodificar()
        finally:
            pipeline.vae = vae_rapido
    return malla


def generar_forma(imagen: Image.Image, pasos: int = 5, resolucion: int = 256,
                  semilla: int = 0, dispositivo: str = "auto", intentos: int = 3):
    """Devuelve la malla (trimesh) generada a partir de una imagen sin fondo.

    Si sale plana (como un recorte de cartón), prueba hasta `intentos` semillas
    y se queda con la más gruesa: la difusión solo tarda unos segundos.
    """
    pipeline = cargar_pipeline(dispositivo)
    # En GPU, trozos grandes = menos llamadas a la red, pero más VRAM.
    trozos = 20000
    if str(pipeline.device).startswith("cuda"):
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        trozos = 200000 if vram_gb >= 12 else 60000 if vram_gb >= 8 else 20000

    mejor, mejor_grosor = None, -1.0
    for intento in range(max(intentos, 1)):
        malla = _una_forma(pipeline, imagen, pasos, resolucion, semilla + intento, trozos)
        if malla is None:
            continue
        g = grosor(malla)
        if g > mejor_grosor:
            mejor, mejor_grosor = malla, g
        if g >= GROSOR_MINIMO:
            break
        if intento + 1 < intentos:
            print(f"El modelo salió plano (grosor {g:.0%}); se prueba la semilla {semilla + intento + 1}.")
    if mejor is None:
        raise RuntimeError("El modelo no generó ninguna superficie. Prueba otra imagen "
                           "(un objeto centrado sobre fondo liso) u otra --semilla.")
    if mejor_grosor < GROSOR_MINIMO:
        print(f"Aviso: el modelo sigue siendo plano (grosor {mejor_grosor:.0%}). Suele pasar con "
              "dibujos 2D de colores planos; una imagen con sombras o en vista 3/4 da más volumen.")
    else:
        print(f"Grosor del modelo: {mejor_grosor:.0%}")
    return mejor


def ruta_impresion(salida: str) -> str:
    return os.path.splitext(salida)[0] + "_impresion.stl"


def ruta_color(salida: str, extension: str) -> str:
    return os.path.splitext(salida)[0] + "_impresion_color" + extension


def generar(imagen: Image.Image, salida: str, pasos: int = 5, resolucion: int = 256,
            semilla: int = 0, dispositivo: str = "auto", textura: bool = True,
            fondo: str = "auto", ajustar: bool = True, intentos: int = 3,
            acabado: str | None = None, colores: int = 1,
            altura_mm: float = 60, escala_acabado: float | None = None,
            profundidad_acabado: float | None = None, **opciones_textura) -> str:
    """Genera el modelo 3D de `imagen` y lo guarda en `salida` (.glb, .obj, .stl).

    Con `acabado` (un nombre de acabados.ACABADOS o "ninguno") también guarda
    *_impresion.stl: en mm, Z arriba, apoyado en la cama y con el relieve elegido.
    Si además hay textura y `colores` > 1, guarda *_impresion_color.3mf con cada
    triángulo pintado con uno de `colores` filamentos (y una vista previa .glb).

    Con `textura=True` también lo pinta (necesita GPU); `opciones_textura` se
    pasan a `textura.texturizar` (max_caras, resolucion_textura...).
    La imagen ya preparada se guarda junto a la salida como *_entrada.png.
    """
    imagen, notas = preparar(imagen, fondo, ajustar)
    for nota in notas:
        print(nota)
    imagen.save(os.path.splitext(salida)[0] + "_entrada.png")
    if textura:
        from textura import comprobar_dependencias
        comprobar_dependencias()  # falla antes de gastar tiempo en la forma

    t = time.time()
    malla = generar_forma(imagen, pasos, resolucion, semilla, dispositivo, intentos)
    print(f"Forma: {time.time() - t:.1f} s ({len(malla.faces)} caras)")

    from textura import preparar_malla
    malla = preparar_malla(malla)  # quita trozos sueltos y caras degeneradas

    if acabado is not None:
        import acabados

        t = time.time()
        avisos = []
        impresion = acabados.aplicar(malla, acabado, altura_mm, escala_acabado,
                                     profundidad_acabado, avisos)
        for aviso in avisos:
            print(aviso)
        impresion.export(ruta_impresion(salida))
        print(f"Impresión: {ruta_impresion(salida)} ({acabado}, {impresion.extents.round(1)} mm, "
              f"{len(impresion.faces)} caras, {time.time() - t:.1f} s)")

    t = time.time()
    if textura:
        from textura import texturizar
        liberar_gpu_forma()
        malla = texturizar(malla, imagen, **opciones_textura)
        print(f"Textura: {time.time() - t:.1f} s")
        if acabado is not None and colores > 1:
            import multicolor

            t = time.time()
            paleta = multicolor.pintar_impresion(impresion, malla, colores,
                                                 ruta_color(salida, ".3mf"), ruta_color(salida, ".glb"))
            lista = ", ".join(f"{i + 1}: #{r:02X}{g:02X}{b:02X}" for i, (r, g, b) in enumerate(paleta))
            print(f"Multicolor: {ruta_color(salida, '.3mf')} ({time.time() - t:.1f} s). "
                  f"Filamentos -> {lista}")
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
    p.add_argument("--intentos", type=int, default=3,
                   help="Si el modelo sale plano, cuántas semillas probar (1 = no reintentar)")
    p.add_argument("--dispositivo", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--fondo", default="auto", choices=["auto", "ia", "blanco", "mantener"],
                   help="Cómo quitar el fondo: auto (transparencia si la hay, si no IA), "
                        "ia, blanco (relleno del fondo claro, sin IA) o mantener (no tocarlo)")
    p.add_argument("--sin-ajustar", action="store_true",
                   help="No recortar, centrar ni limpiar la imagen (solo quitar el fondo)")

    t = p.add_argument_group("textura")
    t.add_argument("--sin-textura", action="store_true", help="Solo la forma, sin color")
    t.add_argument("--resolucion-textura", type=int, default=2048, choices=[1024, 2048],
                   help="1024 es más rápido y gasta menos VRAM")
    t.add_argument("--max-caras", type=int, default=40000,
                   help="La malla se reduce a estas caras antes de pintarla")
    t.add_argument("--vram-baja", action="store_true", default=None,
                   help="Mueve los modelos a CPU cuando no se usan (auto si la GPU tiene <16 GB)")
    import acabados
    i = p.add_argument_group("impresión 3D (genera <salida>_impresion.stl)")
    i.add_argument("--acabado", choices=["ninguno"] + list(acabados.ACABADOS),
                   help="Relieve de la superficie: " + ", ".join(
                       f"{k} ({v[3].lower()})" for k, v in acabados.ACABADOS.items()))
    i.add_argument("--colores", type=int, default=1,
                   help="Filamentos para impresora multicolor (2-16): genera un 3MF pintado. "
                        "Necesita textura")
    i.add_argument("--altura-mm", type=float, default=60, help="Altura de la pieza impresa (mm)")
    i.add_argument("--escala-acabado", type=float, help="Tamaño del patrón en mm")
    i.add_argument("--profundidad-acabado", type=float, help="Profundidad del relieve en mm")
    args = p.parse_args()

    inicio = time.time()
    generar(Image.open(args.imagen), args.salida, args.pasos, args.resolucion,
            args.semilla, args.dispositivo, textura=not args.sin_textura, fondo=args.fondo,
            ajustar=not args.sin_ajustar, intentos=args.intentos, acabado=args.acabado,
            colores=args.colores, altura_mm=args.altura_mm,
            escala_acabado=args.escala_acabado, profundidad_acabado=args.profundidad_acabado,
            max_caras=args.max_caras, resolucion_textura=args.resolucion_textura,
            vram_baja=args.vram_baja)
    print(f"Listo: {args.salida} ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    main()
