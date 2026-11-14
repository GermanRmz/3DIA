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
_modelo_cargado = None  # "mini" o "estandar"
_vae_normal = None  # decodificador original, por si FlashVDM falla
_gpu_pequena = False  # <8 GB: codificador y decodificador normal viven en la CPU


def elegir_dispositivo(dispositivo: str = "auto") -> str:
    if dispositivo != "auto":
        return dispositivo
    return "cuda" if torch.cuda.is_available() else "cpu"


def descargar_forma():
    """Quita el modelo de forma de la memoria (para cambiar de modelo)."""
    global _pipeline, _vae_normal, _modelo_cargado, _en_gpu_antes
    _pipeline, _vae_normal, _modelo_cargado, _en_gpu_antes = None, None, None, False
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cargar_pipeline(dispositivo: str = "auto", modelo: str = "mini"):
    """Carga el modelo una sola vez (la primera vez descarga ~4 GB el mini, ~5 GB
    el estándar). Cambiar de modelo descarga el anterior de la memoria."""
    global _pipeline, _vae_normal, _gpu_pequena, _modelo_cargado
    if _pipeline is not None and _modelo_cargado != modelo:
        descargar_forma()
    if _pipeline is not None and _pipeline.device.type == "cpu" and _en_gpu_antes:
        _mover_forma("cuda")
    if _pipeline is None:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        dispositivo = elegir_dispositivo(dispositivo)
        en_gpu = dispositivo == "cuda"
        if en_gpu:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        carpeta = modelos.carpeta_forma(con_vae_rapido=en_gpu, modelo=modelo)
        _, dit, vae_rapido = modelos.MODELOS_FORMA[modelo]
        _modelo_cargado = modelo
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            carpeta,
            subfolder=dit,
            use_safetensors=True,
            variant="fp16",
            device=dispositivo,
            dtype=torch.float16 if en_gpu else torch.float32,
        )
        if en_gpu:
            # FlashVDM: decodificador de volumen jerárquico que solo evalúa
            # la red cerca de la superficie. Es el paso que más tardaba.
            from hy3dgen.shapegen.models import ShapeVAE

            _gpu_pequena = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 < 8
            if _gpu_pequena:
                # El codificador de imagen (2.3 GB) solo sube a la GPU para codificar.
                _pipeline.conditioner.to("cpu")
                _codificador_bajo_demanda(_pipeline)
            # El decodificador normal (de reserva y para las pruebas rápidas de
            # grosor) espera en la CPU si la GPU es pequeña.
            _vae_normal = _pipeline.vae.to("cpu" if _gpu_pequena else dispositivo)
            torch.cuda.empty_cache()
            _pipeline.vae = ShapeVAE.from_pretrained(
                carpeta, subfolder=vae_rapido, use_safetensors=True,
                variant="fp16", device=dispositivo, dtype=torch.float16)
            _pipeline.enable_flashvdm(topk_mode="merge", replace_vae=False)
        _cachear_codificacion(_pipeline)
    return _pipeline


def _cachear_codificacion(pipeline):
    """Los intentos con otras semillas usan la misma imagen: se codifica una vez."""
    original = pipeline.encode_cond
    cache = {}

    def encode_cond(image, *args, **kwargs):
        import hashlib
        clave = hashlib.sha1(image.detach().cpu().numpy().tobytes()).hexdigest()
        if clave not in cache:
            cache.clear()
            cache[clave] = original(image, *args, **kwargs)
        return cache[clave]

    pipeline.encode_cond = encode_cond


def _codificador_bajo_demanda(pipeline):
    """En GPUs de menos de 8 GB, el codificador de imagen (DINOv2, 2.3 GB) solo
    está en la GPU mientras codifica; luego pasa a la CPU y deja sitio al resto."""
    original = pipeline.encode_cond

    def encode_cond(*args, **kwargs):
        # Con el modelo estándar (2.2 GB) no caben a la vez generador y
        # codificador en 6 GB: el generador espera en la CPU mientras se codifica.
        apartar = _modelo_cargado == "estandar"
        if apartar:
            pipeline.model.to("cpu")
            torch.cuda.empty_cache()
        pipeline.conditioner.to(pipeline.device)
        try:
            return original(*args, **kwargs)
        finally:
            pipeline.conditioner.to("cpu")
            torch.cuda.empty_cache()
            if apartar:
                pipeline.model.to(pipeline.device)

    pipeline.encode_cond = encode_cond


_en_gpu_antes = False


def _mover_forma(dispositivo: str):
    if dispositivo == "cuda" and _gpu_pequena:
        # Solo el generador y el decodificador rápido; el resto sube bajo demanda.
        _pipeline.model.to(dispositivo)
        _pipeline.vae.to(dispositivo)
        _pipeline.device = torch.device(dispositivo)
        return
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


def _latentes(pipeline, imagen, pasos, semilla, trozos):
    latentes = pipeline(
        image=imagen,
        num_inference_steps=pasos,
        num_chunks=trozos,
        generator=torch.manual_seed(semilla),
        output_type="latent",
        enable_pbar=False,
    )
    if not torch.isfinite(latentes).all():
        raise RuntimeError("La difusión dio valores inválidos (NaN). Prueba con --dispositivo cpu "
                           "para comprobar si es un problema de la GPU.")
    return latentes


def _exportar(pipeline, vae, latentes, resolucion, trozos):
    anterior, pipeline.vae = pipeline.vae, vae
    try:
        tipo = next(vae.parameters()).dtype
        return pipeline._export(latentes.to(tipo), "trimesh", octree_resolution=resolucion,
                                num_chunks=trozos, enable_pbar=False)[0]
    finally:
        pipeline.vae = anterior


def _vae_de_reserva(pipeline):
    """Decodificador normal (sin FlashVDM), en la GPU si hace falta."""
    if _vae_normal is None:
        return pipeline.vae
    return _vae_normal.to(pipeline.device)


def _guardar_reserva():
    if _gpu_pequena and _vae_normal is not None:
        _vae_normal.to("cpu")
        torch.cuda.empty_cache()


def _grosor_rapido(pipeline, latentes, trozos) -> float:
    """Grosor de una decodificación de muy baja resolución (unos segundos).
    Una lámina plana puede ni aparecer a esta resolución: cuenta como grosor 0."""
    try:
        malla = _exportar(pipeline, _vae_de_reserva(pipeline), latentes, 64, trozos)
    except (RuntimeError, IndexError):
        return 0.0
    return 0.0 if malla is None else grosor(malla)


def _decodificar(pipeline, latentes, resolucion, trozos):
    """Decodificación completa: FlashVDM y, si no ve la superficie, la normal."""
    if _vae_normal is not None:
        try:
            malla = _exportar(pipeline, pipeline.vae, latentes, resolucion, trozos)
        except (RuntimeError, IndexError):
            malla = None
        if malla is not None:
            return malla
        print("FlashVDM no encontró la superficie; se usa el decodificador normal (más lento).")
    try:
        return _exportar(pipeline, _vae_de_reserva(pipeline), latentes, resolucion, trozos)
    finally:
        _guardar_reserva()


def generar_forma(imagen: Image.Image, pasos: int = 5, resolucion: int = 380,
                  semilla: int = 0, dispositivo: str = "auto", intentos: int = 6,
                  modelo: str = "mini"):
    """Como _generar_forma; si el modelo estándar no cabe en la GPU, repite con el mini."""
    try:
        return _generar_forma(imagen, pasos, resolucion, semilla, dispositivo, intentos, modelo)
    except torch.OutOfMemoryError:
        if modelo == "mini":
            raise
        print("El modelo estándar no cabe en la memoria de la GPU; se usa el mini.")
        descargar_forma()
        return _generar_forma(imagen, pasos, resolucion, semilla, dispositivo, intentos, "mini")


def _generar_forma(imagen, pasos, resolucion, semilla, dispositivo, intentos, modelo):
    """Devuelve la malla (trimesh) generada a partir de una imagen sin fondo.

    Con dibujos 2D el modelo a veces decide que el objeto es un recorte plano
    (una lámina), según la semilla. Por eso se prueban hasta `intentos`
    semillas midiendo el grosor con una decodificación rápida de baja
    resolución, y solo la elegida se decodifica con calidad.
    """
    pipeline = cargar_pipeline(dispositivo, modelo)
    # En GPU, trozos grandes = menos llamadas a la red, pero más VRAM.
    trozos = 20000
    if str(pipeline.device).startswith("cuda"):
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        trozos = 200000 if vram_gb >= 12 else 60000 if vram_gb >= 8 else 20000

    mejor, mejor_grosor = None, -1.0
    try:
        for intento in range(max(intentos, 1)):
            s = semilla + intento
            latentes = _latentes(pipeline, imagen, pasos, s, trozos)
            g = _grosor_rapido(pipeline, latentes, trozos) if intentos > 1 else 1.0
            if intentos > 1:
                print(f"Semilla {s}: grosor {g:.0%}" + ("" if g >= GROSOR_MINIMO else " (plano)"))
            if g > mejor_grosor:
                mejor, mejor_grosor = latentes, g
            if g >= GROSOR_MINIMO:
                break
    finally:
        _guardar_reserva()
    if mejor_grosor < GROSOR_MINIMO:
        print(f"Aviso: en {intentos} intentos el modelo salió plano (grosor {mejor_grosor:.0%}). "
              "Prueba con más --intentos u otra --semilla; una imagen con sombras o en vista 3/4 "
              "da más volumen.")
    malla = _decodificar(pipeline, mejor, resolucion, trozos)
    if malla is None:
        raise RuntimeError("El modelo no generó ninguna superficie. Prueba otra imagen "
                           "(un objeto centrado sobre fondo liso) u otra --semilla.")
    print(f"Grosor del modelo: {grosor(malla):.0%}")
    return malla


def ruta_impresion(salida: str) -> str:
    return os.path.splitext(salida)[0] + "_impresion.stl"


def ruta_color(salida: str, extension: str) -> str:
    return os.path.splitext(salida)[0] + "_impresion_color" + extension


def generar(imagen: Image.Image, salida: str, pasos: int = 5, resolucion: int = 380,
            semilla: int = 0, dispositivo: str = "auto", textura: bool = True,
            fondo: str = "auto", ajustar: bool = True, intentos: int = 6,
            modelo_forma: str = "mini", superres: bool = True,
            acabado: str | None = None, colores: int = 1,
            altura_mm: float = 60, escala_acabado: float | None = None,
            profundidad_acabado: float | None = None, segmentacion: dict | None = None,
            **opciones_textura) -> str:
    """Genera el modelo 3D de `imagen` y lo guarda en `salida` (.glb, .obj, .stl).

    Con `acabado` (un nombre de acabados.ACABADOS o "ninguno") también guarda
    *_impresion.stl: en mm, Z arriba, apoyado en la cama y con el relieve elegido.
    Si además hay textura y `colores` > 1, guarda *_impresion_color.3mf con cada
    triángulo pintado con uno de `colores` filamentos (y una vista previa .glb).

    Con `textura=True` también lo pinta (necesita GPU); `opciones_textura` se
    pasan a `textura.texturizar` (max_caras, resolucion_textura...).
    Con `segmentacion` (dict con modo, partes, volumen, manual y union, ver
    segmentar.py) también divide la pieza de impresión: *_partes.zip (un STL por
    parte), *_partes.3mf (todas colocadas, pintadas si hay colores) y una vista
    previa *_partes.glb.
    La imagen ya preparada se guarda junto a la salida como *_entrada.png.
    """
    imagen, notas = preparar(imagen, fondo, ajustar, superres)
    for nota in notas:
        print(nota)
    imagen.save(os.path.splitext(salida)[0] + "_entrada.png")
    if textura:
        from textura import comprobar_dependencias
        comprobar_dependencias()  # falla antes de gastar tiempo en la forma

    t = time.time()
    malla = generar_forma(imagen, pasos, resolucion, semilla, dispositivo, intentos, modelo_forma)
    print(f"Forma: {time.time() - t:.1f} s ({len(malla.faces)} caras)")

    # Partes desprendidas (la bolita de una antena) se reconectan y los restos
    # sueltos se quitan; antes del limpiador de Hunyuan, que borraría las primeras.
    import limpieza
    avisos_trozos = []
    malla = limpieza.reparar(malla, avisos_trozos)
    for aviso in avisos_trozos:
        print(aviso)
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
        # Para poder pasar después el color de la textura a las partes (dividir.py).
        np.savetxt(os.path.splitext(ruta_impresion(salida))[0] + ".txt",
                   impresion.metadata["transformacion"])
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
    if acabado is not None and segmentacion and segmentacion.get("modo") not in (None, "", "no"):
        dividir(impresion, salida, segmentacion)
    malla.export(salida)
    return salida


def ruta_partes(salida: str, extension: str) -> str:
    return os.path.splitext(salida)[0] + "_partes" + extension


def dividir(impresion, salida: str, opciones: dict):
    """Divide la pieza de impresión en partes con uniones y guarda los archivos."""
    import segmentar

    t = time.time()
    cortes = segmentar.planificar(impresion, opciones["modo"], opciones.get("partes", 2),
                                  opciones.get("volumen", (220, 220, 250)),
                                  opciones.get("manual", ""))
    if not cortes:
        print("Segmentación: la pieza ya cabe entera; no hace falta dividirla.")
        return
    res = segmentar.segmentar(impresion, cortes, opciones.get("union"))
    for aviso in res.avisos:
        print(aviso)
    etiquetas, paleta = None, impresion.metadata.get("paleta")
    if paleta is not None:
        etiquetas = segmentar.colores_de_partes(res, impresion, impresion.metadata["etiquetas_color"])
    segmentar.exportar_zip(ruta_partes(salida, ".zip"), res)
    segmentar.exportar_3mf(ruta_partes(salida, ".3mf"), res, etiquetas, paleta)
    segmentar.vista_explosionada(res.partes, ruta_partes(salida, ".glb"))
    tam = ", ".join("x".join(f"{d:.0f}" for d in p.extents) for p in res.partes)
    print(f"Segmentación: {len(res.partes)} partes ({tam} mm)"
          + (f" + {len(res.pasadores)} pasadores" if res.pasadores else "")
          + f" -> {ruta_partes(salida, '.zip')} ({time.time() - t:.1f} s)")


def main():
    p = argparse.ArgumentParser(description="Imagen -> modelo 3D con Hunyuan3D-2")
    p.add_argument("imagen", help="Imagen de entrada (png, jpg, webp)")
    p.add_argument("-o", "--salida", default="modelo.glb", help="Archivo de salida (.glb, .obj, .stl)")
    p.add_argument("--pasos", type=int, default=5, help="Pasos de difusión de la forma (el modelo turbo usa 5)")
    p.add_argument("--resolucion", type=int, default=380,
                   help="Resolución de la malla (380 recomendado; hasta 512 para más detalle); "
                        "en CPU usa 128-192 para que no tarde tanto")
    p.add_argument("--modelo", default="mini", choices=list(modelos.MODELOS_FORMA),
                   help="mini: rápido; estandar: Hunyuan3D-2 de 1.1 B, más detalle (+5 GB de descarga)")
    p.add_argument("--sin-superres", action="store_true",
                   help="No ampliar con IA las imágenes pequeñas ni las vistas de la textura")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--intentos", type=int, default=6,
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
    import segmentar
    d = p.add_argument_group("dividir en partes (genera <salida>_partes.zip/.3mf; necesita --acabado)")
    d.add_argument("--segmentar", choices=list(segmentar.MODOS),
                   help="; ".join(f"{k}: {v.lower()}" for k, v in segmentar.MODOS.items()))
    d.add_argument("--partes", type=int, default=2, help="Partes para horizontal/vertical")
    d.add_argument("--volumen", default="220x220x250",
                   help="Volumen de impresión AnchoxFondoxAlto en mm (modo cama)")
    d.add_argument("--cortes", default="", help='Cortes manuales, p. ej. "z=30, x=0"')
    d.add_argument("--union", default="espiga", choices=list(segmentar.UNIONES),
                   help="; ".join(f"{k}: {v.lower()}" for k, v in segmentar.UNIONES.items()))
    d.add_argument("--diametro-union", type=float, default=5.0, help="Diámetro de espiga/pasador (mm)")
    d.add_argument("--largo-union", type=float, default=8.0, help="Largo de espiga/pasador (mm)")
    d.add_argument("--holgura", type=float, default=0.2, help="Juego para que encaje (mm)")
    d.add_argument("--iman", default="6x3", help="Imán DiámetroxAlto en mm")
    args = p.parse_args()

    segmentacion = None
    if args.segmentar:
        if args.acabado is None:
            args.acabado = "ninguno"
        im_d, im_h = (float(v) for v in args.iman.lower().split("x"))
        segmentacion = {
            "modo": args.segmentar, "partes": args.partes, "manual": args.cortes,
            "volumen": tuple(float(v) for v in args.volumen.lower().split("x")),
            "union": segmentar.OpcionesUnion(args.union, args.diametro_union, args.largo_union,
                                             args.holgura, im_d, im_h),
        }

    inicio = time.time()
    generar(Image.open(args.imagen), args.salida, args.pasos, args.resolucion,
            args.semilla, args.dispositivo, textura=not args.sin_textura, fondo=args.fondo,
            ajustar=not args.sin_ajustar, intentos=args.intentos, modelo_forma=args.modelo,
            superres=not args.sin_superres, nitida=not args.sin_superres, acabado=args.acabado,
            colores=args.colores, altura_mm=args.altura_mm,
            escala_acabado=args.escala_acabado, profundidad_acabado=args.profundidad_acabado,
            segmentacion=segmentacion,
            max_caras=args.max_caras, resolucion_textura=args.resolucion_textura,
            vram_baja=args.vram_baja)
    print(f"Listo: {args.salida} ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    main()
