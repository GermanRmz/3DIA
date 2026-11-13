"""Texturizado (color) de la malla con Hunyuan3D-Paint.

Flujo: limpiar y reducir la malla -> desplegar UVs -> quitar luces/sombras de la
foto -> generar 6 vistas con difusión -> proyectarlas sobre la textura -> rellenar
huecos.

Optimizaciones respecto al ejemplo oficial:
- Se usa el modelo "turbo" de pintura (destilado, sin guía doble).
- Quitar luces/sombras con 25 pasos en vez de 50 (las 6 vistas usan 10 pasos
  fijos en el modelo turbo).
- La malla se reduce antes de desplegar UVs (xatlas es lento con muchas caras).
- El relleno de huecos de la textura está vectorizado (ver relleno_uv.py).
- En GPUs con poca VRAM los modelos se mueven a CPU cuando no se usan.

Necesita GPU NVIDIA. Si el módulo compilado `custom_rasterizer` no está, se usa
el rasterizador en PyTorch puro de rasterizador.py (mismo resultado).
"""
import contextlib
import os
import sys

import torch

import modelos
from relleno_uv import rellenar_textura

_pintor = None
_config_pintor = None


class _PasosFijos:
    """Envuelve un pipeline de diffusers para forzar `num_inference_steps`.

    hy3dgen tiene los pasos escritos a mano (50); así se pueden cambiar
    sin modificar la librería. El resto de atributos pasan tal cual.
    """

    def __init__(self, pipeline, pasos):
        self._pipeline = pipeline
        self.pasos = pasos

    def __call__(self, *args, **kwargs):
        kwargs["num_inference_steps"] = self.pasos
        return self._pipeline(*args, **kwargs)

    def __getattr__(self, nombre):
        return getattr(self._pipeline, nombre)


def _uv_inpaint_rapido(self, texture, mask):
    """Sustituto vectorizado de MeshRender.uv_inpaint."""
    if isinstance(texture, torch.Tensor):
        texture = texture.cpu().numpy()
    vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()
    return rellenar_textura(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)


@contextlib.contextmanager
def _carga_ligera():
    """Reduce el pico de RAM al cargar los modelos de pintura (de ~17 GB a ~6 GB).

    La UNet de pintura se construye en float32 (7.3 GB) y además lee su archivo
    .bin entero a la RAM (otros 7.3 GB) antes de pasar a float16. Aquí se
    construye directamente en float16 y el .bin se lee mapeado desde el disco.
    """
    carga_original = torch.load
    tipo_original = torch.get_default_dtype()

    def carga_mapeada(*args, **kwargs):
        kwargs.setdefault("mmap", True)
        return carga_original(*args, **kwargs)

    torch.load = carga_mapeada
    torch.set_default_dtype(torch.float16)
    try:
        yield
    finally:
        torch.load = carga_original
        torch.set_default_dtype(tipo_original)


def _vram_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def comprobar_dependencias():
    """Lanza un error claro si falta algo para texturizar."""
    if not torch.cuda.is_available():
        raise RuntimeError("El texturizado necesita una GPU NVIDIA con CUDA.")
    try:
        import custom_rasterizer  # noqa: F401
    except ImportError:
        import rasterizador
        sys.modules["custom_rasterizer"] = rasterizador


def cargar_pintor(resolucion_textura: int = 2048, pasos_luz: int = 25,
                  vram_baja: bool | None = None):
    """Carga el modelo de pintura una sola vez (la primera vez descarga ~15 GB)."""
    global _pintor, _config_pintor
    comprobar_dependencias()
    if vram_baja is None:
        vram_baja = _vram_gb() < 16

    if _pintor is None or _config_pintor["vram_baja"] != vram_baja:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline, Hunyuan3DTexGenConfig
        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        MeshRender.uv_inpaint = _uv_inpaint_rapido
        _pintor = None
        torch.cuda.empty_cache()
        carpeta = modelos.carpeta_pintura()
        config = Hunyuan3DTexGenConfig(os.path.join(carpeta, modelos.LUZ),
                                       os.path.join(carpeta, modelos.PINTURA), modelos.PINTURA)
        # hy3dgen sube ambos modelos (~7 GB) a la GPU al cargarlos; con poca VRAM
        # se cargan en CPU y el offload sube cada uno solo mientras se usa.
        config.device = "cpu" if vram_baja else "cuda"
        config.render_size = config.texture_size = resolucion_textura
        with _carga_ligera():
            _pintor = Hunyuan3DPaintPipeline(config)
        if vram_baja:
            _pintor.enable_model_cpu_offload(device="cuda")
        luz = _pintor.models["delight_model"]
        luz.pipeline = _PasosFijos(luz.pipeline, pasos_luz)
        _config_pintor = {"vram_baja": vram_baja, "resolucion": resolucion_textura}

    _pintor.models["delight_model"].pipeline.pasos = pasos_luz
    if _config_pintor["resolucion"] != resolucion_textura:
        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        _pintor.config.render_size = resolucion_textura
        _pintor.config.texture_size = resolucion_textura
        _pintor.render = MeshRender(default_resolution=resolucion_textura,
                                    texture_size=resolucion_textura)
        _config_pintor["resolucion"] = resolucion_textura
    return _pintor


def preparar_malla(malla, max_caras: int | None = None):
    """Quita trozos sueltos y caras degeneradas; opcionalmente reduce caras."""
    from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

    malla = FloaterRemover()(malla)
    malla = DegenerateFaceRemover()(malla)
    if max_caras and len(malla.faces) > max_caras:
        malla = FaceReducer()(malla, max_facenum=max_caras)
    return malla


def texturizar(malla, imagen, max_caras: int = 40000, resolucion_textura: int = 2048,
               pasos_luz: int = 25, vram_baja: bool | None = None):
    """Devuelve la malla con textura de color. `imagen` debe venir sin fondo (RGBA)."""
    pintor = cargar_pintor(resolucion_textura, pasos_luz, vram_baja)
    malla = preparar_malla(malla, max_caras)
    try:
        resultado = pintor(malla.copy(), image=imagen)
    except torch.OutOfMemoryError:
        # Último recurso: subir a la GPU capa por capa (más lento, cabe en ~4 GB).
        print("Sin memoria en la GPU; se reintenta moviendo los modelos capa por capa (más lento).")
        torch.cuda.empty_cache()
        for nombre in ("delight_model", "multiview_model"):
            pintor.models[nombre].pipeline.enable_sequential_cpu_offload(device="cuda")
        resultado = pintor(malla.copy(), image=imagen)
    torch.cuda.empty_cache()
    return resultado
