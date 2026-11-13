"""Texturizado (color) de la malla con Hunyuan3D-Paint.

Flujo: limpiar y reducir la malla -> desplegar UVs -> quitar luces/sombras de la
foto -> generar 6 vistas con difusión -> proyectarlas sobre la textura -> rellenar
huecos.

Optimizaciones respecto al ejemplo oficial:
- Se usa el modelo "turbo" de pintura (destilado, sin guía doble).
- Menos pasos de difusión por defecto (configurables).
- La malla se reduce antes de desplegar UVs (xatlas es lento con muchas caras).
- El relleno de huecos de la textura está vectorizado (ver relleno_uv.py).
- En GPUs con poca VRAM los modelos se mueven a CPU cuando no se usan.

Necesita GPU NVIDIA y el módulo compilado `custom_rasterizer` (ver README).
"""
import torch

import modelos
from relleno_uv import rellenar_textura

_pintor = None
_config_pintor = None


class _PasosFijos:
    """Envuelve un pipeline de diffusers para forzar `num_inference_steps`.

    hy3dgen tiene los pasos escritos a mano (50 y 30); así se pueden cambiar
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


def _vram_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def comprobar_dependencias():
    """Lanza un error claro si falta algo para texturizar."""
    if not torch.cuda.is_available():
        raise RuntimeError("El texturizado necesita una GPU NVIDIA con CUDA.")
    try:
        import custom_rasterizer  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Falta el módulo 'custom_rasterizer' (se compila una vez). "
            "Mira la sección 'Textura' del README.") from e


def cargar_pintor(resolucion_textura: int = 2048, pasos_vistas: int = 15,
                  pasos_luz: int = 25, vram_baja: bool | None = None):
    """Carga el modelo de pintura una sola vez (la primera vez descarga ~15 GB)."""
    global _pintor, _config_pintor
    comprobar_dependencias()
    if vram_baja is None:
        vram_baja = _vram_gb() < 16

    if _pintor is None or _config_pintor["vram_baja"] != vram_baja:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        MeshRender.uv_inpaint = _uv_inpaint_rapido
        _pintor = None
        torch.cuda.empty_cache()
        _pintor = Hunyuan3DPaintPipeline.from_pretrained(modelos.carpeta_pintura(),
                                                         subfolder=modelos.PINTURA)
        if vram_baja:
            _pintor.enable_model_cpu_offload()
        for nombre in ("delight_model", "multiview_model"):
            modelo = _pintor.models[nombre]
            modelo.pipeline = _PasosFijos(modelo.pipeline, 0)
        _config_pintor = {"vram_baja": vram_baja, "resolucion": None}

    _pintor.models["delight_model"].pipeline.pasos = pasos_luz
    _pintor.models["multiview_model"].pipeline.pasos = pasos_vistas
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
               pasos_vistas: int = 15, pasos_luz: int = 25, vram_baja: bool | None = None):
    """Devuelve la malla con textura de color. `imagen` debe venir sin fondo (RGBA)."""
    pintor = cargar_pintor(resolucion_textura, pasos_vistas, pasos_luz, vram_baja)
    malla = preparar_malla(malla, max_caras)
    resultado = pintor(malla, image=imagen)
    torch.cuda.empty_cache()
    return resultado
