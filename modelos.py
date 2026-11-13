"""Descarga de los pesos desde Hugging Face.

hy3dgen descarga carpetas enteras (con cada modelo repetido en dos formatos)
y, si la conexión falla a mitad, sigue con lo que haya en caché y acaba en un
"Model path not found". Aquí se bajan solo los archivos que se usan, con
reintentos, y se comprueba que estén antes de cargar nada.
"""
import os
import time

# Conexiones lentas: más margen antes de dar una descarga por perdida.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

REPO_FORMA = "tencent/Hunyuan3D-2mini"
DIT_FORMA = "hunyuan3d-dit-v2-mini-turbo"
VAE_RAPIDO = "hunyuan3d-vae-v2-mini-turbo"

REPO_PINTURA = "tencent/Hunyuan3D-2"
LUZ = "hunyuan3d-delight-v2-0"
PINTURA = "hunyuan3d-paint-v2-0-turbo"


def descargar(repo: str, patrones: list[str], ignorar: list[str], necesarios: list[str],
              intentos: int = 5) -> str:
    """Descarga `patrones` de `repo` y devuelve la carpeta local."""
    from huggingface_hub import snapshot_download

    for intento in range(1, intentos + 1):
        try:
            carpeta = snapshot_download(repo_id=repo, allow_patterns=patrones,
                                        ignore_patterns=ignorar)
            faltan = [f for f in necesarios if not os.path.exists(os.path.join(carpeta, f))]
            if not faltan:
                return carpeta
            error = f"faltan {faltan}"
        except Exception as e:  # cortes de red, timeouts...
            error = e
        print(f"Descarga de {repo} incompleta ({error}). Reintento {intento}/{intentos}...")
        time.sleep(5 * intento)
    raise RuntimeError(f"No se pudo descargar {repo}. Revisa tu conexión y vuelve a ejecutar; "
                       "lo ya descargado no se pierde.")


def carpeta_forma(con_vae_rapido: bool) -> str:
    """Modelo de forma (~3.8 GB) y, en GPU, el decodificador rápido (~0.2 GB)."""
    sub = [DIT_FORMA] + ([VAE_RAPIDO] if con_vae_rapido else [])
    archivos = [f"{s}/{n}" for s in sub for n in ("config.yaml", "model.fp16.safetensors")]
    return descargar(REPO_FORMA, archivos, [], archivos)


def carpeta_pintura() -> str:
    """Modelos de textura (~15 GB). La UNet de pintura se carga del .bin, no del .safetensors."""
    return descargar(
        REPO_PINTURA,
        [f"{LUZ}/*", f"{PINTURA}/*"],
        [f"{PINTURA}/unet/diffusion_pytorch_model.safetensors", "*/README.md", "*/.gitattributes"],
        [f"{LUZ}/unet/diffusion_pytorch_model.safetensors",
         f"{PINTURA}/unet/diffusion_pytorch_model.bin",
         f"{PINTURA}/unet/modules.py"],
    )
