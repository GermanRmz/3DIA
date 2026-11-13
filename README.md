# 3DIA: imagen a 3D

Convierte una imagen (por ejemplo, la foto de una figura) en un modelo 3D `.glb`
con forma y color (textura), usando el modelo abierto [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).

Incluye:

- `generar3d.py`: línea de comandos, imagen -> modelo 3D.
- `app.py`: app web para subir la imagen, ver el modelo y descargarlo.
- `textura.py`: pinta la malla con Hunyuan3D-Paint (turbo).
- `relleno_uv.py`: relleno rápido de los huecos de la textura.

## Requisitos

- Python 3.10 a 3.12 y Git.
- GPU NVIDIA. Solo forma: 6 GB de VRAM. Forma + textura: 12 GB o más
  (con menos de 16 GB los modelos se mueven a CPU cuando no se usan).
  En CPU solo se puede generar la forma, y tarda decenas de minutos.
- Unos 20 GB de disco para los pesos, que se descargan solos la primera vez.
- Para la textura: [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)
  y, en Windows, Visual Studio Build Tools con "Desarrollo para el escritorio con C++".

## Instalación

```bash
git clone https://github.com/GermanRmz/3DIA
cd 3DIA
python -m venv venv
```

Activa el entorno (`venv\Scripts\activate` en Windows, `source venv/bin/activate` en Linux/Mac)
e instala PyTorch con CUDA y luego el resto:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Textura (una sola vez)

El texturizado usa un rasterizador CUDA que hay que compilar.

1. Mira qué CUDA usa tu PyTorch: `python -c "import torch; print(torch.version.cuda)"`
   (por ejemplo `12.4`) e instala el **CUDA Toolkit** de esa misma versión.
2. En Windows abre "x64 Native Tools Command Prompt for VS 2022", entra en la
   carpeta del proyecto, activa el entorno (`venv\Scripts\activate`) y ejecuta
   (cambia `v12.4` por tu versión):

```bat
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_HOME%\bin;%PATH%
set DISTUTILS_USE_SDK=1
pip install --no-build-isolation "git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2#subdirectory=hy3dgen/texgen/custom_rasterizer"
```

En Linux basta con `export CUDA_HOME=/usr/local/cuda` y el `pip install` de arriba.
Si sale "unsupported Microsoft Visual Studio version", añade antes
`set NVCC_APPEND_FLAGS=-allow-unsupported-compiler`.

Comprueba que quedó bien con `python -c "import custom_rasterizer"`.
Si no lo instalas, puedes seguir generando solo la forma con `--sin-textura`.

## Uso

App web (abre http://localhost:7860):

```bash
python app.py
```

Línea de comandos:

```bash
python generar3d.py figura.png -o figura.glb
```

Opciones útiles:

- `--sin-textura`: solo la forma (funciona también en CPU con `--dispositivo cpu`).
- `--resolucion 192`: forma más rápida, menos detalle.
- `--semilla 1`: otra variante.
- `--resolucion-textura 1024`: textura más rápida y con menos VRAM.
- `--pasos-textura 30`: más calidad de color (por defecto 15).
- `--max-caras 20000`: menos caras al pintar, más rápido.
- `--vram-baja`: fuerza a mover modelos a la CPU si te quedas sin memoria.

## Velocidad

- Forma: decodificador FlashVDM (solo evalúa cerca de la superficie) y trozos
  grandes en GPU.
- Textura: modelo de pintura turbo, menos pasos de difusión, malla reducida antes
  de desplegar UVs y relleno de huecos vectorizado (~5x más rápido que el original).
- Los modelos se cargan una sola vez: en la app web la primera generación es la
  más lenta.

Consejos para la imagen: un solo objeto, centrado, fondo blanco o transparente.

## Licencia y créditos

El código de este repositorio es propio. El modelo y la librería `hy3dgen` son de Tencent
y se rigen por la *Tencent Hunyuan 3D 2.0 Community License*, que entre otras cosas:

- no cubre el uso en la Unión Europea, Reino Unido ni Corea del Sur;
- prohíbe usar el modelo o sus resultados para mejorar otros modelos de IA.

Este proyecto no está afiliado a Tencent.
