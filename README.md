# 3DIA: imagen a 3D

Convierte una imagen (por ejemplo, la foto de una figura) en un modelo 3D `.glb`
con forma y color (textura), usando el modelo abierto [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).

Incluye:

- `generar3d.py`: línea de comandos, imagen -> modelo 3D.
- `app.py`: app web para subir la imagen, ver el modelo y descargarlo.
- `textura.py`: pinta la malla con Hunyuan3D-Paint (turbo).
- `relleno_uv.py`: relleno rápido de los huecos de la textura.
- `rasterizador.py`: rasterizador en PyTorch (sustituye al módulo CUDA compilado).
- `modelos.py`: descarga de pesos con reintentos.
- `preparar_imagen.py`: analiza la imagen y quita el fondo antes de generar.

## Requisitos

- Python 3.10 a 3.12 y Git.
- GPU NVIDIA con 6 GB de VRAM o más (con menos de 16 GB, los modelos de
  textura se mueven a la GPU solo mientras se usan).
  En CPU solo se puede generar la forma, y tarda decenas de minutos.
- Unos 4 GB de disco para los pesos de la forma, que se descargan solos la primera vez.
- Para la textura, unos 14 GB más de pesos y 16 GB de RAM si la GPU tiene poca VRAM.

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

### Textura

No hace falta compilar nada: el rasterizador de Hunyuan3D-2 (CUDA) está
reimplementado en PyTorch puro en `rasterizador.py`. Si aun así tienes compilado
`custom_rasterizer`, se usa ese.

Con poca VRAM (por ejemplo 6 GB) los modelos de pintura se cargan en la RAM y se
suben a la GPU solo mientras se usan, así que conviene tener 16 GB de RAM.

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
- `--max-caras 20000`: menos caras al pintar, más rápido.
- `--vram-baja`: fuerza a mover modelos a la CPU si te quedas sin memoria.

## Velocidad

- Forma: decodificador FlashVDM (solo evalúa cerca de la superficie) y trozos
  grandes en GPU.
- Textura: modelo de pintura turbo, la mitad de pasos al quitar luces y sombras, malla reducida antes
  de desplegar UVs y relleno de huecos vectorizado (~5x más rápido que el original).
- Los modelos se cargan una sola vez: en la app web la primera generación es la
  más lenta.

### Preparación de la imagen

Antes de generar hay dos pasos independientes:

- **Quitar el fondo** (`--fondo`): por defecto respeta la transparencia si la
  imagen la trae y, si no, usa IA (BiRefNet vía `rembg`, ~220 MB la primera vez,
  unos segundos por imagen). Distingue partes blancas del objeto (una cara) de un
  fondo blanco. `--fondo blanco` usa relleno por color sin IA y `--fondo mantener`
  no lo toca.
- **Ajustar la imagen** (se desactiva con `--sin-ajustar`): corrige la rotación de
  fotos de móvil, reduce las muy grandes, limpia manchas, recorta y centra, y
  avisa si el objeto está cortado o es muy pequeño.

Lo que recibe el modelo se guarda como `<salida>_entrada.png`. Consejos: un solo
objeto, completo, bien iluminado y lo más grande posible en la imagen.

## Licencia y créditos

El código de este repositorio es propio. El modelo y la librería `hy3dgen` son de Tencent
y se rigen por la *Tencent Hunyuan 3D 2.0 Community License*, que entre otras cosas:

- no cubre el uso en la Unión Europea, Reino Unido ni Corea del Sur;
- prohíbe usar el modelo o sus resultados para mejorar otros modelos de IA.

Este proyecto no está afiliado a Tencent.
