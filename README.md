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
- `acabados.py`: relieves de superficie para impresión 3D (tejido, moleteado...).
- `multicolor.py`: 3MF pintado por filamentos para impresoras multicolor.
- `segmentar.py`: divide la pieza en partes con uniones (espigas, pasadores, imanes).

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

## Impresión 3D y acabados

Con `--acabado` (o la sección "Impresión 3D" de la app) se genera además
`<salida>_impresion.stl`: en milímetros, con Z hacia arriba, apoyado en la cama y
cerrado (estanco), listo para el laminador. El acabado es **relieve real** en la
geometría, no color:

| Acabado | Aspecto | Patrón / profundidad por defecto |
|---|---|---|
| `tejido` | punto de lana (columnas de V) | 2.5 mm / 0.5 mm |
| `moleteado` | diamantes en relieve | 2 mm / 0.4 mm |
| `estrias` | estrías verticales | 2 mm / 0.4 mm |
| `anillos` | anillos horizontales | 1.5 mm / 0.3 mm |
| `martillado` | metal martillado | 3 mm / 0.4 mm |
| `piedra` | celdas tipo piedra o escamas | 4 mm / 0.4 mm |
| `madera` | vetas de madera | 2 mm / 0.3 mm |
| `ninguno` | liso | — |

```bash
python generar3d.py gato.jpg -o gato.glb --acabado tejido --altura-mm 60
```

Ajusta con `--escala-acabado` y `--profundidad-acabado` (mm). Para que el relieve
se note, el patrón debe ser varias veces mayor que el diámetro de la boquilla
(con 0.4 mm, patrones de 2 mm o más) y la profundidad de al menos 0.3 mm.

### Impresoras multicolor

Con textura y `--colores N` (o el control "Colores de filamento" de la app) se
genera además `<salida>_impresion_color.3mf`: la textura se reduce a N colores
como máximo (los casi iguales se unen) y cada triángulo queda pintado con su
filamento, en el formato de pintura de Bambu Studio / OrcaSlicer (`paint_color`)
y de PrusaSlicer (`mmu_segmentation`). Al abrirlo, asigna a cada filamento el
color que indica el programa al terminar (el filamento 1 es el más abundante).

```bash
python generar3d.py gato.jpg -o gato.glb --acabado tejido --colores 4
```

### Dividir en partes

Para piezas grandes, la pieza de impresión se corta en partes con uniones para
pegarlas o encajarlas (sección "Dividir en partes" de la app, o `--segmentar`):

| Modo | Qué hace |
|---|---|
| `horizontal` | `--partes N` capas de igual altura |
| `vertical_x`, `vertical_y` | `--partes N` rodajas verticales |
| `cama` | los cortes justos para que cada parte quepa en `--volumen 220x220x250` |
| `manual` | `--cortes "z=30, x=0"` (mm) |

| Unión | Qué hace |
|---|---|
| `espiga` | cilindro macho en una parte y agujero con holgura en la otra |
| `pasadores` | agujeros en ambas partes y pasadores sueltos para imprimir |
| `imanes` | alojamientos para imanes redondos (`--iman 6x3`) |
| `ninguna` | corte plano para pegar |

Las uniones se colocan solas dentro de cada sección (dos si caben, para que no
giren; más pequeñas o ninguna si no caben). Se ajustan con `--diametro-union`,
`--largo-union` y `--holgura` (0.2 mm por defecto; súbela si no entran).
Salen `<salida>_partes.zip` (un STL por parte y los pasadores), `<salida>_partes.3mf`
(todas colocadas en la cama y, con `--colores`, pintadas) y una vista separada.

```bash
python generar3d.py dragon.jpg -o dragon.glb --altura-mm 300 --segmentar cama --volumen 220x220x250 --union espiga
```

### Modelos planos

Con dibujos 2D de frente, el modelo a veces decide que el objeto es un recorte
plano (una lámina de grosor casi cero); depende de la semilla. Por eso se prueban
hasta 6 semillas (`--intentos`): cada una se decodifica a muy baja resolución
para medir el grosor en unos segundos, y solo la primera con volumen se decodifica
con calidad. La imagen se codifica una sola vez para todos los intentos.

`python diagnostico.py imagen.png` genera y mide la forma de varias maneras en tu
GPU, por si hay que investigar más.

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
