# 3DIA: imagen a 3D

Convierte una imagen (por ejemplo, la foto de una figura) en un modelo 3D `.glb`
usando el modelo abierto [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).

Incluye:

- `generar3d.py`: línea de comandos, imagen -> modelo 3D.
- `app.py`: app web para subir la imagen, ver el modelo y descargarlo.

Por ahora genera solo la **forma** (sin color/textura).

## Requisitos

- Python 3.10 a 3.12 y Git.
- GPU NVIDIA con 6 GB de VRAM o más (recomendado). En CPU funciona, pero
  una generación tarda decenas de minutos.
- Unos 10 GB de disco para los pesos, que se descargan solos la primera vez.

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

## Uso

App web (abre http://localhost:7860):

```bash
python app.py
```

Línea de comandos:

```bash
python generar3d.py figura.png -o figura.glb
```

Opciones útiles: `--resolucion 192` (más rápido, menos detalle), `--semilla 1`
(otra variante), `--dispositivo cpu`.

Consejos para la imagen: un solo objeto, centrado, fondo blanco o transparente.

## Licencia y créditos

El código de este repositorio es propio. El modelo y la librería `hy3dgen` son de Tencent
y se rigen por la *Tencent Hunyuan 3D 2.0 Community License*, que entre otras cosas:

- no cubre el uso en la Unión Europea, Reino Unido ni Corea del Sur;
- prohíbe usar el modelo o sus resultados para mejorar otros modelos de IA.

Este proyecto no está afiliado a Tencent.
