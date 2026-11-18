# Pixal3D / TRELLIS.2 en ComfyUI con AMD (RX 9060 XT, ROCm 7.2 en Windows)

Estado: geometría y textura correctas con los parches de abajo (probado con el hacha y el perro robot).

## Fallo principal encontrado
En ROCm/Windows (gfx1200, torch 2.9.1+rocm7.2.1), cualquier multiplicación de matrices
(`F.linear` / `torch.matmul`) con **más de 524288 filas (2^19)** devuelve basura a partir de esa
fila, sin error. En el decodificador de textura (y de forma) de TRELLIS.2 los niveles finos tienen
~1-7 millones de vóxeles, así que la textura salía magenta / con ruido en franjas.
`prueba_linear_grande.py` lo reproduce en segundos.

## Archivos para `ComfyUI\custom_nodes\`
- `amd_parche_rocm.py` (**necesario**): parte Linear/matmul en trozos de hasta 2^18 filas.
- `amd_volcado_textura.py` (solo depuración): guarda las entradas del horneado en
  `output\amd_horneado.npz`; con `AMD_DIAG_TEXTURA=1` compara el decodificador de textura
  capa por capa GPU vs CPU (añade 30-60 min). Se puede borrar cuando todo funcione.

## Otros cambios en el núcleo de ComfyUI (hechos a mano, con copia `.orig`)
- `comfy_extras\mesh3d\uv_unwrap\parameterize.py`: solve fp64 del LSCM en CPU
  (hipBLAS fp64 por lotes daba HIPBLAS_STATUS_ALLOC_FAILED).
- `comfy\quant_ops.py`: `COMFY_SIN_HIP=1` desactiva los kernels hip de comfy_kitchen (solo pruebas).

## Arranque
    python_embeded\python.exe ComfyUI\main.py --fp32-vae

Workflow: plantilla `3d_pixal3d_trellis2_image_to_model`, booleano en `false` (Pixal3D),
cargador de arriba `pixal3d_bf16`, cargador de abajo `trellis_2_bf16`.

## Scripts de prueba
- `prueba_linear_grande.py`: fallo de 2^19 filas, sin y con parche.
- `prueba_textura_amd.py`: etapas de textura (UV, rasterizado, muestreo, JFA, AO) GPU vs CPU.
- `prueba_horneado_real.py`: analiza `amd_horneado.npz` (colores de vóxeles, UV, horneado).
