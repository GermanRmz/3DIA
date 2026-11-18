# Reinstalar Pixal3D / TRELLIS.2 en la PC con AMD desde cero

Probado en: Windows 10, RX 9060 XT 16 GB (`gfx1200`), 32 GB de RAM, ROCm 7.2.1 de AMD (TheRock),
torch 2.9.1+rocm7.2.1, Python 3.12.10 y ComfyUI v0.45.0.
Resultado: geometría y textura correctas, unos 20–40 min por figura con la calidad máxima de la plantilla.

Todos los archivos citados están en este repositorio, rama `optimizacion-y-textura`, carpeta `amd/trellis2/`.
Para descargar cualquiera desde cmd:
```bat
curl -L -o ARCHIVO https://raw.githubusercontent.com/GermanRmz/3dia/optimizacion-y-textura/amd/trellis2/RUTA
```

---

## 0. Requisitos (una sola vez)
- El driver AMD Adrenalin más reciente.
- [Git para Windows](https://git-scm.com/download/win).
- **Python 3.12.10 de 64 bits** de python.org. Solo hace falta para compilar cumesh.
- **Visual Studio 2022 Build Tools** con la carga «Desarrollo para el escritorio con C++». Solo hace falta para compilar cumesh.
- Unos 80 GB libres. Los modelos bf16 ocupan unos 25 GB.
- **Exclusión de Windows Defender** para la carpeta de trabajo. Defender llegó a bloquear el `clang` de ROCm durante la compilación. En PowerShell como administrador:
  ```powershell
  Add-MpPreference -ExclusionPath "C:\AMD-AI"
  ```

## 1. Entorno base con el kit de la comunidad
Kit: [tsubasaprogame/Pixal3D-TRELLIS.2-AMD-ROCm-WINDOWS](https://github.com/tsubasaprogame/Pixal3D-TRELLIS.2-AMD-ROCm-WINDOWS). Probado en el commit `a4b7b16`.
```powershell
cd C:\
git clone https://github.com/tsubasaprogame/Pixal3D-TRELLIS.2-AMD-ROCm-WINDOWS
cd Pixal3D-TRELLIS.2-AMD-ROCm-WINDOWS
```
Abre `scripts\native\verify_pixal3d_amd.py` y cambia la línea 17 para que acepte tu tarjeta:
```python
SUPPORTED_ARCHITECTURES = {"gfx1201", "gfx1200"}
```
Instala el entorno: ROCm, torch y Python embebido en `C:\AMD-AI\Pixal3D-ROCm`.
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File installer/amd/Pixal3DNative.ps1 -RuntimeRoot C:\AMD-AI\Pixal3D-ROCm
```
La comprobación de `cumesh` falla con `0xC0000005` porque el kit trae cumesh compilado solo para `gfx1201`. Es normal: el paso 2 lo arregla.
**Después del paso 2, no vuelvas a ejecutar el instalador**, porque reinstalaría ese cumesh.

## 2. cumesh compilado para gfx1200

**Camino rápido:** la rueda ya compilada está en el Release
[amd-cumesh-gfx1200](https://github.com/GermanRmz/3DIA/releases/tag/amd-cumesh-gfx1200).
Descárgala desde esa página o con GitHub CLI, instálala y salta al paso 3:
```bat
gh release download amd-cumesh-gfx1200 --repo GermanRmz/3dia --dir C:\AMD-AI\respaldo
C:\AMD-AI\Pixal3D-ROCm\python_embeded\python.exe -m pip install --no-deps --force-reinstall C:\AMD-AI\respaldo\cumesh-1.0-cp312-cp312-win_amd64.whl
```
En el mismo Release están `paquetes_pip.txt` (versiones exactas de todos los paquetes), `versiones.txt`
y `comfyui_commit.txt`. Para comprobar el entorno, compara su `paquetes_pip.txt` con `python_embeded\python.exe -m pip freeze`.

Si no la tienes, compílala.

**a) Python para compilar.** En «Developer PowerShell for VS 2022»:
```powershell
py -3.12 -m venv C:\AMD-AI\compilar
C:\AMD-AI\compilar\Scripts\python.exe -m pip install --upgrade pip setuptools wheel ninja numpy
cd C:\Pixal3D-TRELLIS.2-AMD-ROCm-WINDOWS\installer\downloads\amd
C:\AMD-AI\compilar\Scripts\python.exe -m pip install rocm-7.2.1.tar.gz rocm_sdk_core-7.2.1-py3-none-win_amd64.whl rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl "torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl"
C:\AMD-AI\compilar\Scripts\rocm-sdk.exe init
```

**b) Código, en los commits probados:**
```powershell
cd C:\AMD-AI
git clone https://github.com/visualbruno/CuMesh
git -C CuMesh checkout d10e54c30ddd03d11472c1431693f985501c7966
git clone https://github.com/NVIDIA/cuBQL
git -C cuBQL checkout a7c1072119b34312e8b5dbf73a2e22628309acb7
```

**c) Cambios para AMD.** Descarga `cumesh/parche_cumesh_amd.py`, `cumesh/compilar_cumesh.bat`, `cumesh/prueba_cumesh.py` y `cumesh/prueba_bvh.py` en `C:\AMD-AI` y ejecuta:
```bat
C:\AMD-AI\compilar\Scripts\python.exe C:\AMD-AI\parche_cumesh_amd.py C:\AMD-AI\CuMesh C:\Pixal3D-TRELLIS.2-AMD-ROCm-WINDOWS
```
El script hace estos cambios:
- copia los archivos del kit;
- `ext.cpp` → `ext.cu`;
- en `clean_up.cu`, usa `rocprim::tuple` en vez de `cuda::std::tuple`;
- cambia `TORCH_CHECK_VALUE` por `TORCH_CHECK`;
- compila `bindings` y `binding` de xatlas como `.cu` con hipcc, porque `cl` da LNK2001 con `c10::ValueError`.

**d) Compilar e instalar.** En cmd, no en PowerShell:
```bat
C:\AMD-AI\compilar_cumesh.bat
```
El script carga `vcvars64`, `ROCM_HOME`, `GPU_ARCHS=gfx1200;gfx1201` y rutas cortas para `INCLUDE`, porque `hipcc` falla con `Program Files (x86)`. Después compila la rueda en `C:\AMD-AI\wheels` y la instala.

**e) Probar.** Las dos pruebas deben terminar en `TODO OK`:
```bat
cd /d C:\AMD-AI
C:\AMD-AI\Pixal3D-ROCm\python_embeded\python.exe prueba_cumesh.py
C:\AMD-AI\Pixal3D-ROCm\python_embeded\python.exe prueba_bvh.py
```

## 3. ComfyUI v0.45.0
```bat
cd /d C:\AMD-AI\Pixal3D-ROCm
git clone https://github.com/comfyanonymous/ComfyUI
git -C ComfyUI checkout v0.45.0
powershell -NoProfile -Command "Get-Content ComfyUI\requirements.txt | Select-String -NotMatch '^torch' | Set-Content req_comfy.txt"
python_embeded\python.exe -m pip install -r req_comfy.txt
```
**No instales `torch` de PyPI.** Reemplazaría el torch de AMD.

## 4. Parches de AMD para ComfyUI (imprescindible)
Descarga `parche_comfyui_core_amd.py` y `amd_parche_rocm.py` en `C:\AMD-AI\Pixal3D-ROCm` y ejecuta:
```bat
python_embeded\python.exe parche_comfyui_core_amd.py ComfyUI
```
Debe terminar con `RESUMEN: todo aplicado`. El script hace tres cosas:
- **Copia `amd_parche_rocm.py` a `custom_nodes`.** Corrige el fallo de ROCm por el que las multiplicaciones de más de 524 288 filas salen corruptas. Sin él, la textura sale magenta o con ruido.
- **Hace en CPU el solve del desenrollado UV.** Sin esto da `HIPBLAS_STATUS_ALLOC_FAILED`.
- **Añade el interruptor `COMFY_SIN_HIP=1`.** Solo es para pruebas.

Para comprobar el fallo y el parche en unos segundos:
```bat
python_embeded\python.exe prueba_linear_grande.py
```
Debe salir `el parche corrige el fallo`.

**Si actualizas ComfyUI, vuelve a ejecutar `parche_comfyui_core_amd.py`.**

## 5. Modelos
Guárdalos en `C:\AMD-AI\Pixal3D-ROCm\ComfyUI\models\<carpeta>\`:

| Carpeta | Archivo | URL |
|---|---|---|
| diffusion_models | pixal3d_bf16.safetensors | https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/diffusion_models/pixal3d_bf16.safetensors |
| diffusion_models | trellis_2_bf16.safetensors | https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/diffusion_models/trellis_2_bf16.safetensors |
| vae | trellis_2_shape_vae_bf16.safetensors | https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/vae/trellis_2_shape_vae_bf16.safetensors |
| vae | trellis_2_texture_vae_bf16.safetensors | https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/vae/trellis_2_texture_vae_bf16.safetensors |
| clip_vision | dino_v3_L_naf_fp32.safetensors | https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/clip_vision/dino_v3_L_naf_fp32.safetensors |
| geometry_estimation | moge_2_vitl_normal_fp16.safetensors | https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors |
| background_removal | birefnet.safetensors | https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors |

Ejemplo:
```bat
curl -L -o ComfyUI\models\diffusion_models\pixal3d_bf16.safetensors https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/diffusion_models/pixal3d_bf16.safetensors
```
**Usa los bf16, no los `int8_convrot`.** Los int8 dan peor color y, con `COMFY_SIN_HIP`, cierran ComfyUI por un fallo de `torch._int_mm`.

## 6. Arrancar y generar
```bat
cd /d C:\AMD-AI\Pixal3D-ROCm
python_embeded\python.exe ComfyUI\main.py --fp32-vae
```
- Al arrancar debe aparecer `[AMD] parche ROCm activo: Linear/matmul por trozos de hasta 262144 filas`.
- **`--fp32-vae` es necesario.** Sin él, los VAE dan NaN en la textura.
- Abre http://127.0.0.1:8188 y carga uno de estos dos workflows:
  - **`workflows/pixal3d_amd_bf16.json`**: arrástralo a la ventana. Ya trae los modelos bf16.
  - O la plantilla `3d_pixal3d_trellis2_image_to_model`, poniendo en «Switch Model Here» el cargador de arriba en `pixal3d_bf16` y el de abajo en `trellis_2_bf16`.
- En el booleano «Switch to Trellis2»: `false` usa Pixal3D (recomendado) y `true` usa TRELLIS.2.
- Los demás valores se quedan como en la plantilla: upsample 1536, remesh 768, 700 000 caras y textura de 4096.

## 7. Problemas conocidos
| Síntoma | Causa y solución |
|---|---|
| Textura magenta, con ruido o en franjas | Falta `amd_parche_rocm.py` en `custom_nodes`. Repite el paso 4. |
| `HIPBLAS_STATUS_ALLOC_FAILED` en Unwrap Mesh UVs | Falta el cambio en `parameterize.py`. Repite el paso 4. |
| `Trellis2 coords can't be empty` | El cargador que usa el booleano tiene el modelo equivocado. Con `false`, el de arriba debe ser `pixal3d_bf16`. |
| `python.exe` queda en 0 bytes («This app can't run on your PC») | Pasó dos veces sin causa clara. Restáuralo con `copy /Y C:\AMD-AI\python_exe_respaldo.exe C:\AMD-AI\Pixal3D-ROCm\python_embeded\python.exe`, o desde `cache\downloads\python-3.12.10-embed-amd64.zip`, y ponlo en solo lectura: `attrib +r python_embeded\python.exe`. |
| Una variable de entorno sigue activa (`AMD_DIAG_TEXTURA`, `COMFY_SIN_HIP`) | Abre una ventana de cmd nueva. |

## 8. Diagnóstico (solo si algo vuelve a fallar)
- `prueba_textura_amd.py`: compara GPU y CPU en desenrollado UV, rasterizado, muestreo, JFA y AO con una malla sintética.
- `amd_volcado_textura.py`: cópialo a `custom_nodes`. Guarda en `output\amd_horneado.npz` las entradas reales del horneado; `prueba_horneado_real.py` las analiza. Con `set AMD_DIAG_TEXTURA=1` compara además el decodificador de textura capa por capa entre GPU y CPU, lo que añade entre 30 y 60 minutos. **Quítalo de `custom_nodes` cuando no lo uses.**
- `prueba_numerica_amd.py`, `prueba_dinov3.py`, `prueba_ss_flow.py`: pruebas numéricas de TRELLIS.2.

## 9. Opcional: nodo de visualbruno (ComfyUI-Trellis2)
No es necesario para Pixal3D. Si quieres probarlo:
```bat
git clone https://github.com/visualbruno/ComfyUI-Trellis2 ComfyUI\custom_nodes\ComfyUI-Trellis2
python_embeded\python.exe -m pip install onnxruntime git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
python_embeded\python.exe parche_trellis2_amd.py ComfyUI\custom_nodes\ComfyUI-Trellis2
```
De su `requirements.txt` instala solo `meshlib`, `requests`, `pymeshlab`, `scipy`, `open3d`, `plotly` y `rembg`. No instales `opencv-python`, `torch`, `cumesh`, `flex_gemm`, `o_voxel` ni `nvdiffrast`.
El parche sustituye `nvdiffrast` por una versión en PyTorch, usa atención SDPA, y deja `triton` opcional y `explicit_gemm`.

## Respaldo de tu PC
Lo que solo existe en tu PC es la rueda compilada de cumesh, que ahorra todo el paso 2. Ejecuta `respaldar_pc.bat`: junta en `C:\AMD-AI\respaldo` la rueda, la lista de paquetes (`paquetes_pip.txt`), las versiones, tus workflows guardados y los archivos de ComfyUI modificados.

Ya está subida en el Release `amd-cumesh-gfx1200`. Si vuelves a compilarla, súbela así:
```bat
gh release upload amd-cumesh-gfx1200 C:\AMD-AI\wheels\cumesh-1.0-cp312-cp312-win_amd64.whl --repo GermanRmz/3dia --clobber
```
O desde la web, creando un *Release*:
1. Abre https://github.com/GermanRmz/3dia/releases/new.
2. En «Tag» escribe `amd-cumesh-gfx1200` y elige la rama `optimizacion-y-textura` como destino.
3. Arrastra `C:\AMD-AI\respaldo\cumesh-1.0-cp312-cp312-win_amd64.whl` y pulsa «Publish release».

Para reinstalarla después, descárgala desde esa página e instálala como se explica al principio del paso 2.
