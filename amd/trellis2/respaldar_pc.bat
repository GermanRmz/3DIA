@echo off
rem Junta en C:\AMD-AI\respaldo lo que solo existe en tu PC y no esta en GitHub:
rem la rueda compilada de cumesh, la lista exacta de paquetes, tus workflows guardados
rem y las copias de los archivos de ComfyUI modificados. No copia los modelos (son enormes
rem y se vuelven a descargar).
set RAIZ=C:\AMD-AI\Pixal3D-ROCm
set DEST=C:\AMD-AI\respaldo
if not exist %DEST% mkdir %DEST%

copy /Y C:\AMD-AI\wheels\cumesh-*.whl %DEST%\ >nul && echo [OK] rueda de cumesh
%RAIZ%\python_embeded\python.exe -m pip freeze > %DEST%\paquetes_pip.txt && echo [OK] paquetes_pip.txt
%RAIZ%\python_embeded\python.exe -c "import torch,sys;print(sys.version);print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))" > %DEST%\versiones.txt 2>&1 && echo [OK] versiones.txt
git -C %RAIZ%\ComfyUI rev-parse HEAD > %DEST%\comfyui_commit.txt 2>nul && echo [OK] comfyui_commit.txt
if exist %RAIZ%\ComfyUI\user\default\workflows (
  xcopy /E /I /Y %RAIZ%\ComfyUI\user\default\workflows %DEST%\workflows >nul && echo [OK] workflows guardados
)
copy /Y %RAIZ%\ComfyUI\comfy_extras\mesh3d\uv_unwrap\parameterize.py %DEST%\ >nul && echo [OK] parameterize.py
copy /Y %RAIZ%\ComfyUI\comfy\quant_ops.py %DEST%\ >nul && echo [OK] quant_ops.py
copy /Y %RAIZ%\ComfyUI\custom_nodes\amd_parche_rocm.py %DEST%\ >nul && echo [OK] amd_parche_rocm.py
dir %DEST%
echo.
echo Sube a GitHub la rueda .whl (ver INSTALAR_AMD.md, "Respaldo de tu PC") y guarda esta carpeta.
