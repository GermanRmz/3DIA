@echo off
rem Compila CuMesh para AMD (gfx1200 y gfx1201) y lo instala en el Python del kit.
rem Requisitos: Visual Studio 2022 Build Tools (C++), el venv C:\AMD-AI\compilar con los
rem paquetes ROCm/torch del kit, y C:\AMD-AI\CuMesh ya preparado con parche_cumesh_amd.py.
rem Ejecutar en cmd (no PowerShell). Si Defender bloquea clang: Add-MpPreference -ExclusionPath "C:\AMD-AI"

set VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
set VENV=C:\AMD-AI\compilar
set CUMESH=C:\AMD-AI\CuMesh
set WHEELS=C:\AMD-AI\wheels
set PYKIT=C:\AMD-AI\Pixal3D-ROCm\python_embeded\python.exe

call "%VCVARS%" || goto :error
for /f "delims=" %%i in ('%VENV%\Scripts\rocm-sdk.exe path --root') do set ROCM_HOME=%%i
set HIP_PATH=%ROCM_HOME%
set BUILD_TARGET=rocm
set GPU_ARCHS=gfx1200;gfx1201
set DISTUTILS_USE_SDK=1

rem hipcc falla con espacios/parentesis de "Program Files (x86)" en INCLUDE: usar rutas cortas
%VENV%\Scripts\python.exe -c "import os,ctypes;b=ctypes.create_unicode_buffer(1024);f=ctypes.windll.kernel32.GetShortPathNameW;r=[(b.value if f(p,b,1024) else p) for p in os.environ['INCLUDE'].split(';') if p];open(r'C:\AMD-AI\rutas_cortas.bat','w').write('set INCLUDE='+';'.join(r))" || goto :error
call C:\AMD-AI\rutas_cortas.bat

%ROCM_HOME%\bin\hipcc.exe --version || goto :error

cd /d %CUMESH%
if not exist %WHEELS% mkdir %WHEELS%
%VENV%\Scripts\python.exe -m pip wheel --no-deps --no-build-isolation . -w %WHEELS% || goto :error

for %%f in (%WHEELS%\cumesh-*.whl) do %PYKIT% -m pip install --no-deps --force-reinstall %%f || goto :error
echo.
echo CuMesh compilado e instalado. Prueba: %PYKIT% prueba_cumesh.py  y  prueba_bvh.py
goto :eof

:error
echo.
echo ERROR: revisa el mensaje de arriba (primer "error:" o "error LNK").
exit /b 1
