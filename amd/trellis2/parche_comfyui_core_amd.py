"""Aplica a una instalacion de ComfyUI todos los cambios para AMD (ROCm en Windows).

1. Copia amd_parche_rocm.py a ComfyUI/custom_nodes (Linear/matmul por trozos de 2^18 filas;
   sin esto la textura y parte de la forma de TRELLIS.2/Pixal3D salen corruptas).
2. comfy_extras/mesh3d/uv_unwrap/parameterize.py: el solve fp64 del LSCM se hace en CPU
   (hipBLAS fp64 por lotes da HIPBLAS_STATUS_ALLOC_FAILED en el desenrollado UV).
3. comfy/quant_ops.py: interruptor COMFY_SIN_HIP=1 para desactivar los kernels hip de
   comfy_kitchen (solo para pruebas; normalmente no se usa).

Es idempotente: se puede ejecutar varias veces. Guarda copias .orig de los originales.
Uso (desde C:\\AMD-AI\\Pixal3D-ROCm, con ComfyUI cerrado):
    python_embeded\\python.exe parche_comfyui_core_amd.py ComfyUI
Probado con ComfyUI v0.45.0. Repetir despues de actualizar ComfyUI.
"""
import os
import shutil
import sys

aqui = os.path.dirname(os.path.abspath(__file__))
comfy = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(aqui, "ComfyUI"))
if not os.path.isfile(os.path.join(comfy, "main.py")):
    sys.exit(f"No encuentro ComfyUI en {comfy} (falta main.py)")

errores = 0


def editar(rel, viejo, nuevo, marca):
    global errores
    ruta = os.path.join(comfy, rel)
    if not os.path.isfile(ruta):
        print(f"[FALLA] no existe {rel}")
        errores += 1
        return
    with open(ruta, encoding="utf-8") as fh:
        t = fh.read()
    if marca in t:
        print(f"[ya estaba] {rel}")
        return
    if viejo not in t:
        print(f"[FALLA] {rel}: no encuentro el texto a cambiar (version de ComfyUI distinta?)")
        errores += 1
        return
    if not os.path.exists(ruta + ".orig"):
        shutil.copy2(ruta, ruta + ".orig")
    with open(ruta, "w", encoding="utf-8", newline="") as fh:
        fh.write(t.replace(viejo, nuevo, 1))
    print(f"[OK] {rel}")


# 1. nodo con el parche de multiplicaciones por trozos
origen = os.path.join(aqui, "amd_parche_rocm.py")
destino = os.path.join(comfy, "custom_nodes", "amd_parche_rocm.py")
if os.path.isfile(origen):
    shutil.copy2(origen, destino)
    print("[OK] custom_nodes/amd_parche_rocm.py")
else:
    print("[FALLA] falta amd_parche_rocm.py junto a este script")
    errores += 1

# 2. LSCM en CPU
editar("comfy_extras/mesh3d/uv_unwrap/parameterize.py",
       "x = torch.linalg.solve(AtA, Atb).cpu().numpy()",
       "x = torch.linalg.solve(AtA.cpu(), Atb.cpu()).numpy()  # AMD: hipBLAS fp64 batched falla",
       "# AMD: hipBLAS fp64 batched falla")

# 3. interruptor COMFY_SIN_HIP
editar("comfy/quant_ops.py",
       "    for k, v in ck.list_backends().items():",
       "    import os as _os\n"
       "    if _os.environ.get('COMFY_SIN_HIP'):\n"
       "        ck.registry.disable('hip')  # AMD: prueba sin kernels hip\n"
       "    for k, v in ck.list_backends().items():",
       "COMFY_SIN_HIP")

print("\nRESUMEN:", "todo aplicado" if not errores else f"{errores} cambio(s) sin aplicar")
sys.exit(1 if errores else 0)
