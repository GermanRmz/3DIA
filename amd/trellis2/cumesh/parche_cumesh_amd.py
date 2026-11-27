"""Prepara el codigo de CuMesh para compilarlo en Windows con ROCm (RX 9060 XT gfx1200 / 9070 XT gfx1201).

Reune todos los cambios que hicieron falta a mano:
  1. Copia del kit Pixal3D-TRELLIS.2-AMD-ROCm-WINDOWS: setup.py, cumesh/bvh.py y src/bvh.
  2. src/ext.cpp -> src/ext.cu (lo compila hipcc, no cl).
  3. src/clean_up.cu: cuda::std::tuple -> rocprim::tuple / rocprim::tie (no existe en AMD).
  4. src/bvh/bindings.cpp: TORCH_CHECK_VALUE/INDEX/TYPE -> TORCH_CHECK.
  5. src/bvh/bindings.cpp -> bindings.cu y third_party/xatlas/binding.cpp -> binding.cu, y
     setup.py apuntando a ellos (con cl el enlace falla: LNK2001 c10::ValueError).
Es idempotente. Uso:
    python parche_cumesh_amd.py C:\\AMD-AI\\CuMesh C:\\Pixal3D-TRELLIS.2-AMD-ROCm-WINDOWS
Commits probados: CuMesh d10e54c30ddd03d11472c1431693f985501c7966,
                  cuBQL  a7c1072119b34312e8b5dbf73a2e22628309acb7
"""
import os
import shutil
import sys

if len(sys.argv) < 3:
    sys.exit(__doc__)
cumesh = os.path.abspath(sys.argv[1])
kit = os.path.join(os.path.abspath(sys.argv[2]), "scripts", "native", "cubql-bvh")
for p in (cumesh, kit):
    if not os.path.isdir(p):
        sys.exit(f"No existe la carpeta {p}")


def leer(rel):
    with open(os.path.join(cumesh, rel), encoding="utf-8") as fh:
        return fh.read()


def escribir(rel, t):
    with open(os.path.join(cumesh, rel), "w", encoding="utf-8", newline="") as fh:
        fh.write(t)


# 1. archivos del kit (solo la primera vez, para no pisar los cambios posteriores)
marca = os.path.join(cumesh, ".parche_amd_kit")
if not os.path.exists(marca):
    shutil.copy2(os.path.join(kit, "setup.py"), os.path.join(cumesh, "setup.py"))
    shutil.copy2(os.path.join(kit, "cumesh", "bvh.py"), os.path.join(cumesh, "cumesh", "bvh.py"))
    shutil.copytree(os.path.join(kit, "src", "bvh"), os.path.join(cumesh, "src", "bvh"), dirs_exist_ok=True)
    open(marca, "w").close()
    print("[OK] archivos del kit copiados")
else:
    print("[ya estaba] archivos del kit")

# 2. ext.cu
shutil.copy2(os.path.join(cumesh, "src", "ext.cpp"), os.path.join(cumesh, "src", "ext.cu"))
print("[OK] src/ext.cu")

# 3. clean_up.cu con rocprim
for rel in ("src/clean_up.cu", "src/clean_up.hip"):
    if not os.path.exists(os.path.join(cumesh, rel)):
        continue
    t = leer(rel)
    n = (t.replace("__host__ __device__ ::cuda::std::tuple<int&, int&, int&> operator()(int3& key) const",
                   "__host__ __device__ ::rocprim::tuple<int&, int&, int&> operator()(int3& key) const")
          .replace("return {key.x, key.y, key.z};", "return ::rocprim::tie(key.x, key.y, key.z);"))
    if n != t:
        escribir(rel, n)
    print(f"[OK] {rel} ({'cambiado' if n != t else 'ya estaba'})")

# 4. TORCH_CHECK_* -> TORCH_CHECK
import re  # noqa: E402
for rel in ("src/bvh/bindings.cpp", "src/bvh/bindings_hip.cpp"):
    if os.path.exists(os.path.join(cumesh, rel)):
        t = leer(rel)
        escribir(rel, re.sub(r"TORCH_CHECK_(VALUE|INDEX|TYPE)\(", "TORCH_CHECK(", t))
        print(f"[OK] {rel}")

# 5. bindings como .cu y setup.py
shutil.copy2(os.path.join(cumesh, "src", "bvh", "bindings.cpp"), os.path.join(cumesh, "src", "bvh", "bindings.cu"))
shutil.copy2(os.path.join(cumesh, "third_party", "xatlas", "binding.cpp"),
             os.path.join(cumesh, "third_party", "xatlas", "binding.cu"))
t = leer("setup.py")
n = t.replace("bindings.cpp", "bindings.cu").replace("binding.cpp", "binding.cu")
escribir("setup.py", n)
ok = "bindings.cu" in n and "binding.cu" in n and "binding.cpp" not in n
print(f"[{'OK' if ok else 'FALLA'}] setup.py usa bindings.cu y xatlas/binding.cu")
print("\nListo. Ahora ejecuta compilar_cumesh.bat")
