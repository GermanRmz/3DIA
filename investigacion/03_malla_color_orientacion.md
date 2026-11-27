# 03. Malla, pulido, color y orientación

Fecha: 2026-11-25. Solo lectura de fuentes: **no se instaló, probó ni imprimió nada**.

Etiquetas:
- **[V]** verificado: leído en la fuente primaria (LICENSE, código, ficha PyPI o documentación oficial).
- **[F]** lo dice el fabricante o autor; no comprobado.
- **[N]** no verificado: conocimiento general, fuente secundaria o inferencia.

Versiones y ruedas (wheels) de Windows: API JSON de PyPI (`https://pypi.org/pypi/<paquete>/json`) y API de GitHub, consultadas hoy **[V]**. Que haya rueda no significa que se haya probado. El `venv` del proyecto usa Python 3.12.10. Contiene trimesh 5.2.0, scikit-image 0.26.0, scipy y **pymeshlab 2025.7.post1 (GPL3)**. No se vio manifold3d al buscarlo por nombre, aunque está en `requirements.txt`.

Las notas de licencia son lectura técnica, no consejo legal.

## Resumen: qué elegir

| Área | Recomendado | Alternativa | Hay que construir |
|---|---|---|---|
| Reparar | trimesh + parches locales + manifold3d; último escalón: libigl + scikit-image | Blender sin interfaz, en proceso aparte | Puerta de validación, mapa de desviación |
| Pulir | vtkConstrainedSmoothingFilter con límite por vértice | PyMeshLab TwoStep aparte, o no pulir | Máscara y detector de ojos con revisión humana |
| Color | scikit-image (Lab, CIEDE2000) + limpieza propia + 3MF con `paint_color` | Importador de GLB de Bambu Studio | Paleta real, islas, colores reservados |
| Orientar | Puntuador propio con trimesh | Tweaker-3 u orientación de Bambu | Restricción de resistencia por junta |

Hallazgos que cambian el plan:
1. **MeshLib gratis no sirve para vender figuras** (lectura prudente de su licencia).
2. **Bambu Studio v02.09.02.62 conserva `paint_color` de un 3MF genérico**, según el código. No toma de él los colores de los filamentos.
3. **Bambu Studio importa GLB con textura y lo convierte en pintura** desde la 2.7. Es la referencia a superar.
4. Hay perfiles oficiales de PETG para A1 mini con boquilla 0.2, pero Bambu advierte de menor resistencia.

## Cómo leer las licencias

| Licencia | (a) Uso personal | (b) Vender figuras | (c) Vender el programa |
|---|---|---|---|
| MIT, BSD, Apache 2.0 | Sí | Sí | Sí, conservando avisos |
| MPL 2.0 | Sí | Sí | Sí; se publican los archivos MPL modificados |
| GPL 3 | Sí | Sí: la salida solo queda cubierta si su contenido es una obra cubierta | Se puede vender, pero un conjunto combinado se distribuye bajo GPL. Proceso aparte: más defendible, no automático [N] |
| AGPL 3 | Como GPL | Como GPL | Como GPL, más obligaciones al ofrecer por red una versión modificada |
| No comercial | Según texto | No | No sin contrato |

Base: [GPLv3](https://www.gnu.org/licenses/gpl-3.0.txt) y [FAQ GNU](https://www.gnu.org/licenses/gpl-faq.en.html). gnu.org no respondió a la descarga directa: la frase sobre la salida se leyó en extractos de buscador. Consultar a un abogado antes de vender el programa.

La licencia de una herramienta no da derechos sobre el personaje, la foto o el modelo 3D de entrada.

## 1. Reparar y cerrar la malla

No se encontraron cifras primarias de tiempo o RAM con 700 mil caras para ninguna herramienta. Todo lo de esta tabla puede usarse en CPU. **Hay que medirlo.**

| Herramienta | Versión; Windows/Python | Licencia; (a)/(b)/(c) | Qué hace y límites |
|---|---|---|---|
| [trimesh](https://pypi.org/project/trimesh/) | 5.2.2 (2026-11-22); Python >=3.10, sin código compilado [V] | [MIT](https://raw.githubusercontent.com/mikedh/trimesh/main/LICENSE.md) [V]; sí/sí/sí | Carga, soldado y medidas; no es un reparador general [N] |
| [manifold3d](https://pypi.org/project/manifold3d/) | 3.5.5 (2026-11-13); Win 3.9-3.15 [V] | [Apache 2.0](https://raw.githubusercontent.com/elalish/manifold/master/LICENSE) [V]; sí/sí/sí | Booleanas con "salida manifold garantizada" [F, [README](https://github.com/elalish/manifold)]. **Si la entrada no es 2-manifold orientada, devuelve vacío y error** [V, [doc](https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html)]. Lleva propiedades por vértice. `LevelSet` crea un sólido desde una SDF [V] |
| [MeshLib](https://pypi.org/project/meshlib/) | 3.2.0.75 (2026-11-23); Win, Python >=3.8 [V] | [No comercial, evaluación o educación](https://raw.githubusercontent.com/MeshInspector/MeshLib/master/LICENSE) [V]. (a) personal no comercial: sí. (b) **no** (lectura prudente). (c) **no**, prohíbe transferirlo. Comercial: cuota anual [F, [licencias](https://meshlib.io/license/)]; precio no encontrado | Reparación, huecos, SDF [F] |
| [PyMeshLab](https://pypi.org/project/pymeshlab/) | 2025.7.post1 (2026-01-30); Win 3.10-3.14 [V] | [GPL 3](https://raw.githubusercontent.com/cnr-isti-vclab/PyMeshLab/main/LICENSE) [V]; sí/sí/condiciones GPL | `meshing_close_holes`, `meshing_repair_non_manifold_*`, `generate_alpha_wrap`, `generate_resampled_uniform_mesh` [V, [filtros](https://pymeshlab.readthedocs.io/en/latest/filter_list.html)] |
| [pymeshfix](https://pypi.org/project/pymeshfix/) | 0.18.1 (2026-04-23); Win 3.10-3.12 [V] | LICENSE: [GPL 3](https://raw.githubusercontent.com/pyvista/pymeshfix/main/LICENSE); PyPI: AGPL 3. **Contradicción** [V]. Sí/sí/tratar como AGPL | Deja intactas las zonas sin defecto; supone **un solo sólido cerrado** [F, [README](https://github.com/pyvista/pymeshfix)]. Puede perder accesorios sueltos |
| [libigl](https://pypi.org/project/libigl/) | 2.6.3 (2026-09-01); Win 3.8-3.12 [V] | Núcleo [MPL 2.0](https://raw.githubusercontent.com/libigl/libigl/main/LICENSE.MPL2); sí/sí/sí con cuidado. El [CMake de main](https://github.com/libigl/libigl-python-bindings/blob/main/CMakeLists.txt) activa por defecto `copyleft` (CGAL, TetGen) [V]: para (c), **compilar sin copyleft** y revisar el paquete final | El árbol de bindings contiene `fast_winding_number`, `signed_distance` y `marching_cubes` [V]. Hay otro `marching_cubes` en `copyleft`: no usar ese |
| [CGAL (bindings)](https://pypi.org/project/cgal/) | 6.0.1.post202410241521; Win 3.8-3.12 [V] | GPL 3+ [V, PyPI] o comercial de GeometryFactory ([código](https://sources.debian.org/src/cgal/6.1.1-2/include/CGAL/alpha_wrap_3.h), leído en extracto); sí/sí/GPL o pago | [Alpha Wrap](https://doc.cgal.org/latest/Alpha_wrap_3/index.html): cerrado, 2-manifold y sin autointersecciones; **encierra** la entrada y **filtra detalle fino y cavidades** [V]. Puede engordar y pegar dedos (inferencia). No comprobado que estos bindings lo incluyan; PyMeshLab sí |
| [OpenVDB](https://docs.openvdb.org/python.html) | No existe `openvdb` en PyPI (404 hoy) [V] | [Apache 2.0](https://raw.githubusercontent.com/AcademySoftwareFoundation/openvdb/master/LICENSE) [V]; sí/sí/sí | El módulo Python expone "casi ninguna" de las herramientas C++ [V]. Compilar en Windows: coste alto [N] |
| [Blender / bpy](https://pypi.org/project/bpy/) | bpy 5.3.0 (2026-11-10), **solo Python 3.13** [V]; fuera del venv 3.12. Usar `blender.exe --background` | [GPL, sin otra licencia](https://raw.githubusercontent.com/Blender/blender/main/COPYING) [V]; sí/sí/condiciones GPL | Remesh Voxel usa OpenVDB y genera una malla manifold nueva [V, [manual](https://docs.blender.org/manual/en/latest/modeling/modifiers/generate/remesh.html)]. Conservación de UV: [N] |
| [Open3D](https://pypi.org/project/open3d/) | 0.20.0 (2026-09-16); Win 3.10-3.14 [V] | [MIT](https://raw.githubusercontent.com/isl-org/Open3D/main/LICENSE) [V]; sí/sí/sí | Distancias y visor [N] |
| [VTK](https://pypi.org/project/vtk/) | 9.7.2 (2026-11-04); Win 3.10-3.15 [V] | BSD [V, PyPI]; sí/sí/sí | Suavizado y medición; ver área 2 |

fTetWild e instant-meshes no se evaluaron.

### Recomendación

1. Guardar el GLB original sin tocar (UV y textura).
2. En una copia, soldar por **posición** aunque difieran las UV. Las costuras UV no son huecos reales.
3. Medir aristas abiertas, aristas no manifold, componentes, autointersecciones y volumen.
4. Si los huecos son pequeños: cierre local. Aceptar solo si manifold3d construye el objeto.
5. Si falla: SDF con signo por número de vueltas (libigl) y marching cubes de [scikit-image](https://pypi.org/project/scikit-image/) (0.26.0, BSD, Win 3.11-3.14 [V]). Esta ruta no necesita `copyleft`.
6. Copiar color y etiquetas desde el maestro por punto más cercano **comprobando la normal**. Sin ello un dedo puede tomar el color del vecino. La rama de color usa etiquetas por cara; no necesita UV en el sólido.

Cuenta, no medición: un cubo de 100 mm con paso 0.2 mm son 500 celdas por lado, 125 millones de valores, unos 0.5 GB en float32 o 1 GB en float64, sin contar índices ni malla. Conviene trabajar por bloques y afinar solo ojos y manos.

**Alternativa:** Blender Voxel como programa externo. MeshLib puede servir de referencia en una evaluación no comercial, no en la línea de venta. No usar Alpha Wrap para el acabado final.

### Hay que construir

- Estados `válida / reparada / fallida`, sin caídas silenciosas.
- Distancia en ambos sentidos por zona (ojos, dedos, resto).
- Detección de doble superficie, cavidades internas y dedos pegados. Cerrado no significa correcto.
- Control de autointersecciones: construir en manifold3d no prueba su ausencia [N].

### Prueba concreta

Cinco GLB reales de ~700 mil caras a 10 cm. Rutas: local + manifold3d, libigl a 0.2 y 0.1 mm, Blender a 0.2 mm, pymeshfix. Registrar tiempo, pico de RAM, aristas abiertas, componentes, volumen, desviación p95 y máxima por zona, y `--info` de Bambu. Pasa si: 0 aristas abiertas, manifold3d lo acepta, no se pierden componentes protegidos y la desviación en ojos y dedos no supera el umbral del autor (propuesta: 0.1 mm).

## 2. Pulido selectivo con máscara

| Herramienta | Versión y Windows | Licencia; (a)/(b)/(c) | Utilidad y límite |
|---|---|---|---|
| [vtkConstrainedSmoothingFilter](https://vtk.org/doc/nightly/html/classvtkConstrainedSmoothingFilter.html) | VTK 9.7.2 [V] | BSD; sí/sí/sí | Acepta una **matriz de restricción por punto** [V]: sirve para un máximo en mm. Es Laplaciano y, con límites amplios, puede autointersecar [V]. No detecta bordes |
| PyMeshLab TwoStep y Taubin | 2025.7.post1 [V] | GPL 3; sí/sí/condiciones | TwoStep "conserva o realza rasgos", con umbral de ángulo; opción de solo caras seleccionadas en Taubin [V, [doc](https://pymeshlab.readthedocs.io/en/latest/filter_list.html)]; confirmar en TwoStep |
| [robust_laplacian](https://pypi.org/project/robust_laplacian/) 1.1.0 y [potpourri3d](https://pypi.org/project/potpourri3d/) 1.4.0 | Win 3.9-3.14 [V] | MIT [V]; sí/sí/sí | Base para filtro propio y distancias geodésicas |
| libigl | 2.6.3 [V] | MPL, misma cautela | Curvatura. No se halló filtro bilateral en el árbol de bindings [V, por nombre de archivo] |
| [SAM 2](https://github.com/facebookresearch/sam2) | Windows/ROCm sin comprobar | Código [Apache 2.0](https://raw.githubusercontent.com/facebookresearch/sam2/main/LICENSE) [V]; pesos [N] | Segmenta desde punto o caja; no reconoce "ojo" por texto [N] |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | Igual | Código [Apache 2.0](https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/LICENSE) [V]; pesos [N] | Cajas por texto. Precisión en chibis animales: sin dato |
| [SAM 3](https://github.com/facebookresearch/sam3) | Igual | [Licencia propia](https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE) [V que es propia]; condiciones sin analizar | No elegir hasta leerla |
| [MediaPipe](https://pypi.org/project/mediapipe/) | 1.2.3 (2026-11-11); Win [V] | Apache 2.0 [V, PyPI]; sí/sí/sí | Pensado para caras humanas [N] |
| [Ultralytics](https://pypi.org/project/ultralytics/) | 8.4.218 (2026-11-23) [V] | AGPL 3 [V]; sí/sí/complica la venta | Evitar |

### Recomendación

Máscara = zonas pintadas por el autor + ojos detectados + bordes de color + bordes geométricos. Agrandarla unos anillos con transición suave. Peso por vértice de 0 (protegido) a 1; distancia permitida = peso x límite en mm. Rechazar el resultado si hay cruces o se supera el límite.

Medir **antes** de pulir: si el ruido es menor que la capa, no pulir. Pulir la malla no quita líneas de capa.

Ojos: renderizar vistas fijas con mapa de índice de cara, detectar en 2D y votar en 3D. Empezar por **color de textura + clics del autor**; añadir SAM 2 solo si ahorra trabajo. Bordes: ángulo diedro y salto de color Lab entre caras vecinas, con trimesh y NumPy.

**Alternativa:** TwoStep en proceso aparte, o Blender con grupo de vértices. También se puede no pulir.

### Hay que construir

Pintor de máscara, votación de vistas a caras, control de cruces y paredes, y un filtro bilateral de normales con máscara si VTK borra bordes.

### Prueba concreta

Dos figuras con 10 puntos marcados en ojos, dedos y bordes. Comparar sin pulir, VTK y TwoStep, con límites 0.05 y 0.1 mm. Medir desplazamiento en la máscara (debe ser 0), máximo fuera, rugosidad, cruces y preferencia a ciegas sobre una cabeza impresa. Detector: 10 figuras, contando ojos hallados y falsos positivos.

## 3. Color a 4 filamentos y 3MF

| Herramienta | Versión; Windows | Licencia; (a)/(b)/(c) | Nota |
|---|---|---|---|
| [scikit-image](https://scikit-image.org/docs/stable/api/skimage.color.html) | 0.26.0; Win 3.11-3.14 [V] | BSD; sí/sí/sí | `deltaE_ciede2000` [V]. Suficiente y ya instalado |
| [colour-science](https://pypi.org/project/colour-science/), [coloraide](https://pypi.org/project/coloraide/) | 0.4.7, 8.17; Python >=3.11 [V] | BSD-3, MIT; sí/sí/sí | Opcionales |
| [OpenCV](https://pypi.org/project/opencv-python/) | 5.1.0.95 (2026-11-17); Win [V] | Apache 2.0 [V]; sí/sí/sí | Salto a 5.x sin probar en el proyecto |
| [PyMaxflow](https://pypi.org/project/PyMaxflow/) | 1.3.2; Win hasta 3.13 [V] | GPL [V]; sí/sí/condiciones | Evitar |
| [gco-wrapper](https://pypi.org/project/gco-wrapper/) | 3.0.9; Win hasta 3.11 [V] | Envoltorio MIT [V]; núcleo gco sin revisar [N] | Sin rueda 3.12. Evitar |
| [lib3mf](https://pypi.org/project/lib3mf/) | 2.5.0 (2026-02-23); Win [V] | [BSD](https://raw.githubusercontent.com/3MFConsortium/lib3mf/master/LICENSE) [V]; sí/sí/sí | 3MF estándar. Soporte de `paint_color`: [N] |
| [Intrinsic](https://github.com/compphoto/Intrinsic) | - | [Solo académico](https://raw.githubusercontent.com/compphoto/Intrinsic/main/LICENSE) [V]; no apto para (b)/(c) | Descartar |
| [Marigold IID](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1) | v1-1 | OpenRAIL++ según ficha [V]; restricciones sin leer | Albedo por imagen. Plan B; VRAM, ROCm y coherencia entre vistas [N] |
| Quitado de luz de [Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) | - | `tencent-hunyuan-community` [V, ficha] | Solo si se conserva Hunyuan |

**Sombras:** pedir imagen de color plano y usar solo el albedo del GLB. Al agrupar, dar menos peso a L que a a/b y proteger negro y blanco, donde L es la información. Esto atenúa; no elimina.

**Paleta:** muestras impresas de los filamentos reales; elegir los 4 que minimizan CIEDE2000 ponderado por área, con peso extra en ojos y colores reservados. No usar centros k-means que no existen como filamento.

**Islas:** componentes de caras vecinas medidas en mm², uniendo las pequeñas al vecino con mejor equilibrio entre color y frontera, sin tocar las protegidas. Suavizado de etiquetas propio (ICM). El ancho mínimo depende de la boquilla: se mide impreso. Subdividir caras grandes en fronteras.

### Formato de Bambu Studio, leído en el código v02.09.02.62

- [bbs_3mf.cpp](https://github.com/bambulab/BambuStudio/blob/v02.09.02.62/src/libslic3r/Format/bbs_3mf.cpp) lee `paint_color` de cada `<triangle>` **sin exigir que el archivo sea de Bambu** [V].
- [TriangleSelector.cpp](https://github.com/bambulab/BambuStudio/blob/v02.09.02.62/src/libslic3r/TriangleSelector.cpp): triángulo entero con filamento 1=`4`, 2=`8`, 3=`0C`, 4=`1C` [V]. Coincide con `multicolor.py`. Solo hexadecimal en **mayúsculas**; si no, se ignora el triángulo. La misma cadena admite triángulos subdivididos.
- "De Bambu" significa que el metadato `Application` empieza por `BambuStudio-` [V]. El nuestro pone `3DIA`: genérico.
- Con un genérico avisa *"The 3mf is not from Bambu Lab, load geometry data and color data only"* [V, [Plater.cpp](https://github.com/bambulab/BambuStudio/blob/v02.09.02.62/src/slic3r/GUI/Plater.cpp)]. Si hay pintura, **no** abre el diálogo de materiales estándar: conserva índices y añade los filamentos que falten **con colores elegidos por el programa** [V]. El usuario debe asignar sus 4 colores.
- Sin pintura, los colores estándar pasan al flujo de importación de color. Las **texturas dentro del 3MF se descartan** [V].
- La detección "Prusa" busca `PrusaSlicer` en el metadato, no el espacio de nombres `slic3rpe` [V, [3mf.cpp](https://github.com/bambulab/BambuStudio/blob/v02.09.02.62/src/libslic3r/Format/3mf.cpp)]. Mantener `slic3rpe:mmu_segmentation` no la activa.
- La 2.9.1 mejora la lectura de 3MF estándar [F, [notas](https://github.com/bambulab/BambuStudio/releases/tag/v02.09.01.70)]. Fijar la versión: otras pueden actuar distinto. La v02.09.02.62 salió el 2026-11-24.
- No falsear `Application`: cambia la ruta de carga y promete ajustes completos. Para un proyecto completo, partir de una plantilla guardada por Bambu Studio y sustituir la malla; sin probar.

**GLB directo:** desde v02.07.00.55 convierte OBJ, glTF y GLB con textura en pintura; recomienda una sola textura y en esa beta no admite Draco [F, [notas 2.7](https://github.com/bambulab/BambuStudio/releases/tag/v02.07.00.55)]. La 2.7.1 detecta y repara mallas al importar y puede llamar al reparador de Windows [F, [notas 2.7.1](https://github.com/bambulab/BambuStudio/releases/tag/v02.07.01.57)]. No se halló cómo hacerlo por línea de comandos.

Licencia: [AGPL 3](https://raw.githubusercontent.com/bambulab/BambuStudio/master/LICENSE) [V]. (a) sí; (b) sí; (c) invocar la copia instalada por el usuario; no copiar ni enlazar código sin aceptar AGPL.

### Laminar y validar por línea de comandos

La [wiki](https://github.com/bambulab/BambuStudio/wiki/Command-Line-Usage) documenta `--slice` (0=todas), `--orient`, `--arrange`, `--load-settings`, `--load-filaments`, `--export-3mf`, `--info` y `--outputdir` [V]. Los JSON deben ser **completos**, no los perfiles heredados de `resources/profiles/BBL`. Prioridad: línea de comandos, luego archivos, luego 3MF.

Forma propuesta, sin ejecutar:

```
bambu-studio.exe --info modelo.3mf
bambu-studio.exe --load-settings "maquina.json;proceso.json" --load-filaments "f1.json;f2.json;f3.json;f4.json" --slice 0 --debug 2 --outputdir salida --export-3mf resultado.3mf modelo.3mf
```

- `--info` trabaja con la malla sin reparar (comentario del [código](https://github.com/bambulab/BambuStudio/blob/v02.09.02.62/src/BambuStudio.cpp)) e imprime `manifold = yes/no` ([Model.cpp](https://github.com/bambulab/BambuStudio/blob/v02.09.02.62/src/libslic3r/Model.cpp)) [V].
- Con un 3MF no Bambu, pedir una placa concreta se convierte en 0; `--import-positions` exige 3MF Bambu [V].
- Códigos [V]: 0 correcto; -51 validación; -52 parcialmente dentro; -59 demasiados triángulos; -62 filamento/cama; -100 laminado; -101 choque de rutas; -104 ruta fuera. Windows puede mostrar los negativos sin signo [N].
- Laminar sin error no prueba que la malla sea buena. Conservar ambas puertas.
- Funcionamiento en esta PC, y con varios filamentos: [N].

### Menos purga con AMS lite

Según [Bambu](https://wiki.bambulab.com/en/software/bambu-studio/reduce-wasting-during-filament-change) [V]:
- En casi todos los casos, cambios por capa = filamentos de esa capa menos uno.
- Agrupar los cambios abajo acorta la torre. En su ejemplo, de 14.76 a 9.05 g; no es trasladable a nuestras figuras.
- Multiplicador de purga 0.8-0.9 como inicio, con prueba de mezcla.
- Purga en soporte: activa por defecto, solo con torre. Purga en relleno: el color puede transparentarse. Purga en un objeto sacrificable: admite color irregular.
- Imprimir varias unidades reparte la purga. Separar piezas de un solo color ayuda solo si no comparten capas con otros colores en la placa (inferencia).

Pasar de capa 0.16 a 0.08 duplica las capas y puede casi duplicar los cambios. **Menos líneas y menos purga se oponen.**

### Recomendación, alternativa y faltantes

- **Recomendación:** etiqueta por cara en Lab, 4 filamentos reales, limpieza propia y 3MF genérico con `paint_color` más hoja "índice -> filamento".
- **Alternativa:** importar el GLB en Bambu Studio y elegir 4 colores a mano.
- **Construir:** catálogo de filamentos, colores reservados, caras internas fijas, islas protegidas, escape del XML, prueba automática de los cuatro códigos y límite de 4 colores.

### Prueba concreta

Cubo con índices 1-4, un nombre con caracteres especiales e islas de 0.2, 0.4, 0.8 y 1.2 mm. Abrir, comprobar, asignar filamentos, guardar, reabrir y laminar por CLI. Repetir con la perrita: nuestro 3MF contra GLB importado por Bambu. Medir error de color en 20 puntos, islas, gramos de purga, cambios y minutos. Pasa solo si los índices 1-4 sobreviven a guardar y laminar.

## 4. Orientación, capas, boquilla y acabado

| Opción | Versión y Windows | Licencia; (a)/(b)/(c) | Qué considera |
|---|---|---|---|
| [Tweaker-3](https://github.com/ChristophSchranz/Tweaker-3) | Release 3.9.1 (2021-10-14); commit más reciente 2026-03-24; Python + NumPy; sin paquete `tweaker3` en PyPI [V]. Windows sin probar | [GPL 3](https://raw.githubusercontent.com/ChristophSchranz/Tweaker-3/master/LICENSE) [V]; sí/sí/condiciones | Minimiza soporte por volumen o superficie; `--favside` da peso a **una** dirección [V]. No conoce juntas |
| [Auto-orientar de Bambu](https://wiki.bambulab.com/en/software/bambu-studio/auto-orientation) | 02.09.02.62 | AGPL; sí/sí/externo | Voladizo, base, estabilidad y superficie aparente; admite errores por puentes [V]. No figura la fuerza de junta |
| Propio: trimesh + [embreex](https://pypi.org/project/embreex/) | embreex 4.4.0.post1 (2026-10-21); Win 3.9-3.15 [V] | MIT + texto de licencia embreex por leer completo [N] | Hay que programarlo y calibrarlo |

**Recomendación:** generar unas 500 direcciones y apoyos naturales; descartar lo que no quepa en la cama; luego evaluar:
1. **Resistencia, como restricción:** descartar ejes y cuellos cuya tracción o flexión separe capas. En general, eje acostado.
2. **Vista:** en zonas marcadas, escalón geométrico aproximado = altura de capa x |n_z|. Calibrar con impresiones.
3. **Soporte:** área y contactos, con castigo en ojos, dedos y pasadores.
4. **Estabilidad, altura, tiempo y torre:** desempates.

Mostrar tres opciones del frente de Pareto al autor, no una nota única opaca. **Alternativa:** Tweaker-3 con `--favside` aparte, o Bambu más corrección manual. Si cara y junta piden direcciones contrarias: **separar la junta como pieza**.

Resistencia: un [estudio con PET-G y PLA](https://pmc.ncbi.nlm.nih.gov/articles/PMC9230522/) sitúa la orientación como el factor de más efecto sobre la resistencia última, por delante del relleno y la temperatura [V]. No se dan porcentajes aquí: son probetas, no juntas. Un [segundo estudio](https://www.mdpi.com/2504-4494/10/2/64) no se pudo abrir.

### Capas y boquilla

- Capas variables con 0.4: rango 0.08-0.28 mm según [Bambu](https://wiki.bambulab.com/en/software/bambu-studio/adaptive-layer-height) [V]. Ayudan en pendientes suaves; no borran las rayas de paredes verticales.
- Varios colores y torre exigen el mismo perfil de alturas en los objetos de la placa [V, misma página].
- El perfil va en `Metadata/layer_heights_profile.txt` [V, constante en bbs_3mf.cpp]. Carga desde un genérico: [N]. Primera versión: usar la interfaz de Bambu.
- [Perfil A1 mini 0.2](https://github.com/bambulab/BambuStudio/blob/v02.09.02.62/resources/profiles/BBL/machine/Bambu%20Lab%20A1%20mini%200.2%20nozzle.json): capas 0.04-0.14 [V]. Hay procesos oficiales de 0.06 a 0.14.
- **PETG + 0.2:** la [tabla](https://wiki.bambulab.com/en/general/filament-guide-material-table) marca compatibles PETG y PETG HF; no PETG-CF ni Translucent (aunque existe un perfil Translucent 0.2: discrepancia) [V]. Hay [perfiles](https://github.com/bambulab/BambuStudio/tree/v02.09.02.62/resources/profiles/BBL/filament) PETG Basic, HF, Matte y Generic para A1M 0.2 [V].
- **AMS lite:** la tabla desaconseja blandos, quebradizos, rugosos y PETG-CF [V]. No se leyó una casilla para PETG normal. La combinación PETG + 0.2 + AMS lite es **probable, no certificada** [N].
- La [FAQ 0.2](https://wiki.bambulab.com/en/knowledge-sharing/02-mm-nozzle-FAQ) avisa de atascos, cambios de color lentos y **menor resistencia** de la mayoría de piezas [V].
- Flujo máximo en los JSON: PETG Basic y HF 0.2, 1 mm3/s; PETG HF 0.4, 18 mm3/s [V]. La purga con 0.2 será mucho más lenta.

**Recomendación:** 0.4 con capas 0.08-0.12 para juntas y 4 colores; 0.2 para cabezas o piezas pequeñas sin carga, con PETG seco y sin partículas. Son trabajos separados. **Alternativa:** 0.4 a 0.12-0.16 y lijar.

### Posproceso de PETG

- [Prusa](https://help.prusa3d.com/article/petg_2059): se lija fácil en seco o mojado; disolver y suavizar solo en laboratorio, por personal cualificado, con diclorometano [V]. **No usar vapor casero.**
- Ruta propuesta: lijado 200/400/800 ([3D Insider](https://3dinsider.com/petg-post-processing), secundaria), imprimación de relleno, lijado y pintura [N]. Probar adherencia con su PETG.
- No lijar ni pintar dentro de las juntas; medir el espesor de pintura y reservarlo en la holgura. Lijar en mojado o con protección.
- Elegir costuras accesibles para lijar.

### Hay que construir

Zonas visibles y cargas de junta, puntuador, visor de tres opciones, reparto en placas y bloqueo si no cabe. El cubo de 180 mm no es todo útil: validar margen y torre al laminar.

### Prueba concreta

Cabeza de 35 mm y cupón de eje en dos orientaciones, en PETG: 0.4 a 0.16, 0.08 y variable; 0.2 a 0.06 y 0.08; una cabeza más con lijado e imprimación. Tres repeticiones del eje. Medir horas, gramos, contactos de soporte, fotos a 30 cm con luz rasante, ciclos y rotura. Comparar Tweaker, Bambu y el puntuador en 10 piezas, a ciegas.

## Orden sugerido

1. Cubo 3MF de índices: resuelve el riesgo de formato.
2. Cinco GLB y rutas de reparación.
3. Cupón de eje y cabeza, antes de programar el puntuador.
4. Color de la perrita frente al importador de Bambu.
5. Pulido, solo si la cabeza impresa muestra ruido que no son capas.

## Lo que no se pudo verificar

- Tiempo y RAM a 700 mil caras. Los umbrales de prueba son propuestas, no datos.
- Calidad en ojos y dedos de cualquier reparador.
- Detectores y albedo en gfx1200; licencias de todos los pesos.
- CLI real, GLB de Pixal3D en Bambu y PETG/0.2/AMS lite.
- Alcance de GPL en un producto vendido; contradicción GPL/AGPL de pymeshfix.
- Precio comercial de MeshLib; núcleo de gco; texto completo de embreex.
