# 02. De imagen a 3D con textura, en local y con AMD

Fecha de consulta: 2026-11-25. No se descargó ni se ejecutó ningún modelo. Solo se leyeron páginas, licencias y código.

Equipo: RX 9060 XT 16 GB (gfx1200), Windows 10, 32 GB de RAM, ROCm 7.2.1, ComfyUI v0.45.0.

**Etiquetas**
- **[V]** verificado: leído en la fuente primaria (licencia, ficha del modelo, lista de archivos, código).
- **[F]** afirmación del fabricante o de los autores, sin comprobación independiente.
- **[C]** dato de un tercero (usuario, issue, kit de la comunidad), no repetido aquí.
- **[L]** dicho por las guías de este repositorio; no se repitió.
- **[NV]** no verificado.
- **[I]** deducción propia.

El resumen de licencias no es asesoría legal. Antes de vender, guardar copia con fecha de cada licencia.

## 1. Respuesta corta

- **Nada de lo encontrado justifica sustituir Pixal3D hoy.** No se halló una alternativa de malla con textura que tenga pruebas independientes de ser mejor, quepa en 16 GB con AMD y Windows, y tenga licencia igual de abierta.
- **Recomendación: seguir con Pixal3D** bf16 en ComfyUI v0.45.0. Una vista como base; cuatro vistas solo si ganan la prueba del apartado 10.
- **Alternativa (fallback): TRELLIS.2-4B**, que ya está en el mismo workflow. Comparte VAE, DINOv3, horneado y remallado con Pixal3D, así que no protege de un fallo en esas piezas.
- **Novedad más interesante: UltraTex** (octubre de 2026). Retexturiza una malla; no genera la forma. Sus pesos dependen de una base no comercial: solo para prueba personal.
- **Lo pendiente no es el modelo**: medir si la malla es cerrada, controlar las sombras horneadas y conectar ComfyUI por HTTP.

## 2. Comparación principal

| Modelo y versión | Entrada | Salida | VRAM y disco | Windows + AMD | Licencia de pesos |
|---|---|---|---|---|---|
| **Pixal3D**, [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) rev. `b0cb2e1b79`; [Comfy-Org/Pixal3D](https://huggingface.co/Comfy-Org/Pixal3D) rev. `f37641be37` | 1 imagen; pesos `_mv` para varias | Malla + textura | Original 46,05 GB; Comfy bf16 11,0 GB, y otros 11,0 GB para multivista [V]. VRAM [NV] | **Funciona en esta PC** [L] | MIT [V] |
| **TRELLIS.2-4B**, [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) rev. `af44b45f2e` | 1 imagen | Malla + PBR con opacidad [F] | 24 GB mínimo y Linux, según el código oficial [F]; 16,24 GB; Comfy bf16 10,34 GB [V] | **Funciona en esta PC** con parches [L] | MIT [V] |
| **Hunyuan3D-2.0**, [tencent/Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) | 1 imagen | Malla + textura; PBR [NV] | 6 GB forma, 16 GB con textura [F]; repositorio completo 74,89 GB [V] | El autor del kit AMD dice que funciona, sin mediciones [C] | Comunidad Tencent [V] |
| **Hunyuan3D-2mv**, [tencent/Hunyuan3D-2mv](https://huggingface.co/tencent/Hunyuan3D-2mv) | Varias vistas | Malla; textura con el modelo 2.0 | 4,93 GB por variante; repositorio 29,58 GB [V] | Nodos en el núcleo de ComfyUI [V]; en esta PC [NV] | Enlaza la de 2.0 [V] |
| **Hunyuan3D-2.1**, [tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) | 1 imagen | Malla + PBR [F] | 10 GB forma, 21 GB textura, 29 GB total [F]; 14,91 GB [V] | [NV]; la textura supera 16 GB según ese dato | Comunidad Tencent [V] |
| **Hunyuan3D-Omni**, [tencent/Hunyuan3D-Omni](https://huggingface.co/tencent/Hunyuan3D-Omni) | Imagen + control [NV] | Forma [NV] | 25,73 GB [V]; VRAM [NV] | [NV] | Comunidad Tencent [V, etiqueta] |
| **Step1X-3D**, [stepfun-ai/Step1X-3D](https://huggingface.co/stepfun-ai/Step1X-3D) rev. `bf7084495b` | 1 imagen | Malla + textura con SDXL [F] | 27-29 GB [F]; 19,64 GB [V] | [NV] | Apache 2.0 [V, etiqueta] |

Fuentes de VRAM: [Hunyuan 2.0](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), [Hunyuan 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1), [Step1X-3D](https://github.com/stepfun-ai/Step1X-3D). Los tamaños vienen de la lista de archivos de Hugging Face. El pico real de Pixal3D y TRELLIS.2 en esta PC no está medido.

### Malla cerrada y textura

| Modelo | Malla cerrada | Textura |
|---|---|---|
| Pixal3D / TRELLIS.2 | **No garantizada.** Microsoft avisa de agujeros y ofrece scripts de relleno [F] | PBR: color base, metal, rugosidad [F]. El nodo de ComfyUI acepta de 64 a 8192 px [V]; el workflow local usa 4096 [V] |
| Hunyuan3D 2.x | No se halló garantía en lo leído [NV] | 2.1: PBR [F]; su ejemplo usa 6 vistas a 512 [V]; tamaño final del atlas [NV] |
| Step1X-3D | TSDF "watertight" al generar [F]; después de reducir y texturizar [NV] | Textura por SDXL [F]; PBR y resolución [NV] |

Avisos sobre TRELLIS.2 crudo:
- Un usuario informa de 91,1% de mallas no cerradas y 14,9% no manifold en 101 modelos [C]: [issue 181](https://github.com/microsoft/TRELLIS.2/issues/181). El texto no dice si soldó las costuras UV; no soldarlas infla el dato [I].
- Otro usuario, con ROCm en gfx1151, informa de unas 220 000 aristas de borde antes de rellenar; la demo oficial dio 0 tras soldar [C]. Lo atribuye a umbrales numéricos: [issue 169](https://github.com/microsoft/TRELLIS.2/issues/169). Es otra tarjeta, pero conviene medirlo aquí.

El workflow local ya lleva `RemeshMesh` en modo `solid` a 768 y `DecimateMesh` a 700 000 caras [V]. ComfyUI describe `solid` como "one watertight solid from messy input" [V: es una descripción, no una prueba]. Queda por medir si sigue cerrada tras decimar y exportar. Las costuras UV duplican vértices, así que un validador que no suelde la verá abierta [I]. El propio nodo avisa de que `solid` redondea pliegues y une huecos pequeños [V]; podría pegar dedos u orejas [I]. [Código v0.45.0](https://github.com/Comfy-Org/ComfyUI/blob/v0.45.0/comfy_extras/nodes_mesh_postprocess.py).

### Evidencia de calidad

[Artículo de Pixal3D](https://arxiv.org/html/2605.10922v1). Todo es dato de los autores **[F]**:

| Método | IoU en Toys4K | LPIPS | Fidelidad, 30 personas (1-5) | Calidad (1-5) |
|---|---|---|---|---|
| TRELLIS (versión 1) | 79,48 | 0,204 | 1,86 | 1,99 |
| TripoSG | 73,54 | 0,250 | 2,25 | 2,14 |
| Hunyuan3D-2.1 | 83,33 | 0,179 | 2,77 | 2,50 |
| Direct3D-S2 | 74,23 | 0,268 | 3,21 | 3,64 |
| Pixal3D | 93,57 | 0,108 | 4,91 | 4,74 |

Límites de esta tabla:
1. Mide geometría, no textura.
2. Corresponde a la rama `paper`. La rama `main`, basada en TRELLIS.2, es otra versión [V, README].
3. No compara con TRELLIS.2 ni Step1X-3D.
4. No se vio ninguna prueba específica con chibis de perro.

Multivista, mismo artículo: F-Score 64,94 con 2 vistas y 67,85 con 4 [F]. Exige cámaras conocidas y precisas, y los bordes malos del recorte se convierten en defectos [F]. Vistas inventadas por un modelo de imagen pueden incumplirlo [I].

Los pesos `_mv` están en el repositorio oficial [V]. El nodo de ComfyUI acepta frente, izquierda, atrás y derecha, con elevación 0 y FOV 20 grados por defecto [V]. La ficha no explica los `_mv` [V].

## 3. Novedades hasta noviembre de 2026

| Modelo | Qué es | ¿Sustituye a Pixal3D? |
|---|---|---|
| [UltraTex](https://github.com/yiboz2001/UltraTex), 3-oct-2026 | Textura multivista 2K sobre una malla [F] | No; complemento. Ver apartado 4 |
| [Direct3D-V3](https://huggingface.co/wushuang98/Direct3D-V3), 19-nov-2026, 15,28 GB | Refina una malla inicial. Modelo inicial y textura sin publicar [V] | No; instalación para Linux y CUDA. Vigilar |
| [Hi3D v1.0](https://huggingface.co/hi3d-official/Hi3Dv1.0), 23-oct-2026, 7,74 GB | Refina geometría; salida sin textura [V] | No; Linux y CUDA, 1536 validado solo en 96 GB [F] |
| [Idol](https://huggingface.co/speridlabs/idol), 28-oct-2026 | Gaussian splats; 7,2 GB de VRAM [F] | No; no produce malla. Además usa DINOv3 y el VAE de FLUX.2-dev [V] |
| [LATO.2](https://huggingface.co/0x4c48/LATO.2), jul-2026, etiqueta MIT | Malla por vértices y conexiones [F] | No demostrado; textura y AMD [NV] |
| [Hunyuan3D-Buffalo 1.0](https://github.com/Tencent-Hunyuan/Hunyuan3D-Buffalo1.0), ago-2026 | Generación y edición [F] | No; el repositorio solo tiene README y recursos [V] |
| [UltraShape 1.0](https://github.com/PKU-YuanGroup/UltraShape-1.0) | Refina forma [F] | No; licencia de Hunyuan 2.1 [V] |
| [SAM 3D Objects](https://huggingface.co/facebook/sam-3d-objects), 13,17 GB | Objetos desde foto | No demostrado; licencia SAM, acceso restringido [V], AMD [NV] |
| [TripoSG](https://huggingface.co/VAST-AI/TripoSG), 7,95 GB | Forma | No; peor en la tabla de Pixal3D [F] |

No se encontraron pesos abiertos de Hunyuan 2.5 o 3.x. Se revisaron Tencent, las listas de imagen a 3D de Hugging Face y GitHub. La búsqueda no es exhaustiva y apenas cubre ModelScope.

ComfyUI v0.46.0 salió el 18-nov-2026 [V]. En `comfy_extras`, `ldm/trellis2` y `ldm/flux` no cambió ningún archivo de TRELLIS.2, malla o UltraTex respecto a v0.45.0 [V, solo esas carpetas]. No hay motivo visto ahí para actualizar.

## 4. Albedo, sombras y textura

| Opción | Licencia | Evaluación |
|---|---|---|
| Color base de Pixal3D | MIT [V] | Primera opción. Que salga sin sombras: [NV] |
| Imagen de entrada con color plano y luz uniforme | La del modelo de imagen | Lo más barato. Probar antes que otro modelo [I] |
| `RenderMesh` + editar vistas + `BakeTextureFromViews`, en v0.45.0 [V] | GPL-3 (ComfyUI) + editor; [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) es Apache 2.0 [V] | Buena vía comercial para corregir zonas. Coherencia entre vistas [NV]; hay que construirla |
| [Marigold IID appearance v1.1](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1) / [lighting v1.1](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1) | OpenRAIL++ [V, etiqueta]; texto completo [NV] | Extrae albedo. *Appearance* se entrenó con InteriorVerse (interiores) [F]; chibis y AMD [NV] |
| [Lotus-2](https://huggingface.co/jingheya/Lotus-2) | Apache 2.0 [V, etiqueta]; base [NV] | Direct3D-V3 lo cita para normales; albedo no confirmado |
| Delight de Hunyuan 2.0 | Comunidad Tencent [V] | Evitar para vender |
| Hunyuan Paint 2.1 | Comunidad Tencent [V] | Evitar: territorio y 21 GB [F] |
| [UltraTex](https://huggingface.co/Comfy-Org/UltraTex) | FLUX no comercial [V, etiqueta] | Solo prueba personal |
| [UltraTex con Qwen-Image 2.1](https://huggingface.co/orwelian84/UltraTex-Qwen-Image-2.1) | LoRA MIT [V, etiqueta]; [base solo no comercial](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) [V] | No resuelve la venta |
| [Mesh Encoder AMD](https://github.com/BoomerCyb/ComfyUI-Trellis2-Mesh-Encoder-AMD) | MIT [V] | Texturizaría un GLB con TRELLIS.2; validado solo en 9070 XT [C]. El núcleo v0.45.0 no trae ese codificador [V] |

**UltraTex.** Código MIT [V]; licencia propia de sus pesos en ModelScope [NV]. La [guía de ComfyUI](https://docs.comfy.org/tutorials/3d/ultratex) declara 13,16 GB de descarga en int8 o 21,87 GB en precisión completa, vistas a 1024 y mapas a 4096; no da VRAM [F]. Pide v0.46.0, aunque `UltraTexConditioning` y `BlockSparseAttention` ya existen en v0.45.0 [V]; que la plantilla funcione ahí: [NV]. Necesita malla con UV [F]. Pinta albedo, pero no se encontró una prueba de que elimine sombras de la referencia. No se encontró prueba en AMD. En esta PC los int8 de Pixal3D ya dieron problemas [L].

**Contradicción de FLUX 9B.** La [licencia no comercial de BFL](https://raw.githubusercontent.com/black-forest-labs/flux2/main/model_licenses/LICENSE-FLUX-NON-COMMERICAL) limita el uso del modelo a fines sin pago directo ni indirecto. También permite usar las salidas comercialmente, salvo lo prohibido [V]. No se pudo abrir la licencia en la página del 9B, que tiene acceso restringido. Para vender figuras, tratarlo como no apto salvo licencia de BFL [I].

## 5. Licencias por uso

"Sí*" significa permitido con condiciones. (c) supone que el programa usa la pieza; si la incluye, también aplica (d).

| Componente | (a) Personal | (b) Vender figuras | (c) Vender programa | (d) Redistribuir pesos o código |
|---|---|---|---|---|
| **Pixal3D**, código y pesos: [MIT](https://huggingface.co/TencentARC/Pixal3D/blob/main/LICENSE) [V] | Sí | Sí | Sí* | Sí*, con LICENSE y [NOTICE](https://huggingface.co/TencentARC/Pixal3D/blob/main/NOTICE) |
| **TRELLIS.2**, [código](https://github.com/microsoft/TRELLIS.2) y pesos: MIT [V] | Sí | Sí | Sí* | Sí* |
| **O-Voxel** (en TRELLIS.2) y [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM): MIT [V] | Sí | Sí | Sí* | Sí* |
| **DINOv3**: [licencia de Meta](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) [V] | Sí* | Sí*: no prohíbe el comercio | Sí* | Sí*, bajo la misma licencia y con copia |
| **MoGe**: [código MIT y partes Apache](https://github.com/microsoft/MoGe/blob/main/LICENSE) [V]; [pesos MIT](https://huggingface.co/Comfy-Org/MoGe) [V, etiqueta] | Sí | Sí | Sí* | Sí* |
| **NAF**: [código Apache 2.0](https://github.com/valeoai/NAF) [V]; pesos [NV] | Sí | Sí | Sí* | Código Sí*; pesos por confirmar |
| **BiRefNet**: [MIT](https://huggingface.co/ZhengPeng7/BiRefNet) [V, etiqueta] | Sí | Sí | Sí* | Sí* |
| **RMBG-2.0**: CC BY-NC [V, etiqueta]; no hace falta | Solo no comercial | No sin otra licencia | No sin otra licencia | Solo no comercial |
| **CuMesh** ([MIT](https://github.com/visualbruno/CuMesh)), [xatlas](https://github.com/jpcy/xatlas) (MIT) [V] | Sí | Sí | Sí* | Sí* |
| **cuBQL**: [Apache 2.0](https://github.com/NVIDIA/cuBQL) [V] | Sí | Sí | Sí* | Sí* |
| **Eigen** en esas piezas: [MPL-2.0 según el kit](https://github.com/tsubasaprogame/Pixal3D-TRELLIS.2-AMD-ROCm-WINDOWS/blob/main/THIRD_PARTY_NOTICES.md) [C] | Sí | Sí | Sí* | Sí*, con el código fuente de esos archivos |
| **Kit AMD**: MIT [V] | Sí | Sí | Sí* | Sí* |
| **nvdiffrast**: [licencia NVIDIA](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt) [V] | Solo investigación o evaluación | No | No | Solo con esa licencia y límite |
| **nvdiffrec**: [licencia NVIDIA](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt) [V, cláusula de uso] | Solo investigación o evaluación | No | No | Restringido; revisar entera |
| **ComfyUI**: [GPL-3](https://github.com/Comfy-Org/ComfyUI/blob/v0.45.0/LICENSE) [V] | Sí | Sí | Sí*; proceso aparte por HTTP reduce riesgo [I] | Cumplir GPL, con fuente y parches |
| **Hunyuan 2.0 y 2.1**, [licencia 2.1](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE) [V] | Sí* fuera de UE, Reino Unido y Corea del Sur | Sí* solo en territorio permitido | Sí* ahí; licencia aparte si había más de 1 M de usuarios/mes al lanzar el modelo | Sí* ahí, con licencia y aviso |
| **Hunyuan Omni** [V, etiqueta] | Por leer | Por leer | Por leer | Por leer |
| **Step1X-3D**: Apache 2.0 [V, etiqueta]; base SDXL [NV]; su horneador reutiliza piezas de Hunyuan 2.0 [V] | Sí, con dependencias por revisar | Por revisar dependencias | Por revisar dependencias | No como paquete hasta revisar |
| **Hi3D 1.0**: [licencia propia](https://huggingface.co/hi3d-official/Hi3Dv1.0/blob/main/LICENSE) [V, parcial] | Sí* | Sí*: la ficha exime a las salidas sueltas de marca | Sí* con "Powered by Hi3D 1.0"; licencia aparte desde 200 000 usuarios o 1 M USD al mes | Sí*, con avisos |
| **Direct3D-V3**: código [MIT](https://github.com/DreamTechAI/Direct3D-V3) [V]; pesos MIT [V, etiqueta] | Sí | Sí, si la malla inicial lo permite | Igual | Sí*; DINOv3 aparte |
| **UltraTex + FLUX 9B** | Sí, sin ingresos | No recomendado | No sin licencia de BFL | Solo no comercial; avisos por revisar |
| **SAM 3D Objects**: [licencia SAM](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE) | Por leer completa | Por leer | Por leer | Por leer |

Puntos importantes:

1. **Pixal3D y la UE.** La ficha lleva `extra_gated_eu_disallowed: true`, pero el modelo no está restringido y la LICENSE es MIT sin territorio [V]. La pregunta sobre esto sigue sin respuesta en el [issue 33](https://github.com/TencentARC/Pixal3D/issues/33). Guardar LICENSE y NOTICE de la revisión usada.
2. **DINOv3 no es MIT.** El paquete Comfy está etiquetado MIT y contiene `dino_v3_L_naf_fp32` [V]. La etiqueta no cambia la licencia de Meta [I]. Prohíbe usos militares y armas, e ingeniería inversa; Meta puede modificarla [V]. No se vio tope de usuarios.
3. **nvdiffrast no aparece en la ruta local.** En el código Python de ComfyUI v0.45.0 no hay importaciones de nvdiffrast, o_voxel, flex_gemm, cumesh, cuBQL ni xatlas [V, búsqueda]. El remallado se declara aproximación de CuMesh en PyTorch [V]. Por eso compilar cumesh podría sobrar para estos workflows [I]; probarlo en copia. ComfyUI sí depende del paquete externo `comfy-kitchen`, cuya licencia no se revisó [NV].
4. **`amd/trellis2/nvdiffrast_torch.py`** se presenta como sustituto en PyTorch [L]. Su procedencia línea a línea no se auditó [NV].
5. **Hunyuan.** Tencent no reclama las salidas, pero la sección 5c prohíbe usarlas o mostrarlas fuera del territorio [V]. Es un riesgo para vender por internet.
6. Estas licencias no dan derechos sobre personajes o fotos ajenas.

## 6. AMD y Windows

| Evidencia | Alcance |
|---|---|
| Guías locales | 9060 XT, dos figuras correctas, 20-40 min [L]. Pico sin registrar |
| [Kit base](https://github.com/tsubasaprogame/Pixal3D-TRELLIS.2-AMD-ROCm-WINDOWS) `a4b7b16` | 9070 XT; 499 s para un ejemplo a 1024 con atlas 2048 [C]. No comparable con 1536/4096 |
| [ComfyUI-Trellis2-AMD](https://github.com/dmonkman/ComfyUI-Trellis2-AMD) | Incluye gfx1200 en sus binarios, pero solo probó 6800 XT, con ROCm 10.0 [C] |
| [Issue 16124 de ComfyUI](https://github.com/Comfy-Org/ComfyUI/issues/16124) | `HIPBLAS_STATUS_ALLOC_FAILED` al desenrollar en 7900 XTX: coincide con el fallo local [C] |
| [Issue 169](https://github.com/microsoft/TRELLIS.2/issues/169) | `cudaMemcpy2D` de CuMesh falla por encima de ~1 M de filas en HIP [C]; es distinto del fallo local de 2^19 |

No se encontró prueba pública de Pixal3D en gfx1200 ni reconocimiento oficial del fallo de 2^19. Que ROCm 10 lo corrija: [NV]. No tocar la instalación buena.

Los modelos bf16 ocupan unos 25 GB [L]. Con ~170 GB libres [L] cabe también una prueba de UltraTex. No hacen falta los 46 GB originales de Pixal3D para ComfyUI [I].

## 7. Manejar ComfyUI por HTTP local

Rutas en [server.py v0.45.0](https://github.com/Comfy-Org/ComfyUI/blob/v0.45.0/server.py) [V] y en la [guía](https://docs.comfy.org/development/comfyui-server/comms_routes):

1. `GET /system_stats` y `GET /object_info`: comprobar GPU, nodos y modelos.
2. `POST /upload/image`: subir imagen.
3. `POST /prompt` con `{"prompt": grafo_API, "client_id": "..."}`. Devuelve `prompt_id`, o 400 con `error` y `node_errors`.
4. `/ws`: progreso. `GET /history/{prompt_id}`: resultado. `GET /view`: descarga; también vale leer `output`.
5. `POST /interrupt`, `/queue`, `/free`: cancelar, limpiar cola y descargar modelos.

Escucha en `127.0.0.1` por defecto [V]. No usar `--listen`. Los ejemplos oficiales están en [script_examples](https://github.com/Comfy-Org/ComfyUI/tree/v0.45.0/script_examples) [V].

**De interfaz a API.** El formato API es `{id: {class_type, inputs}}`. La [guía oficial](https://docs.comfy.org/development/api-development/workflow-api-format) indica **File > Export Workflow (API)** [V]. `server.py` no ofrece conversión [V].

Los tres workflows locales tienen 66, 68 y 68 nodos, sin subgrafos ni nodos silenciados [V]. Hay tres trampas:
- `KSampler` guarda `fixed` como un valor extra después de la semilla [V].
- `RemeshMesh` usa un combo dinámico. Sus valores no salen como entradas normales del JSON [V]; en la API se espera la forma `sign_mode` + `sign_mode.fill` [I, a partir del código; confirmar con la exportación].
- `Note` y `MarkdownNote` no son trabajo ejecutable [I].

**Recomendado:** exportar una vez desde la interfaz, guardar cada archivo API con su original y su hash, y cambiar solo campos conocidos. Antes de enviar, comparar con `/object_info`. Las vistas previas se pueden quitar después de comprobar que el GLB no cambia [I].

Si hace falta convertir sin navegador: [`/workflow/convert`](https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint), Unlicense, último cambio 16-mar-2026. Dice admitir subgrafos y combos dinámicos V3 [F]; en v0.45.0 [NV]. [comfyui-autoflow](https://libraries.io/pypi/comfyui-autoflow) dice convertir sin servidor [C]; no revisado.

## 8. Recomendación y alternativa

**Pixal3D bf16**, 1536, remallado `solid` 768, 700 000 caras, textura 4096, una vista con luz plana, ComfyUI v0.45.0 congelado. Motivos: ya funciona aquí [L], cadena MIT más la licencia de DINOv3 [V], y mejor geometría en la comparación de sus autores [F]. Que cada usuario descargue los pesos evita redistribuirlos, pero no quita las condiciones de uso.

**Alternativa: TRELLIS.2-4B**, con `Switch to Trellis2 = true`. Usarla si Pixal3D falla en una figura o su licencia se aclara en contra.

**Revisar la decisión** si Direct3D-V3 publica malla inicial y textura y funciona en AMD, o si aparece un texturizador multivista con base abierta y prueba en 16 GB.

## 9. Qué falta construir

1. Cliente HTTP con manifiesto: semilla, versiones, hashes, parámetros, reanudación.
2. Prueba de arranque que aborte si falta el parche de 2^19 filas.
3. Validador: soldar una copia por posición; contar bordes, no manifold, piezas, autointersecciones y pared mínima. Guardar aparte el GLB con UV.
4. Detector de NaN, magenta y franjas; comparación con la referencia desde la misma cámara.
5. Paso opcional de albedo y edición de vistas.
6. Archivos API y prueba de compatibilidad.
7. Registro de licencias, con copias.

## 10. Prueba concreta de GPU

No ejecutada. Respaldo previo, red bloqueada y una tarea a la vez.

**A. Salud.** Ejecutar `prueba_linear_grande.py` y arrancar con `--fp32-vae`. Si falta el aviso del parche, abortar.

**B. API.** Exportar los tres grafos. Ejecutar la misma imagen y semilla por interfaz y por HTTP. Pasa si ambos GLB son válidos, tienen textura 4096 y difieren menos de 1% en caras. Ese margen es una propuesta.

**C. Comparación.** Cinco imágenes: dos chibis de perro, un humanoide, una figura con partes delgadas y un objeto multicolor. Para cada una:
1. Pixal3D, 1 vista.
2. TRELLIS.2, 1 vista.
3. Pixal3D, 4 vistas, si hay vistas coherentes.

Misma semilla y ajustes: 15 ejecuciones, unas 5-10 horas con los tiempos de la guía [I]. Registrar cada segundo VRAM dedicada y compartida, RAM y paginación. Contar bordes, no manifold y piezas antes y después de `RemeshMesh`, tras `DecimateMesh` y en el GLB soldado.

**D. Umbrales propuestos, por confirmar.**
- 5 de 5 sin cierre, NaN ni magenta.
- 0 bordes abiertos y 0 no manifold en el sólido soldado; piezas solo las esperadas.
- Si se agota la memoria: bajar a 1024 antes de cambiar de modelo.
- Multivista o TRELLIS.2 ganan solo si se prefieren a ciegas en 4 de 5 y no pierden caras delgadas.

**E. Sombras.** Misma figura con luz plana y con sombra marcada. Comparar el color base sin iluminación. Solo si falla, probar Marigold IID en una copia.

**F. UltraTex, opcional y personal.** Otra carpeta con v0.46, malla de C con UV. Medir pico, tiempo y calidad a ciegas frente a Pixal3D 4096. No vender ese resultado.

**G. cumesh.** En una copia del entorno, quitar cumesh y generar. Si funciona, ese paso sobra para esta ruta. Conservar la rueda.

## 11. Dudas que quedan

- Pico de VRAM y RAM, malla final cerrada y sombras en esta PC.
- Entrenamiento y calidad de los pesos `_mv`.
- Contradicción de la etiqueta UE de Pixal3D.
- Licencias de UltraTex en ModelScope, pesos NAF, SDXL, OpenRAIL++, SAM y Hunyuan Omni.
- Causa oficial del fallo de 2^19.
