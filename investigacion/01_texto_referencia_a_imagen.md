# 01. Texto y foto de referencia a imagen estilizada, y vistas de atrás y lados

Fecha de consulta: 2026-11-25. No se descargó ni se ejecutó ningún modelo.

**Etiquetas**
- **[V] Verificado:** leído en la fuente indicada (ficha, LICENSE, API de Hugging Face, documentación o código). No significa probado en nuestra GPU.
- **[F] Afirmación del fabricante o autor del modelo:** lo dice quien lo publica; nadie independiente lo confirma aquí.
- **[T] Tercero:** artículo, issue o prueba de un usuario.
- **[N] No verificado.** "Inferencia" = deducción nuestra.

Los tamaños en disco son decimales (1 GB = 10^9 bytes) y salen de la API de Hugging Face [V]. **Tamaño en disco no es pico de VRAM.** La lectura de licencias no es asesoría legal.

## 1. Respuesta corta

- **Recomendado: Qwen-Image-Edit-2511** (Apache 2.0), en GGUF Q4_K_M, con el LoRA de ángulos de fal y, según la prueba, el LoRA de figuras y el de quitar luz. Es el único conjunto encontrado cuyas piezas principales (modelo, codificador, figura, quitar luz, ángulos) figuran como Apache 2.0. Su punto débil es la memoria: no se encontró una medición en RX 9060 XT.
- **Reserva: FLUX.2 [klein] 4B** (Apache 2.0). Es pequeño y un usuario lo hizo funcionar en una RX 9060 XT de 16 GB con 32 GB de RAM [T]. Su punto débil es la fidelidad al sujeto y las vistas traseras.
- **Comparación obligada, no base: FLUX 3 Image Klein** (octubre de 2026). Más nuevo y probablemente mejor, y su licencia permite vender figuras a quien facture menos de 5 M USD al año, con condiciones. No permite meterlo en un programa en venta sin otra licencia.
- **Descartados para vender:** Qwen-Image-2.1 (solo investigación), FLUX.2 klein 9B (pesos no comerciales) y MiniMax-H3-Image (territorio excluido y tamaño).
- **No usar `int8_convrot`** en esta máquina hasta probarlo (sección 5).
- **Ninguna prueba encontrada mide "foto de un perro a chibi con sus manchas".** Esa respuesta solo sale de nuestra prueba (sección 9).

## 2. Comparación

### 2.1 Versión, tamaño y soporte

| Modelo | Versión y fuente | Qué acepta | Disco | VRAM publicada | ComfyUI v0.45.0 |
|---|---|---|---|---|---|
| **Qwen-Image-Edit-2511** | `6f3ccc0b56`, 2025-12-23. [Ficha](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) [V] | Texto + varias imágenes [V]; hasta 3 en ComfyUI [[T]](https://www.ainvfx.com/blog/open-image-models-in-comfyui-flux-3-image-klein-moss-image-and-more/) | [bf16 40.86; fp8mixed 20.53](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI); [GGUF Q4_K_M 12.84, Q3_K_M 9.92](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF); [codificador fp8 9.38 + VAE 0.25](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) [V] | Oficial: no encontrada. Q4_K_M cupo en 16 GB de una **NVIDIA** RTX 5060 Ti, unos 30 s por imagen [[T]](https://note.com/mayu_hiraizumi/n/n89a343aa1f49?hl=en) (leído solo en extracto) | Archivos oficiales de Comfy-Org [V]. GGUF necesita [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (código Apache 2.0, último cambio 2026-01-12 [V]; con v0.45.0 [N]) |
| Qwen-Image-Edit-2509 | `d3968ef930`, 2025-09-22. [Ficha](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) [V] | Igual; 2511 dice reducir la deriva y mejorar consistencia [F] | fp8 20.43 [V] | No encontrada | Sí, archivos oficiales [V] |
| **Qwen-Image-2.1** (7B) | `d26bb61231`, 2026-09-30. [Ficha](https://huggingface.co/Qwen/Qwen-Image-2.1) [V] | Hasta 10 referencias, salida RGBA [F] | [bf16 14.23; int8 7.26; codificador bf16 17.53, int8 9.35, w4a8 6.31; VAE 0.68](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) [V] | No encontrada | Desde v0.37.0 ([notas](https://github.com/Comfy-Org/ComfyUI/releases)) [V] |
| **FLUX.2 klein 4B** | `e7b7dc27f9`, 2026-02-24. [Ficha](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B); base: [`a3b4f48491`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) [V] | Texto + varias referencias [F] | [bf16 7.75; codificador 8.04 (fp4 3.85); VAE 0.34](https://huggingface.co/Comfy-Org/flux2-klein-4B); [fp8 4.07](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) [V] | ~13 GB según la ficha [F]; 8.4 GB (destilado) y 9.2 GB (base) en RTX 5090 según [Comfy](https://docs.comfy.org/tutorials/flux/flux-2-klein) [F]. Las cifras no coinciden | Sí, plantillas oficiales [V] |
| FLUX.2 klein 9B | `1498bcb8aa`, 2026-11-19. [Ficha, con registro](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) [V] | Igual que 4B | 18.16; [fp8 9.43](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8); [codificador fp8 8.66](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b) [V] | No encontrada | Sí [V] |
| **FLUX 3 Image Klein** (5.6B) | `3b767ff9b5`, 2026-11-19; publicado 2026-10-15. [Repo con registro](https://huggingface.co/black-forest-labs/flux-3-image-klein), [código](https://github.com/black-forest-labs/flux-3-image-klein) [V] | Texto + hasta 10 referencias + cajas [F] | [bf16 11.10; int8 5.83; codificador bf16 8.88, int8 4.86; VAE 0.34; reescritor opcional 6.72 a 18.82](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) [V] | 31 GB bf16 y 26 GB fp8 con el código de referencia en B200, sin reescritor [F]; "8 GB" solo con int8 en ComfyUI [F] | Desde v0.41.0; corrección en 0.41.1 ([docs](https://docs.comfy.org/tutorials/flux/flux-3-klein)) [V] |
| MOSS-Image + LoRA de edición | [Comfy-Org `2929efe3`](https://huggingface.co/Comfy-Org/MOSS-Image), 2026-11-04; edición de [chinoll](https://huggingface.co/chinoll/Moss-Image-Edit) [V] | Edición por LoRA; varias referencias según [docs](https://docs.comfy.org/tutorials/image/moss-image/moss-image) [V]. La misma página es ambigua sobre por dónde entra la referencia | bf16 10.16; int8 5.41; codificador bf16 19.31, int8 9.75, w4a8 7.34 [V] | No encontrada | Edición desde v0.43.0 [V] |
| MiniMax-H3-Image | [`8ced600686`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), 2026-11-16 [V] | Texto + una o más referencias [F] | [bf16 40.23, w4a8 12.55; codificador 15.69 a 27.14](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) [V] | No encontrada | [N] |

Otros con licencia declarada Apache 2.0 [V], no estudiados a fondo: [LongCat-Image-Edit-Turbo](https://huggingface.co/meituan-longcat/LongCat-Image-Edit-Turbo) (2026-02; ~18 GB con descarga a CPU [F]), [FireRed-Image-Edit-1.1](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1) (2026-03; derivado del tamaño de Qwen), [JoyAI-Image-Edit](https://huggingface.co/jdopensource/JoyAI-Image-Edit). [Fibo-2.0-edit-turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) declara CC BY-NC 4.0 [V]: descartado.

**RAM:** ningún fabricante publica un mínimo con estos archivos. Suma solo de pesos: Qwen Q4_K_M + codificador fp8 + VAE = **22.5 GB**; klein 4B bf16 completo = **16.1 GB**; FLUX 3 bf16 con codificador bf16 = **20.3 GB** (aritmética sobre datos [V]). Con 32 GB y Windows, Qwen queda justo (inferencia). Hay que cargar por turnos y no tener Pixal3D en memoria.

### 2.2 Licencias: lo que permite cada una

| Modelo | Licencia de pesos | (a) Uso personal | (b) Vender figuras hechas con las imágenes | (c) Vender el programa | (d) Redistribuir pesos |
|---|---|---|---|---|---|
| Qwen-Image-Edit-2511 / 2509 | Apache 2.0 declarada en ficha [V]; el repo de 2511 no trae archivo LICENSE [V] | Sí | Sí: Apache no lo restringe | Sí | Sí, con copia de licencia y avisos |
| FLUX.2 klein 4B y base 4B | Apache 2.0, [LICENSE.md](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/LICENSE.md) [V] | Sí | Sí; la ficha dice uso comercial [V] | Sí; no usar la marca FLUX como propia | Sí, con licencia y avisos |
| FLUX.2 klein 9B (también 9B KV y base 9B) | [FLUX Non-Commercial v2.1](https://github.com/black-forest-labs/flux2/blob/main/model_licenses/LICENSE-FLUX-NON-COMMERICAL) [V] en GitHub; el LICENSE actual del repo con registro [N] | Sí | **Zona gris.** 2.d permite cualquier uso de la salida, incluso comercial; 1.c y 2.b no permiten usar el modelo en una actividad que genera ingresos. No basar el negocio en esto | No sin licencia de BFL | Solo para fines no comerciales, con avisos |
| FLUX 3 Image Klein | FLUX Kommunity v1.0. Original con registro; texto leído en dos copias idénticas ([1](https://huggingface.co/ml-resources/flux-3-image-klein/blob/main/LICENSE.md), [2](https://huggingface.co/unsloth/flux-3-image-klein-GGUF/blob/main/LICENSE.md)) [V en copias]. Original: [enlace](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md) | Sí | Sí si tú y tus afiliados facturan menos de 5 M USD al año (1.e, 2.a), con revisión de cada salida o filtro, y aviso de IA cuando la ley lo exija (2.e) | **No concedido**: lo comercial se limita a generar salidas; hace falta licencia de BFL. Que cada comprador baje los pesos por su cuenta: zona gris, preguntar a BFL | Solo dentro de los fines permitidos y con aviso de atribución; no para un paquete comercial (inferencia de 2.a y 3) |
| Qwen-Image-2.1 | [Qwen Research, 2026-09-20](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) [V] | Solo "investigación o evaluación"; un hobby no está claramente cubierto | **No** sin licencia comercial aparte (2.b) | No | Solo no comercial, con avisos |
| MOSS-Image | Apache 2.0 ([SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)) [V]. El LoRA original de chinoll **no declara licencia** en su ficha; Comfy-Org etiqueta el paquete como Apache [V] | Sí | Sí la base; LoRA: aclarar | Igual | Base sí; LoRA: aclarar |
| MiniMax-H3-Image | [Licencia comunitaria, 2026-11-10](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE) [V] | Excluye UE, Reino Unido, Corea del Sur y EE. UU., **también para las salidas** (V.4) | Fuera de esos territorios, hasta 1 M USD | Con aviso visible y salvaguardas (IV, V.5) | Solo en territorio permitido |

Matices que aplican a todo:
- **Codificadores:** Qwen2.5-VL-7B, Qwen3-4B, Qwen3-8B y Qwen3-VL-4B declaran Apache 2.0 [V] ([1](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [2](https://huggingface.co/Qwen/Qwen3-4B), [3](https://huggingface.co/Qwen/Qwen3-8B), [4](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)). El VAE de klein 4B viene en el repo Apache [V].
- Varios artículos dicen que FLUX 3 Klein es Apache 2.0. **Es incorrecto para los pesos:** Apache cubre el código y el reescritor de texto [V, [README](https://github.com/black-forest-labs/flux-3-image-klein)].
- El código de ComfyUI tiene su propia licencia; si se empaqueta, se revisa aparte [N].
- Ninguna licencia de modelo da derechos sobre personajes o fotos de terceros. Apache tampoco da garantía sobre los datos de entrenamiento.
- FLUX NC y Kommunity prohíben usar salidas para mejorar modelos parecidos. Si más adelante se entrena un LoRA, los ejemplos no deben venir de ellos [V].

## 3. Conservación de la identidad

| Modelo | Evidencia |
|---|---|
| Qwen-Image-Edit-2511 | Alibaba dice mejor consistencia de personaje y menos deriva que 2509 [F]. [Comparación de un autor](https://dev.to/ai-ksk/flux-2-klein-4b9b-vs-qwen-2511-a-single-and-multi-image-editing-stress-test-14dg) con 13 ediciones: la menor pérdida de parecido frente a klein 4B y 9B [T]. Muestra pequeña, personas y sin flujo publicado |
| FLUX.2 klein 4B / 9B | La misma prueba ve deriva de rostro, sobre todo en 4B base [T]. Otra prueba en RX 9060 XT: el destilado hace bien cambios puntuales, pero falla al **reconstruir el cuerpo entero** y al seguir estilos por texto; sale mejor en dos pasos [[T]](https://note.com/nifty_dunlin2337/n/n83089942c39e?hl=en). [Virse](https://www.virse.ai/blog/qwen-image-edit-vs-flux-klein-commercial-image-editing) favorece a klein en identidad humana (leído solo en extracto) [T]: evidencia contradictoria |
| FLUX 3 Image Klein | El fabricante incluye un ejemplo de [imagen a figura de mesa](https://github.com/black-forest-labs/flux-3-image-klein/blob/main/examples/distilled/1_figurine.json), de un robot, no de una mascota [V que existe]. La ficha (leída en copia) admite que las referencias "no bloquean la estructura" y que los textos cortos dan salidas genéricas [F]. Un comentario reporta ojos deformes [[T]](https://blog.comfy.org/p/flux-3-image-klein-in-comfyui-56b) |
| Qwen-Image-2.1 | Alibaba: conserva identidad de personas y productos [F] |
| Perros y manchas | **Sin prueba independiente encontrada en ningún modelo** |

## 4. LoRA de chibi, luz y ángulos

### Estilo chibi o figura

| LoRA | Base | Licencia declarada | Observación |
|---|---|---|---|
| [Trendy_figurines](https://huggingface.co/starsfriday/Qwen-Image-Edit-2511-Trendy_figurines) `f85d9cbf9c`, 2026-05-15, 1.18 GB | **2511** | Apache 2.0 [V] | Estilo caja sorpresa, proporción Q. Su texto pide PVC brillante y render: **va contra el color plano**. No menciona mascotas en lo leído |
| [Qwen-Edit-3DChibi](https://huggingface.co/rsshekhawat/Qwen-Edit-3DChibi-LoRA), 2025-11-09 | **2509** | Apache 2.0 [V]; datos OmniConsistency declarados MIT [V] | Uso con 2511: [N] |
| [Kontext 3D_Chibi](https://huggingface.co/Kontext-Style/3D_Chibi_lora) | FLUX.1 Kontext dev | Sin licencia en la ficha [V]; base no comercial [N] | Datos hechos con GPT-4o [F]. Descartado |
| [Nano-Banana-Figure](https://huggingface.co/W2GenAI/Nano-Banana-Figure-LoRA) | Qwen-Image-Edit | [N] | 100 pares; dice incluir mascotas (extracto) [F] |
| klein 4B chibi genérico | - | - | Solo se halló `Mason02/klein4b-roblox-chibi-lora`, sin descargas [V] y de otro estilo |

Probar **primero sin LoRA de estilo**: puede borrar las manchas. BFL tiene una [guía de LoRA para klein](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora) con fotos de mascota a otro estilo (leída solo en extracto). Entrenar en esta AMD: [N].

### Color plano (quitar sombras)

- [QIE-2511-Studio-DeLight](https://huggingface.co/prithivMLmods/QIE-2511-Studio-DeLight): Apache 2.0 [V], para 2509 y 2511 [F]. Iguala la luz con difusa; el propio autor habla de sombreado suave, así que **no produce un albedo medido**.
- [Flux2-Klein-Delight](https://huggingface.co/linoyts/Flux2-Klein-Delight-LoRA): base **9B**, licencia "other" [V]. No sirve para 4B.
- [Spellcast3D Albedo](https://huggingface.co/IgorAherne/spellcast3d-marigold-albedo-qwen-image-2.1): el más cercano a albedo real, pero Qwen Research [V]: solo evaluar.

### Vistas de atrás y lados

| Opción | Base y licencia | Evidencia y límites |
|---|---|---|
| [fal Multiple-Angles 2511](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) `e3066224ab`, 0.30 GB | 2511, Apache 2.0 [V] | 96 poses: 8 rumbos, 4 alturas, 3 distancias; más de 3000 pares de splats gaussianos [F]. Incluye flujo de ComfyUI [V]. Cada vista se genera por separado: **nada obliga a que coincidan** (inferencia) |
| [dx8152 Multiple-angles](https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles) | 2509, Apache 2.0 [V] | Copia oficial en Comfy-Org [V] |
| [Character turnaround sheet](https://huggingface.co/tarn59/character_turnaround_sheet_qwen_edit_2511) | 2511, Apache 2.0 [V] | Hoja en una sola imagen. El autor avisa que sale **demasiado estrecha** y recomienda ensanchar 20 % [F]: malo para geometría |
| [fal spritesheet](https://huggingface.co/fal/flux-2-klein-4b-spritesheet-lora), 0.08 GB | klein 4B, Apache 2.0 [V] | Hoja 2x2; ejemplos de vehículos, no de animales; no garantiza frente/lado/atrás exactos |
| [CharacterSheet Triple/QuadView](https://huggingface.co/Alissonerdx/CharacterSheet) | klein 9B o Krea 2; licencia de Civitai [V] | Unos 300 ejemplos, sobre todo humanos; animales menos consistentes [F] |
| [FLUX 3 angles](https://huggingface.co/akhaliq/flux-3-image-klein-angles-lora) | FLUX 3 base; Kommunity [V] | Demo con perro. El autor dice que personas y animales giran de forma fiable, pero entrenó a 512 px con renders sintéticos y puede inventar sujetos [F] |
| [Qwen-2.1 angles v2](https://huggingface.co/akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA) | LoRA Apache; **base de investigación** [V] | La mejor documentación con perro: orbita 12 rumbos; dice que el pelo fino se ablanda [F]. No vale para vender |
| [AnyAngle](https://huggingface.co/lilylilith/QI_2.1_AnyAngle) | Igual [V] | **Idea útil:** generar un 3D burdo (nombra Pixal3D), renderizarlo desde el ángulo y pasarlo como guía [F] |

**Método propuesto para las vistas:** (1) aprobar el frente; (2) Pixal3D de una vista, calidad baja; (3) renderizar atrás y lados; (4) editar cada render con dos referencias, "render = pose y silueta" y "frente = color"; (5) aceptar solo si la silueta coincide; (6) Pixal3D multivista. Así la geometría es común. Que Qwen 2511 siga el render sin el LoRA AnyAngle: [N].

MV-Adapter y Era3D aparecieron en la búsqueda (de 2024, 768 px el primero, con vista a 180 grados en su entrenamiento). Su licencia y calidad en perros: [N].

**Foto de la espalda real:** si la mascota tiene manchas atrás, ningún modelo las conoce. Hay que pedir fotos de frente, ambos lados y lomo; si faltan, avisar que son inventadas.

## 5. Evidencia y riesgos en AMD

| Dato | Estado |
|---|---|
| klein 4B en **RX 9060 XT 16 GB + 32 GB RAM**: destilado ~20 s primera vez y ~7 s después; base ~6 min y ~1 min; varias referencias con base, casi 3 min y usando RAM. "Apenas posible" | [T, leído](https://note.com/nifty_dunlin2337/n/n83089942c39e?hl=en). No dice sistema, ROCm ni resolución |
| klein 4B y 9B con **PyTorch 2.9.1+rocm7.2.1** en RX 9070 XT (gfx1201, Windows) | [T, README leído](https://github.com/Justin-Ju-0413/comfyui-amd-rocm-kit). Sin tiempos ni VRAM; no menciona Qwen Edit |
| Qwen-Image-Edit-2511 en 9060 XT o 9070 XT | **Ninguna medición encontrada** |
| FLUX 3 Klein en AMD | **Ninguna medición encontrada** |
| Imagen negra con `int8_convrot` de difusión en gfx1201, **misma PyTorch 2.9.1+rocm7.2.1**, Windows 11; fp8 funcionó | [Issue #15084](https://github.com/Comfy-Org/ComfyUI/issues/15084), cerrado sin arreglo claro; otro usuario seguía igual en septiembre [V el reporte] |
| Triton apagado: convrot muy lento; Triton global: cierre del proceso | [comfy-kitchen #78](https://github.com/Comfy-Org/comfy-kitchen/issues/78), Linux [T] |
| Cierre 0xC0000005 en matmul fp32 por lotes; bf16 bien | [comfy-kitchen #357](https://github.com/Comfy-Org/comfy-kitchen/issues/357), gfx1201 con ROCm 7.13/7.14, abierto [T]. En 7.2.1: [N] |
| VRAM dinámica produce ruido en gfx1201 | [#16337](https://github.com/Comfy-Org/ComfyUI/issues/16337), ROCm 7.14 [T]. v0.34 la activa por defecto solo en ROCm 7.14 o más [V]: no actualizar a la ligera |

**Fallo de 2^19 filas (inferencia a partir del código del repo):**
- Inferencia normal a 1024 px usa miles de tokens por imagen, muy por debajo de 524 288 filas. Con pocas referencias no debería activarse.
- **`int8_convrot` sí es peligroso.** El issue #357 muestra un `torch.matmul` de **3 dimensiones** con forma como `[248320, 64, 64] @ [64, 64]`, unos 15.9 millones de filas si PyTorch lo aplana. Nuestro `amd_parche_rocm.py` solo trocea `matmul` cuando ambas entradas son **2D**, y además excluye subclases de tensor (líneas de `_tensor_normal` y `a.dim() == 2`) [V]. Los pesos cuantizados pueden ser subclases, así que tampoco quedarían cubiertos.
- Evitarlo: bf16 o GGUF. El repo ya prohíbe int8 para Pixal3D (`INSTALAR_AMD.md`) [V].
- El arranque actual usa `--fp32-vae` para Pixal3D. Mantenerlo en la prueba y comparar.
- Si el codificador falla o sale corrupto en GPU, probarlo en CPU: lento, pero separa causas.

## 6. Quitar fondo y superresolución (breve)

| Pieza | Datos | Conclusión |
|---|---|---|
| [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) `e2bf8e4460` | MIT en pesos y [código](https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE) [V]: los cuatro usos, con aviso. 0.44 GB; 3.5 GB en FP16 y 4.8 GB en FP32 a 1024 px, en NVIDIA [F]. Nativo en ComfyUI desde v0.21.0 [V] | **Usarlo.** Ya está en los modelos del repo; la copia por rembg sobra si todo pasa por ComfyUI. gfx1200 con pelo de perro: [N]. Comparar con [Lucida](https://huggingface.co/egeorcun/lucida) (MIT [V], 0.88 GB) |
| Fondo pedido por texto | Pedir fondo liso que contraste con el perro, luego BiRefNet. Blanco con perro blanco es mala idea | Requisito del prompt |
| Real-ESRGAN x4plus (actual) | [BSD-3](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) en el repositorio [V]; 67 MB según `superres.py` | **Dejarlo opcional.** A 1024 px no hace falta de entrada; útil para fotos pequeñas |
| [SeedVR2 3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) | Apache 2.0 [V]; [fp16 6.78 + VAE 0.50 GB](https://huggingface.co/Comfy-Org/SeedVR2) [V]; nativo desde v0.28.0 [V] | No de entrada: puede inventar detalle [T] y hay [cambios abiertos por picos de memoria en ROCm](https://github.com/Comfy-Org/ComfyUI/pull/17826) |

Orden: BiRefNet **después** de estilizar, y otra vez por cada vista. No ampliar antes de aprobar.

## 7. Recomendación y reserva

**Recomendación: Qwen-Image-Edit-2511 GGUF Q4_K_M**, codificador `qwen_2.5_vl_7b_fp8_scaled`, VAE de Qwen. Primero sin LoRA; luego Trendy_figurines con fuerza baja; DeLight como segunda pasada; ángulos de fal para las vistas. Una edición por pasada. Descarga aproximada: 22.5 GB + 1.5 GB (figuras y ángulos) + DeLight (tamaño [N]).

Motivo: calidad antes que velocidad, sirve para vender figuras y también el programa, evita `int8_convrot`, y tiene el mejor indicio encontrado de identidad. **No hay prueba en nuestra GPU;** la puerta de decisión es la sección 9. [Lightning 4/8 pasos](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) (Apache [V], 0.85 GB) sirve para bocetos; se reporta piel de plástico [T].

**Reserva: FLUX.2 klein 4B destilado bf16**, en dos pasos (chibi; después fondo, pose y color), 1 MP. Vistas con el método de guía 3D, no con la hoja. Si calidad insuficiente, probar base 4B. Descarga: 16.1 GB.

**Si no se va a vender el programa,** medir además FLUX 3 Klein **bf16**. Si gana con claridad, puede ser la opción de uso propio y venta de figuras, aceptando registro, revisión de salidas y límite de ingresos. Es una decisión del autor.

## 8. Lo que habría que construir

1. **Formulario y texto fijo:** rasgos de la mascota (manchas por zona, orejas, cola, ojos), chibi, cuerpo entero, patas separadas, cola y orejas gruesas, sin ropa extra, fondo liso, color plano sin sombras ni brillos.
2. **Lista de rasgos aprobada por el usuario**, para comparar con la salida. Ningún modelo garantiza las manchas.
3. **Cliente HTTP de ComfyUI** con flujos en formato API, semilla, versión, hash de archivos y licencias. Carga y descarga por turnos.
4. **Puertas automáticas:** rechazar negro, NaN, un solo color, silueta cortada o patas fundidas. Una revisión humana cierra cada paso (también cumple la de FLUX 3).
5. **Albedo:** pasada DeLight y, si aún hay sombras, corrección con máscara de pocas tintas. Medir; no suponer.
6. **Vistas:** render 3D guía, edición, comparación de silueta (IoU), escala y color; reintentos. Hoja o 4 imágenes: ya hay flujos multivista en el repo.
7. **Prueba de arranque ampliada:** matmul 3D, matmul sobre subclases cuantizadas y Conv/atención, GPU contra CPU. El parche actual no cubre lo primero.
8. **Entrenar un LoRA chibi para perros** solo si la prueba falla. Hace falta un conjunto de pares con derechos claros; entrenamiento en ROCm Windows: [N].

## 9. Prueba concreta en la GPU

**Entorno:** copia aparte de ComfyUI v0.45.0, misma PyTorch, parche activo, `--fp32-vae`, Triton apagado, sin nodos extra salvo GGUF. Registrar revisión y SHA-256 de cada archivo, y pico de VRAM y RAM.

**Paso 0, sanidad (30 min).** Ejecutar `prueba_linear_grande.py`. Después comparar GPU y CPU en bf16: `Linear` de 600 000 filas, `matmul` 2D de 600 000 filas, y `matmul` 3D `[10000, 64, 64] @ [64, 64]` (640 000 filas). Esperado: los dos primeros pasan; el tercero decide si hay que ampliar el parche. Si la diferencia relativa supera 1 %, parar. No probar matmul fp32 por lotes con trabajo sin guardar: puede cerrar el proceso.

**Paso 1, una imagen.** Qwen 2511 Q4_K_M, 1024x1024, una foto, 20 pasos y después 40, semilla 1. Pasa si no hay negro ni NaN, dos ejecuciones iguales dan resultados prácticamente iguales, y la RAM no agota el archivo de paginación. Si falla: codificador en CPU; luego Q3_K_M; luego klein 4B.

**Paso 2, comparación.** 5 perros (claro, oscuro, manchado, pelo largo, sentado), 1 a 3 fotos cada uno, 4 semillas:

| Configuración | Imágenes |
|---|---|
| A. Qwen 2511 sin LoRA | 20 |
| B. A + Trendy_figurines 0.5 | 20 |
| C. klein 4B destilado en dos pasos | 20 |
| D. FLUX 3 Klein bf16 (si se acepta la licencia) | 20 |

Juzgar sin saber la configuración, por rasgo: manchas en zona correcta, orejas, hocico, color de ojos y pelo, cola, cuerpo entero, cuatro patas separadas, fondo, ausencia de sombra proyectada. Umbrales propuestos, **a confirmar por el autor**: al menos 3 de 5 perros con una imagen de 4 que tenga 8 de 10 rasgos, y ningún rasgo distintivo inventado o perdido. Medir también tiempo y picos.

**Paso 3, DeLight** sobre las 5 mejores. Medir el cambio de luminosidad dentro de zonas de un solo color y el desplazamiento de color de las manchas. Rechazar si una mancha cambia de forma o de tono visible.

**Paso 4, vistas.** Para los 2 mejores perros generar derecha, atrás e izquierda de dos modos: LoRA de fal directo, y guía 3D. Comparar altura, silueta contra espejo o render (meta IoU de 0.90 por confirmar) y continuidad de manchas. Después Pixal3D de 1 vista contra 4, misma semilla: si 4 vistas no mejora de forma visible, quitar esta etapa.

**Decisión:** A o B gana y no supera la memoria: recomendado. Si no, C. Si solo D alcanza el umbral, el autor decide si renuncia a vender el programa con él o consulta a BFL. Si ninguno pasa: LoRA propio o dibujo aprobado a mano.

## 10. Lo que queda sin verificar

- Rendimiento y calidad de Qwen 2511 y FLUX 3 en gfx1200; ninguna cifra encontrada.
- LICENSE originales con registro de klein 9B y FLUX 3 (los de FLUX 3 se leyeron en copias; el 9B en GitHub).
- Fidelidad a mascotas y a manchas, y generación real de color plano.
- Que un LoRA de 2509 funcione en 2511; efecto de los LoRA sobre GGUF.
- Licencias y datos de entrenamiento más allá de lo declarado en las fichas.
- Las fuentes marcadas "extracto" solo se leyeron en resultados de búsqueda; varias páginas se leyeron a través de una herramienta de resumen y no íntegras.
