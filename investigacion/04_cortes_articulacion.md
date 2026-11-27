# 04. Piezas, cortes y articulaciones

Fecha de consulta: 2026-11-25. No se instaló, descargó ni ejecutó ningún modelo, y no se imprimió nada. El código local se leyó en el commit `0a695d5`.

**Etiquetas**

- **[V]** Verificado: leído en la fuente (repositorio, LICENSE, ficha o API de Hugging Face, artículo). Significa "la fuente lo dice", no "lo probamos".
- **[F]** Lo afirma el fabricante, el autor o un usuario; sin prueba independiente.
- **[NV]** No verificado.
- **[P]** Propuesta o cálculo nuestro. Los números [P] son puntos de partida por medir.

**Usos de licencia:** (a) uso personal, (b) vender figuras, (c) vender el programa. No es asesoría legal. Sin licencia declarada = sin permiso claro; se trata como "no".

---

## 1. Qué elegir

| Área | Recomendación | Alternativa |
|---|---|---|
| 1. Piezas | Plantilla + `formas.detectar` + corrección manual; comparar con **HiTOPS** (MIT, sin pesos) | SAM 2.1 sobre vistas, guiado por puntos, y proyección a la malla |
| 2. Esqueleto | **TokenRig/SkinTokens** en proceso aparte, con el parche AMD (PR 14) y commit fijado | Plantillas propias de cuadrúpedo, bípedo y elefante |
| 3. VLM local | **Qwen3-VL-8B-Instruct Q8** con llama.cpp Vulkan, solo para nombres y puntos | Qwen3.5-9B; si falla, sin VLM |
| 4. Cortes | Optimizador propio: planos candidatos + costo de costura tipo Chopper + límites de junta | Corte manual en Bambu Studio |
| 5. Juntas | Biblioteca propia sobre Manifold: giro y bisagra con eje aparte, rótula con alivio, cadena con eje de filamento | BOSL2 (BSD) para prototipos; junta fija con pasadores |
| 6. Colisiones | `python-fcl` para filtrar + Manifold (`MinGap`, intersección) para confirmar | Solo Manifold, más lento |

Tres hallazgos que cambian el plan:

1. **La licencia de PartField bloquea buena parte del campo.** Limita el uso a investigación y educación no comercial [V]. Puppeteer, PartSAM, Particulate e Instruct-Particulate lo usan [V]. No sirven para vender figuras aunque su propia licencia diga Apache, MIT o CC-BY.
2. **Hay una prueba AMD muy cercana para el esqueleto.** El PR 14 de SkinTokens reporta RX 9070 XT, Windows 11, PyTorch 2.9.1 y ROCm 7.2.1: 50 s por rig completo de una jirafa; 200 s solo con CPU Ryzen 7 7800X3D [F]. El PR sigue abierto [V]. No es nuestra tarjeta ni Windows 10.
3. **No hallamos un programa abierto que haga todo:** de una malla cualquiera a una figura FDM articulada con juntas medidas, costuras ocultas y colisiones. El cupón de PETG pesa más que cualquier modelo de IA.

---

## 2. Segmentación 3D de piezas

| Opción | Estado | Licencia código / pesos y a / b / c | CUDA y AMD |
|---|---|---|---|
| **HiTOPS** ([repo](https://github.com/EngineeringAI-LAB/HiTOPS), [paper](https://arxiv.org/abs/2608.00767)) | Código real, push 2026-08-12, 6 estrellas [V] | MIT; **sin pesos** [V]. `third_party/marching_primitives` conserva su licencia, texto [NV]. Sí / sí / probable tras revisarla | Núcleo: numpy, scipy, trimesh, open3d, libigl, sin torch [V]. Remalla cerrada opcional con GPU (cubvh, diso) [V]: evitable. En nuestra PC [NV] |
| **P3-SAM / Hunyuan3D-Part** ([repo](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part), [pesos](https://huggingface.co/tencent/Hunyuan3D-Part)) | Código y pesos; X-Part es "versión ligera" [V] | [Licencia Tencent](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/blob/main/LICENSE) [V]: excluye UE, Reino Unido y Corea del Sur, **incluidos los resultados**; licencia aparte si había más de 1 millón de usuarios mensuales; no usar resultados para mejorar otra IA. a: sí fuera de esas zonas; b: riesgo al enviar figuras allí; c: distribución condicionada | Probado con CUDA 12.1; Sonata y extensión chamfer3D [V]. VRAM no declarada. AMD [NV]. Pesos de Sonata [NV] |
| **PartField** ([repo](https://github.com/nv-tlabs/PartField)) | Código y checkpoint de Objaverse; push 2026-06 [V] | [Licencia NVIDIA](https://github.com/nv-tlabs/PartField/blob/main/LICENSE), cláusula 3.3 [V]. Dudoso / **no** / **no** | CUDA 12.4, torch-scatter [V]; AMD [NV] |
| **PartSAM** ([repo](https://github.com/czvvd/PartSAM)) | Código, pesos y entrenamiento; push 2026-08 [V] | Parte MIT, pero la inferencia importa PartField y su LICENSE lo advierte; pesos sin licencia en HF [V]. Dudoso / **no** / **no** | CUDA 12.4, torch-scatter [V]; AMD [NV] |
| **SAMPart3D** ([repo](https://github.com/Pointcept/SAMPart3D)) | Código y pesos; push 2025-05 [V] | MIT / MIT [V]; dependencias [NV]. Probable en las tres tras auditarlas | Instrucciones CUDA, render con Blender [V]; AMD [NV] |
| **PartCrafter** ([repo](https://github.com/wgsxm/PartCrafter)) | Código y pesos; push 2026-04 [V] | MIT / MIT; baja RMBG-1.4 con licencia propia `bria-rmbg-1.4` [V]: quitarlo para b/c | CUDA y 8 GB [F]; AMD [NV]. **No segmenta nuestra malla: genera otra desde una imagen** [V] |
| **SegviGen** ([repo](https://github.com/Nelipot-Lee/SegviGen), [pesos](https://huggingface.co/fenghora/SegviGen)) | Marzo 2026, tres checkpoints [V] | MIT / MIT; la parte TRELLIS.2 conserva su licencia [V]. Sí / probable / auditar TRELLIS.2, DINOv3 y nvdiffrast | "Mínimo 24 GB" [F]. [Issue 5](https://github.com/Nelipot-Lee/SegviGen/issues/5): MI300X, ROCm 6.4.3, unos 107 s [F]. No demuestra 16 GB ni Windows |
| **GeoSAM2** ([paper](https://arxiv.org/abs/2508.14036)) | Código oficial utilizable [NV] | [NV] | [NV] |
| **HoloPart** ([repo](https://github.com/VAST-AI-Research/HoloPart)) | Código y modelo [F] | Código MIT [V]; pesos [NV] | Completa piezas ya segmentadas; no resuelve el primer paso. AMD [NV] |
| **SAM 2.1** ([pesos](https://huggingface.co/facebook/sam2.1-hiera-large)) | Pesos [V] | Apache-2.0 [V]. Sí / sí / sí con avisos | Segmenta imágenes; la proyección es nuestra. AMD [NV] |
| SAM 3.1 ([ficha](https://huggingface.co/facebook/sam3.1)) | Acceso con aprobación manual [V] | "SAM License" propia [V]; no auditada completa | [NV] |

**Recomendación:** plantilla + geometría + edición. HiTOPS encaja bien en licencia y hardware, pero es nuevo y casi sin adopción: probarlo medio día, no depender de él. Separa formas; no dice "pata" ni dónde gira. **Alternativa:** SAM 2.1 por vistas.

**Construimos:** máscara por cara, árbol padre-hijo, simetría, confianza y edición de frontera. `formas.detectar` recorre solo seis direcciones de ejes [V, código local]: apéndices diagonales o pegados son casos obligatorios del banco.

---

## 3. Esqueleto y tipo de junta

| Opción | ¿Código y pesos de verdad? | Licencia y a / b / c | CUDA y AMD |
|---|---|---|---|
| **SkinTokens / TokenRig** ([repo](https://github.com/VAST-AI-Research/SkinTokens), [pesos](https://huggingface.co/VAST-AI/SkinTokens), [paper](https://arxiv.org/abs/2602.04805)) | **Sí, ambos** [V] | Repo y pesos MIT. Incluye carpeta `michelangelo` y acredita ese proyecto [V], que es [GPL-3](https://github.com/NeuralCarver/Michelangelo/blob/main/LICENSE) [V]. Usa `bpy`. a sí; b probable; **c bloqueado hasta consejo legal o reemplazo** | Pide NVIDIA 14 GB y flash-attn [F]. [PR 14](https://github.com/VAST-AI-Research/SkinTokens/pull/14): AMD y CPU en Windows; `mixed` mejor que bf16 en ROCm [F]. [PR 10](https://github.com/VAST-AI-Research/SkinTokens/pull/10): R9700 [F]. Ambos abiertos [V] |
| **UniRig** ([repo](https://github.com/VAST-AI-Research/UniRig), [pesos](https://huggingface.co/VAST-AI/UniRig)) | **Sí**: entrenado en Articulation-XL2.0; el modelo completo Rig-XL/VRoid sigue pendiente [V] | Repo y pesos MIT; incluye Michelangelo **GPL-3** y `bpy==4.2` [V]. a/b/c como arriba | spconv, torch-scatter, torch-cluster, flash-attn [V]. No hallamos prueba AMD |
| **RigAnything** ([repo](https://github.com/Isabella98Liu/RigAnything), [pesos](https://huggingface.co/Isabellaliu/RigAnything)) | Sí, ambos [V] | Adobe Research: solo investigación no comercial [V]. No / **no** / **no** | `cuda:0` [V]; AMD [NV] |
| **Puppeteer** ([repo](https://github.com/Seed3D/Puppeteer), [pesos](https://huggingface.co/Seed3D/Puppeteer)) | Sí, esqueleto y piel [V] | Apache-2.0, con submódulos PartField y Michelangelo [V]. Conjunto: dudoso / **no** / **no** | CUDA 11.8, flash-attn, torch-scatter, bpy [V]; AMD [NV] |
| **MagicArticulate** ([repo](https://github.com/Seed3D/MagicArticulate)) | Código [V]; pesos no auditados | Apache-2.0 [V]; dependencias [NV] | AMD [NV] |
| **Particulate** ([repo](https://github.com/RuiningLi/particulate), [pesos](https://huggingface.co/rayli/Particulate)) | Sí, ambos [V] | Pesos CC-BY-4.0. La ficha dice código Apache-2.0, pero no hallamos el LICENSE raíz. Incluye y requiere PartField [V]. Dudoso / **no** / **no** | CUDA 12.4, Ubuntu [V]; AMD [NV] |
| **Instruct-Particulate** ([proyecto](https://instruct-particulate.github.io/), [Space](https://huggingface.co/spaces/rayli/instruct-particulate), [pesos](https://huggingface.co/rayli/instruct-particulate-model)) | **Sí, en la práctica**: inferencia en el Space y `step_0080000.pt` (3.05 GB); sin repo GitHub hallado [V] | Space y pesos sin licencia declarada; usa el checkpoint de PartField [V]. No claro / **no** / **no** | torch 2.8 cu128, torch-scatter; clientes de nube en requirements [V]; AMD [NV] |
| **Articulate AnyMesh** ([proyecto](https://articulateanymesh.github.io/), [paper](https://arxiv.org/abs/2502.02590)) | No hallamos repo propio en lo revisado | Sin licencia de código: no usar | Se apoya en VLM [V, resumen] |
| **Rigel3D** ([paper](https://arxiv.org/abs/2605.13129)) | No hallamos código ni pesos | Sin licencia | Genera malla y rig juntos [V, resumen]; no articula nuestro maestro |
| **MotionAnymesh** ([paper](https://arxiv.org/abs/2603.12936)) | Código [NV] | [NV] | Idea útil: ajustar ejes para evitar choques [F] |
| **RigMo** ([repo](https://github.com/haoz19/RigMo)) | Código de entrenamiento; pesos [NV] | [NV] | Pide secuencias animadas [F]; no es nuestro caso |

**Recomendación:** TokenRig con commit y parche fijados, en proceso aparte, para uso personal. **Alternativa:** plantillas propias. UniRig no mejora la licencia y añade spconv.

Aplicar PR 14 es ejecutar código de terceros sin fusionar: leerlo completo antes. También avisa que, con `--use_skeleton`, el orden de huesos puede asignar mal los pesos [F].

**Un rig no es una junta.** Un hueso no da tipo de eje, límites, pared ni espacio. Calì et al. vieron que los rigs profesionales traen demasiadas juntas sin propósito cinemático [V, [PDF](https://reality.tf.fau.de/projects/jointfit/jointfit.pdf)].

**Construimos:** reducción de huesos, paso a plantilla, ajuste del centro a la sección, eje libre, tipo y límites, y aprobación.

---

## 4. VLM local en 16 GB

| Opción | Tamaño | Licencia (código/pesos: a/b/c) | Utilidad y AMD |
|---|---|---|---|
| **Qwen3-VL-8B-Instruct** ([GGUF oficial](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)) | Q8 8.71 + visión F16 1.16 = **9.87 GB de archivos**; Q4_K_M 5.03 GB [V]. Contexto y búferes aparte | Apache-2.0 [V]: sí/sí/sí | Cajas y puntos relativos [F, [repo](https://github.com/QwenLM/Qwen3-VL)]. llama.cpp contiene `qwen3vl` [V] |
| **Qwen3.5-9B** ([original](https://huggingface.co/Qwen/Qwen3.5-9B), [GGUF de tercero](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)) | Q8 9.53 + visión F16 0.92 GB [V] | Apache-2.0 [V]: sí/sí/sí | Más nuevo; cajas en nuestras vistas [NV]. Hubo un [fallo 27B en Vulkan con GPU AMD+NVIDIA](https://github.com/ggml-org/llama.cpp/issues/20651), cerrado [V] |
| Qwen3.8-27B ([ficha](https://huggingface.co/Qwen/Qwen3.8-27B)) | 27B: exigiría cuantización fuerte o parte en RAM [P] | Apache-2.0 [V] | No es primera opción |
| Gemma 4 12B IT ([ficha](https://huggingface.co/google/gemma-4-12B-it)) | 23.9 GB sin cuantizar [V] | Apache-2.0 [V] | llama.cpp contiene módulos gemma4 [V]; cajas [NV] |
| Molmo2-8B ([ficha](https://huggingface.co/allenai/Molmo2-8B)) | 8B | Apache-2.0; datos de terceros [V] | Señala puntos [V]; llama.cpp/AMD [NV] |
| Florence-2-large ([ficha](https://huggingface.co/microsoft/Florence-2-large)) | Pequeño | MIT [V] | Vocabulario de chibis y AMD [NV] |
| LocateAnything-3B ([ficha](https://huggingface.co/nvidia/LocateAnything-3B)) | 3B | Licencia NVIDIA [V], no auditada | No elegir hasta aclarar b/c |

[llama.cpp](https://github.com/ggml-org/llama.cpp/releases) es MIT; `b12728` trae binarios Windows Vulkan y ROCm 10.0 [V]. Usar Vulkan para no tocar ROCm 7.2.1. No hallamos medidas de visión en una RX 9060 XT.

**Recomendación:** Qwen3-VL-8B Q8, una petición a la vez y ComfyUI fuera de la GPU. **Alternativa:** Qwen3.5-9B; después, solo plantillas.

**Cómo usarlo [P]:** no pedirle milímetros. Pedir nombre, vista y 1-3 puntos relativos; lanzar un rayo, cruzarlo con la malla y ajustar el corte por geometría. Dos vistas deben coincidir. Añadir atrás, otro lado y abajo. JSON validado, un reintento. El esquema actual solo admite cajas y ejes x/y/z [V, `mecanismos.py`].

---

## 5. Cortes que escondan la costura

### Trabajo académico

- **Chopper, 2012** ([PDF](https://gfx.cs.princeton.edu/pubs/Luo_2012_CPM/chopper.pdf)) [V]. Planos; optimiza tamaño, conectores, resistencia, fragilidad, simetría y costura. La costura se penaliza con oclusión ambiental, bordes de textura o zonas pintadas por el usuario. Por sí solo ese objetivo favorece piezas diminutas.
- **Filoscia et al., 2020** ([PDF](https://vcg-legacy.isti.cnr.it/Publications/2020/FAGMCC20/fagmcc20.pdf)) [V]. Cortes no planos que siguen pliegues. Imprimieron PLA, lijaron bordes y **aun así** rellenaron huecos con cianoacrilato y bicarbonato. Orden de montaje y conectores quedan como trabajo futuro. Código no hallado.
- **Ureta et al., 2016** ([PDF](https://cims.nyu.edu/~dzorin/papers/gilureta2016imm.pdf)) [V]. Juntas a partir de piezas, con decisión del usuario. Resina DLP; sus 0.2-0.4 mm no son valores FDM.

### Herramientas

| Herramienta | Licencia y estado | a / b / c y límite |
|---|---|---|
| Bambu Studio ([guía](https://wiki.bambulab.com/en/software/bambu-studio/cut-tool)) | AGPL-3; plano, cola de milano, plug, dowel y snap [V] | Sí / sí / no incrustar código sin cumplir AGPL. Salida manual |
| [pychop3d](https://github.com/gregstarr/pychop3d) | MIT; push 2023-07 [V] | Sí / sí / sí con aviso. Referencia; Windows actual [NV] |
| [PrintSplit](https://github.com/wenyen-hsu/printsplit) | GPL-3, Blender 4.2+, push 2026-07, 2 estrellas [V]. Bola, bisagra, giro y doble bola; holgura móvil por defecto 0.3 mm [F] | Sí / sí / no copiar a una app cerrada. Corte manual; sirve para comparar |
| [Manifold](https://github.com/elalish/manifold) | Apache-2.0 [V] | Sí / sí / sí. Corte, booleanas, `MinGap`, Minkowski [V] |
| [CoACD](https://github.com/SarahWeiii/CoACD) | 1.0.14, MIT, rueda Windows [V] | Sí / sí / sí. Convexos para colisión, **no** cortes estéticos |

Todo corre en CPU; ninguno necesita CUDA según lo revisado.

### Recomendación [P]

Generar 30-80 planos por cuello geométrico. Descartar primero los que cruzan ojos, dejan poca pared o no permiten montar. Después puntuar (términos 0-1; **pesos sin validar**):

```
C = 4*cara_protegida + 3*visibilidad + 2*borde_de_color_cruzado
  + 2*pared_justa + 1*area_cortada + 1*mala_orientacion
  + 1*dificultad_de_lijado + 0.5*asimetria + 1*mal_acceso
```

- **Costura fija:** cuello, pliegue, borde de ropa, línea de color o cara inferior. Dos pasadores o llave asimétrica; rebaje interior para pegamento.
- **Costura móvil:** borde redondeado en ambas piezas y hueco intencional. Parece una línea de diseño; no se rellena.
- **Lijado y pintura:** separar piezas antes de pintar, no pintar asientos, añadir a la holgura el grosor de pintura **medido**.

**Alternativa:** corte manual en Bambu Studio. **Construimos:** puntuación por vistas, conectores asimétricos, editor y prueba de camino de montaje.

---

## 6. Juntas imprimibles

### Evidencia de medidas

Regla: **`h` es el hueco por lado**; la diferencia de diámetros es `2h`. Las fuentes mezclan ambas.

| Fuente | Dato | Uso correcto |
|---|---|---|
| [Calì et al. 2012](https://reality.tf.fau.de/projects/jointfit/jointfit.pdf) | SLS y PolyJet: 0.2 mm se fundió; 0.3 se movió con buena fricción; radio funcional mínimo 5 mm [V] | La idea de bandas de fricción. **No son números FDM** |
| [MDPI 2025](https://mdpi-res.com/d_attachment/technologies/technologies-13-00436/article_deploy/technologies-13-00436.pdf) | PLA, Prusa MK3, capa 0.15, eje 10 mm: menor resistencia con t=0.15-0.25; fundido por debajo de 0.05 [V]. La tabla se extrajo desalineada; boquilla 0.35 por posición. Convención de t [NV] | Rango PLA inicial; no es rótula |
| [Pareek 2014](https://utw10945.utweb.utexas.edu/sites/default/files/2014-059-Pareek.pdf) | ABS: ejemplos 0.331 y 0.556 mm según función y esfuerzo; diferencia de diámetros en el primero [V] | Cada función necesita su medida |
| [Guía Univ. Zúrich](https://www.amf.uzh.ch/dam/jcr:fd1ee024-1a98-488b-89ec-ffd85befa786/Design-Guideline-FDM.pdf) | Pasador mínimo 3.0 mm [V]; 0.5 mm entre piezas móviles (visto en buscador) [NV] | Límite prudente |
| [Hubs](https://www.hubs.com/knowledge-base/how-design-parts-fdm-3d-printing/) | Pasador <5 mm: filete en la base; para los críticos, mejor uno comprado [V] | Preferir eje aparte |
| [JLC](https://jlc3dp.com/blog/print-in-place-hinge), [Sovol](https://www.sovol3d.com/blogs/news/print-in-place-3d-printing-how-to-design-hinges-joints-and-moving-parts-that-actually-work) | Valores contradictorios; una de las guías suma 0.05-0.10 por lado en PETG [F, vía buscador] | Solo para el rango del cupón |

**No hallamos una medición independiente de rótulas en PETG con boquilla 0.4.** No existe en lo leído un mínimo universal.

### Valores iniciales a probar [P]

| Clase | h PLA (mm) | h PETG (mm) | Tamaño |
|---|---:|---:|---|
| Eje separado que gira | 0.15-0.25 | 0.20-0.35 | eje 3-4; ensayar 2 y 2.5 |
| Bisagra con filamento 1.75 medido | 0.10-0.20 | 0.15-0.30 | pared 1.2-1.6 |
| Rótula montada | 0.15-0.25 | 0.20-0.35 | bola Ø8; ensayar 6 y 10 |
| Impresa en su sitio | 0.25-0.40 | 0.30-0.50 | Ø8 o más; hueco inferior aparte |
| Eslabones | 0.25-0.40 | 0.30-0.50 | ancho 8 o más, pared 1.2 o más |
| Canal de tendón | hilo + 0.4 total | hilo + 0.5 total | Ø1.5-2.5 |

1.2 mm son tres trazos nominales de 0.4: cálculo, no prueba. En un eje redondo, a igual momento, la tensión crece como `1/d^3`: 4 a 2 mm la multiplica por 8; 4 a 3, por 2.37. "Cabe" no es "aguanta".

### Clases de mecanismo [P]

1. **Giro** (cuello, cola): eje aparte impreso acostado; dos apoyos o retén.
2. **Bisagra** (patas): tres nudillos, eje de filamento o alambre, topes. Primera junta a probar.
3. **Rótula** (hombro, cuello): retención, fricción y holgura como parámetros separados; boca con alivio cónico y ranuras, o casquillo de dos mitades.
4. **Cadena / flexi** (trompa, cola): 5-8 eslabones. Ejes paralelos curvan en un plano; alternarlos 90 grados curva en el espacio y ocupa más.
5. **Tendón / elástico:** canal suave, acceso al nudo y camino de reemplazo. El elástico envejece; el hilo posa solo si hay tensión o fricción. Alma TPU impresa aparte; mezclarla por AMS lite [NV].
6. **Impresa en su sitio:** cadena plana o giro corto. Para lijar y pintar, mejor separadas.

Las trompas robóticas con TPU y tendones ([2603.19124](https://arxiv.org/abs/2603.19124), [CoLI](https://arxiv.org/abs/2606.20389)) solo se vieron en buscador; CoLI usa discos de 50 mm, no trasladables a una trompa pequeña.

### A 10 cm [P]

- **Mano:** muñeca sí, si caben bola Ø8 y pared; dedos no. Mano aparte: dedos de 8 mm o más, dos bisagras por dedo.
- **Pata de elefante:** bisagra oculta con eje 3-4; topes que dejen el pie plano.
- **Trompa:** cortar por pliegues y prever base gruesa desde la imagen. Si un tramo no da 8 mm, proponer engrosar, agrandar, fijar u omitir. Nunca reducir pared en silencio.
- **Perrita:** giro de cabeza, cola y dos patas con bisagra es el conjunto más simple; decide el usuario.

### Rótula actual

Código [V, `_unir` en `personaje.py`]: cuello `0.45R`, boca `0.88R`, centro a `R+1`. Cálculo [P]: para R=4, aun sin chocar las caras, el tubo limita la inclinación a unos `atan(0.43*4/5) = 19 grados`; las caras coplanares la reducen más. La interferencia de entrada es `0.12R` por lado, sin ranuras ni cálculo de pared. No basta cambiar una constante.

### Malla a modelo articulado imprimible

- **Bächer et al. 2012** ([resumen](https://vcg.seas.harvard.edu/publications/20120101-fabricating-articulated-characters-from-skinned-meshes)): malla con piel a juntas de fricción, sin penetración [F, resumen]. [US20150187134A1](https://patents.google.com/patent/US20150187134A1/en) figura abandonada en Google Patents [V]; **no equivale a una búsqueda de patentes**.
- **Calì et al. 2012:** malla simple + juntas elegidas por el usuario; rótula en su sitio con límites y fricción; mano y brazo impresos por SLS o PolyJet [V]. Código no hallado.
- Articulate AnyMesh y MotionAnymesh dan cinemática, no huecos FDM [F].

Aquí se estudian clases de mecanismo. No se deben copiar silueta, proporciones ni encajes de un modelo comercial.

---

## 7. Bibliotecas y colisiones

Todas son CPU; CUDA y AMD no aplican.

| Biblioteca | Versión y licencia | a / b / c | Uso |
|---|---|---|---|
| [manifold3d](https://github.com/elalish/manifold) | 3.5.5, Apache-2.0, rueda Windows [V] | Sí / sí / sí | Base de cortes y juntas |
| [trimesh](https://github.com/mikedh/trimesh) | 5.2.2, MIT [V] | Sí / sí / sí | Carga; no subir de versión sin pruebas |
| [python-fcl](https://github.com/BerkeleyAutomation/python-fcl) | 0.7.0.11, BSD-3, ruedas Windows [V] | Sí / sí / sí | Colisión, distancia y colisión continua |
| [Coal](https://github.com/coal-library/coal) | 3.0.3, BSD; rueda Windows [NV] | Sí / sí / sí | Alternativa |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | 4.0.1, Apache-2.0 [V] | Sí / sí / sí | Dinámica; innecesario al inicio |
| [CadQuery](https://github.com/CadQuery/cadquery) | 2.8.0, Apache-2.0 [V] | Sí / sí / revisar OCCT | CAD paramétrico |
| [build123d](https://github.com/gumyr/build123d), [bd_warehouse](https://github.com/gumyr/bd_warehouse) | build123d 0.13.0; ambos Apache-2.0 [V] | Sí / sí / revisar OCCT | CAD paramétrico; rótula lista [NV] |
| [BOSL2](https://github.com/BelfrySCAD/BOSL2) | BSD-2 [V] | Sí / sí / sí con aviso | Bisagra para filamento 1.75 o impresa en sitio; `gap` por defecto 0.2 [V]; cambiar según cupón |
| [NopSCADlib](https://github.com/nophead/NopSCADlib) | GPL-3 [V] | Sí / sí / cumplir GPL | No como base |
| [dotSCAD](https://github.com/JustinSDK/dotSCAD) | LGPL-3, push 2023 [V] | Sí / sí / cumplir LGPL | No necesario |

Las licencias del motor OCCT y del programa OpenSCAD no se auditaron aquí [NV].

**Recomendación [P]:** generar con `manifold3d`, filtrar con FCL y confirmar con `MinGap`. Paso de 2 grados, refinado a 0.25 cerca del contacto. Exigir hueco mínimo, no solo intersección cero. Probar contención, mallas cerradas, poses combinadas y camino de **inserción** de ejes y bolas. Tocarse en el asiento puede ser intencional: excluirlo por etiqueta. Si una booleana falla, se detiene. El muestreo puede perder choques breves: usar también barrido continuo. **Alternativa:** solo Manifold.

---

## 8. Lo que debemos construir

1. Plan de juntas: padre, centro, eje libre, tipo, límites, material, perfil, confianza y huella del maestro.
2. Fusión de plantilla, geometría, puntos del VLM y huesos; aprobación manual.
3. Optimizador de cortes con máscaras protegidas, visibilidad y conectores.
4. Generador de seis clases con tabla medida por material, boquilla y perfil.
5. Comprobación de pared, inserción, retención, topes y resistencia. Estados: `cabe`, `requiere engrosar`, `no cabe`; el último bloquea.
6. Barrido de colisión y camino de montaje.
7. Editor, vista explosionada y un objeto 3MF por pieza.
8. Pruebas automáticas y registro de licencias.

---

## 9. Cupón de calibración PETG [P]

**Preparación**

- A1 mini, cada placa dentro de 180 mm. Confirmar boquilla 0.4, marca, color y lote de PETG. Secar; guardar temperatura, flujo, capa, orientación y versión de Bambu Studio.
- Cribado: 3 copias por celda en posiciones distintas. Confirmación: 10 de la mejor. Capa 0.12; luego repetir ganadoras en 0.20 sin cambiar otro factor.
- Primer lote sin lijar, lubricar ni pintar. Después, lijado y pintado por separado.
- Marcar cada celda en relieve: tarjeta, diámetro, h. Medir pieza real con calibrador o galgas.

| Tarjeta | Celdas | Construcción | Medir |
|---|---|---|---|
| A. Giro | Ø2, 2.5, 3, 4; h .15 .20 .25 .30 .35 .40 .50 (28) | buje 8 mm, eje acostado; repetir 3 mejores con eje vertical y con filamento | giro libre, par, juego, rotura |
| B. Bisagra | ancho 8, 10, 12; h .20 .30 .40; eje 1.75 (9) | tres nudillos, topes +/-30 grados | ciclos, juego, carga |
| C. Rótula | Ø6, 8, 10; h .15 .25 .35 .45; boca .90 .94 .97 de R (36 posibles; cribar 12 balanceadas) | alivio 20 grados, 2 ranuras; luego casquillo partido | inserción, retención, ángulo, pose |
| D. En su sitio | Ø8, 12; h .25 .30 .40 .50; hueco inferior .20 .40 (16) | bisagra corta | liberación y residuos |
| E. Cadena | ancho 8, 10, 12; h .25 .35 .50 (9) | 6 eslabones, pared 1.2; mejor celda con 1.6 | curva, tracción, ciclos |
| F. Tendón | canal 1.5, 2, 2.5; radio 3, 5, 8 (9) | nailon, PE trenzado y elástico | paso, reemplazo, abrasión |
| G. Costura fija | plana, pliegue, llave; hueco .10 .20 (6) | curva de 25 mm | visibilidad a 30 cm antes y después de pintar |
| H. Pared de boca | .8, 1.2, 1.6, 2.0; ranura sí/no (8) | media boca de rótula | inserción y grieta |

Orden: A, B, C, H; luego D, E, F, G. En PLA repetir A y C.

**Aprobación inicial; el usuario debe confirmar**

- Ida y vuelta completa a 20 ciclos/min, inspección a 0, 10, 50, 100 y 200.
- Sostiene 2 veces el momento del peso durante 60 s.
- Al menos 9 de 10 sin rotura, grieta, atasco ni separación; juego final no mayor a 1.5 veces el inicial.
- En su sitio: se libera sin cuchilla.
- Elegir la menor holgura que pasa **y cuya siguiente holgura también pasa**; no promediar fallos. Ninguna celda que no pase entra en la tabla.
- CSV: material, lote, perfil, capa, orientación, tarjeta, medidas CAD y reales, par, juego, ciclos, fallo y foto.

---

## 10. Banco etiquetado de 10 modelos [P]

Seis salidas propias aprobadas del flujo 3D y cuatro sólidos paramétricos propios con verdad exacta. Congelar por hash.

| N.º | Modelo | Dificultad | Etiqueta |
|---:|---|---|---|
| 1 | Perro de pie, pose abierta | caso base | 4 patas, cola, cabeza; 4 juntas pedidas |
| 2 | Perro sentado | patas pegadas | zona no cortable y junta imposible |
| 3 | Gato, cola diagonal | eje libre | cola y centro |
| 4 | Bípedo, brazos abiertos | cuello, hombros, caderas | 6 partes |
| 5 | Bípedo con falda y pelo | falsos apéndices | cortes prohibidos |
| 6 | Elefante | patas gruesas y trompa | 4 patas, 6 segmentos |
| 7 | Mano grande, paramétrica | cadenas | 10 bisagras y padres |
| 8 | Trompa curva, paramétrica | eje y tendón | 6-8 ejes y ruta |
| 9 | Rueda y tapa, paramétrica | colisión conocida | ejes y ángulo del primer choque |
| 10 | Robot simétrico, paramétrico | eje oblicuo y junta que no cabe | pares, ejes, `no_cabe` |

**Etiqueta:** cara a pieza, árbol, centro, eje, rango, corte ideal y zonas aceptables, cara protegida, diámetro inscrito, orden de montaje y resultado esperado. Dos pasadas separadas; si difieren más de 1.5 mm en centro o 2 mm en frontera, revisar.

**Comparar:** `formas.detectar`, plantilla, HiTOPS, SAM proyectado, TokenRig, VLM e híbrido. **Medir:** IoU y F1 de piezas, frontera, centro, ángulo, árbol correcto, cortes prohibidos, juntas construidas, choques, minutos de corrección, tiempo y pico RAM/VRAM. Tres semillas si el método varía. Etiquetas de 1-10 no se usan para ajustar; tener otros modelos de desarrollo.

**Puerta:** cero cortes en cara protegida; todas las juntas imposibles detenidas; al menos 8/10 planes corregibles en 10 min; centro a 2 mm y eje a 15 grados o menos en las juntas aceptadas; cero choques detectados; modelo 1 impreso tres veces con sus cuatro juntas. La IA se conserva solo si reduce el trabajo manual frente a plantilla + geometría.

---

## 11. Orden y pendientes

1. Cupón y nueva rótula.
2. Plantilla + geometría sobre el banco, con barrido.
3. HiTOPS en CPU.
4. VLM en Vulkan.
5. TokenRig en entorno aparte: leer PR, fijar versiones y comparar CPU/GPU por el fallo local de 2^19 filas.
6. Costuras y pintura.

Pendiente: PETG y boquilla reales; juntas y carga; licencia de Sonata y código de terceros de HiTOPS; GPL en TokenRig; origen de los datos de [Articulation-XL2.0](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0) (ficha CC-BY-4.0 [V], objetos de terceros [NV]); seguridad por piezas pequeñas; patentes.

**Límites de esta búsqueda:** no se leyó el artículo completo de Bächer (la descarga no era PDF) ni todo el código. Las licencias se leyeron por sus cláusulas o encabezados relevantes, no por cada dependencia. "No hallado" no prueba que no exista. Nada se probó en la RX 9060 XT.
