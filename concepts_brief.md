# 3DIA local: de texto y fotos a figuras imprimibles y articuladas

- **Author:** GermanRmz
- **Date:** 2026-11-24

## Overview

3DIA sería una app que corre en la PC del autor: Windows 10, Ryzen 7 3700X, 32 GB de RAM, Radeon RX 9060 XT de 16 GB (gfx1200), ROCm 7.2.1 y unos 170 GB libres. No usa APIs en la nube.

Flujo propuesto:

1. Texto y fotos de referencia (por ejemplo, la perrita del autor) producen una imagen estilizada (chibi).
2. La imagen aprobada produce un modelo 3D maestro con textura. Alternativa de entrada: un GLB subido.
3. La geometría se repara, se pule solo donde haga falta, se corta y se articula.
4. El maestro alimenta dos salidas: (a) textura completa, para ver o imprimir en un color y pintar a mano; (b) 3MF de hasta 4 filamentos para Bambu Studio.
5. Cada pieza recibe una orientación de impresión recomendada. PLA y PETG primero; resina después.

Impresora: Bambu A1 mini (180 mm cúbicos), AMS lite de 4 colores, figuras de ~10 cm, PETG habitual. La boquilla de 0.4 mm es un supuesto. El autor se queja de líneas de capa visibles. Pide la calidad máxima sin importar el tiempo. El uso es personal y venta de figuras; quizá venta del programa.

**Éxito supuesto, sin confirmar por el autor:** foto de la perrita + texto -> chibi de 10 cm, sin errores de malla en Bambu Studio en ambas ramas, con al menos 4 articulaciones que giran sin romperse en PETG.

**Conclusión del análisis:** la generación 3D es la parte más resuelta, pero está fuera de la app. Lo no demostrado es la reparación de malla, las articulaciones de PETG a 10 cm, y localizar piezas sin la API de Claude. Las pruebas físicas de articulación deberían ir antes que la interfaz.

**Base de código y alcance de la revisión.** Las citas `archivo:línea` se refieren a la rama de GitHub `optimizacion-y-textura`, commit `0a695d5`, y las rutas son relativas a su raíz. El checkout local del directorio de trabajo está 52 commits atrasado y no se revisó. La lectura fue estática: no se ejecutó código ni se imprimió nada. "No verificado" marca lo que depende de fuentes externas o de pruebas. "Inferencia" marca lo deducido del código.

**Estado de los revisores:** egm-design, egm-codebase-details, egm-zoom-out, egm-comprehension, egm-implementation-readiness, egm-structural y egm-inventive-catalyst: los 7 `completed`.

## Analysis

### Estado por etapa

| Etapa | Qué existe | Estado |
|---|---|---|
| Texto + fotos -> imagen | No se encontró en la app. Hay recorte y quitado de fondo (`preparar_imagen.py:89-143`) | Falta |
| Imagen -> maestro | Pixal3D/TRELLIS.2 con parches AMD y workflows (`amd/trellis2/INSTALAR_AMD.md:148-160`). La documentación lo da por correcto con dos figuras (`amd/trellis2/LEEME_PIXAL3D_AMD.md:3`) | Funciona a mano; no está conectado a la app |
| App actual | Usa Hunyuan: `app.py:18` importa `generar3d`; `modelos.py:23-28` fija pesos tencent | Hay que cambiarla |
| Cargar y escalar un GLB | `dividir.py:31-59`, `acabados.py:145-158`. Altura por defecto 60 mm | Reutilizable con cambios |
| Reparar | `limpieza.py:46-87` trata solo trozos sueltos | Falta la reparación general |
| Pulir por zonas | No se encontró. `acabados.py:161` añade relieve a toda la malla | Falta |
| Cortes | `formas.py`, `personaje.py`, `segmentar.py`. Vista explosionada posterior al corte (`segmentar.py:262`) | Base reutilizable; falta separar plan y ejecución |
| Articulaciones | `mecanismos.py:233-286` (giro, bisagra, pasadores) y rótula en `personaje.py:404-411` | Requiere rediseño parcial |
| Mano y trompa | Todas las hijas cuelgan de un `cuerpo` (`mecanismos.py:239`, `284`) | Faltan cadenas padre-hijo |
| 3MF de color | `multicolor.py:39-127`, `segmentar.py:302` | Reutilizable con cambios |
| Orientación por pieza | `segmentar.py:246-255` traslada, no rota | Falta |

### Riesgos bloqueantes

**1. Reparación de malla.**
- `segmentar.py:66-67` aborta si la malla no es cerrada.
- `limpieza.reparar` devuelve la malla intacta cuando hay un solo trozo (`limpieza.py:49-50`). Si la booleana falla, concatena sin avisar (`limpieza.py:42-43`). Los trozos alejados más de 4% del tamaño del cuerpo se quitan (`limpieza.py:20`, `65`, `79`), lo que podría afectar accesorios reales.
- Solo se llama desde la generación Hunyuan (`generar3d.py:332`). El GLB subido no pasa por ella (`dividir.py:36-41`).
- Si `process=True` (`acabados.py:148`) suelda las costuras UV del GLB de Pixal3D: no verificado.
- Propuesta: conservar el maestro visual (GLB con UV, inmutable) y derivar un sólido imprimible. Reparar en escalones: soldar y cerrar huecos; parche local; remallado SDF solo si falla lo anterior. Medir la desviación por zona. Una puerta de validación con estados `válida | reparada | fallida`, sin caídas silenciosas.

**2. Articulaciones.**
- El esquema acepta solo `giro` y `bisagra` y los ejes x, y, z (`mecanismos.py:40`, `44`); `PiezaMovil` no tiene padre (`mecanismos.py:107-119`).
- Los pasadores bajan hasta r=1.0 mm (`mecanismos.py:746-754`). Una pieza demasiado fina queda sin eje y solo con aviso (`mecanismos.py:756-758`); lo mismo si no hay apoyo (`726-728`).
- La rótula usa como mucho el 45% del radio inscrito (`personaje.py:380`), un cuello de 0.45 del radio de la bola (`410`) y una boca fija de 0.88 (`411`). No comprueba la pared del padre. Sus caras de corte quedan coplanares, sin cono de alivio: probablemente gira sobre su eje pero no se inclina (inferencia).
- `construir` (`mecanismos.py:233-286`) no comprueba colisiones en el rango de ángulos.
- Holguras: 0.2 (`dividir.py:76`), 0.3 (`mecanismos.py:234`), presión 0.05 (`mecanismos.py:785`). Ninguna depende del material.
- El radio de hombro se fija entre 3.8 y 6.0 mm (`mecanismos.py:534`), grande para una figura de 10 cm.
- Propuesta: biblioteca paramétrica de juntas con tamaño mínimo medido, árbol padre-hijo y barrido de colisión. Un cupón de calibración en PETG fija la holgura por material. "No cabe" bloquea la exportación, no es un aviso. Contar las juntas construidas, no las propuestas.

**3. Escala.** Dedos articulados en una figura completa de 10 cm pueden no ser viables en FDM de 0.4 mm. Alternativa: mano como modelo aparte, más grande. La trompa puede ser una cadena de segmentos, quizá con cordón o núcleo flexible.

**4. Puente con ComfyUI.** No se encontró cliente HTTP en la app. Los workflows están en formato de interfaz, no de API. El entorno depende de versiones y parches fijos (`INSTALAR_AMD.md:100-126`).
- Propuesta: ComfyUI en proceso y entorno separados; geometría en CPU sin hy3dgen.
- Fallo silencioso: con más de 2^19 filas, ROCm devuelve datos erróneos sin error (`LEEME_PIXAL3D_AMD.md:7-12`). El parche cubre `F.linear` y `torch.matmul` 2D (`amd_parche_rocm.py:45-74`); si no se instala, solo avisa (`amd_parche_rocm.py:85-88`). Un modelo de imagen nuevo puede verse afectado: no verificado. Hace falta una prueba de arranque que aborte.

**5. Persistencia y aprobación.** Las salidas van a archivos temporales (`app.py:29`, `67`, `128`). Un error tardío puede costar otra generación de 20-40 min.
- Propuesta: carpeta por proyecto con manifiesto (semilla, versión, parámetros), maestro inmutable, reanudación por etapa y puertas de aprobación: imagen, maestro, reparación/pulido, paleta, cortes, articulaciones, exportación.

### Riesgos mayores

- **Análisis local.** La llamada a la nube está solo en `mecanismos.py:144-191` (envía foto y vistas). `construir` acepta JSON de cualquier origen (`mecanismos.py:199-202`), así que el punto de cambio es claro. `formas.detectar` ya se usa para detectar apéndices (`mecanismos.py:241-246`). Cuatro productores posibles: plantilla (cuadrúpedo, bípedo, elefante), geometría, modelo local, manual. Si un VLM local da coordenadas útiles: no verificado.
- **Editor.** Hoy se edita JSON y se ven cajas sobre vistas 2D (`mecanismos.py:205-209`). Falta saber si Gradio basta para planos, ejes y máscaras 3D.
- **Cama.** Filas de 200 mm (`segmentar.py:249`) y volumen de 220x220x250 (`segmentar.py:94`), frente a 180 mm. Sin límite en Y ni reparto en placas. Hace falta un perfil único de impresora, material y boquilla.
- **Color.** Vecino más próximo (`multicolor.py:39-51`) y k-means RGB con fusión a distancia 40 (`multicolor.py:54-84`). Ojos pequeños pueden perder su color y las sombras pueden ocupar un filamento (inferencia). Se asigna un filamento por triángulo (`multicolor.py:87-96`). Propuesta: paleta con filamentos reales, colores reservados, distancia Lab, unión de islas, caras internas de corte a un filamento fijo. La app permite 8 colores (`app.py:200`, `262`, `293`); deben ser 4 como máximo.
- **Líneas de capa.** Pulir la malla no las elimina: dependen de altura de capa, orientación y lijado. La mejor orientación visual puede ser la peor para la resistencia del eje. Se necesita una función con prioridad a las juntas y a la cara. Boquilla 0.2: no verificado para PETG y AMS lite.
- **Pulido selectivo.** Hace falta definir "zona que lo necesite". Propuesta: máscara protegida editable (ojos, dedos, bordes), desplazamiento máximo en mm y vista antes/después.
- **Montada contra separada.** Montada funde las piezas en un objeto (`mecanismos.py:855-860`), lo que impide orientar, lijar y pintar por separado. La pintura suma grosor a la holgura.
- **Licencias y dependencias.** Hunyuan está en `requirements.txt:2`, `generar3d.py:50`, `71` y `textura.py:144`, `180`; `preparar_malla` importa hy3dgen aun sin textura (`generar3d.py:335`). Los límites territoriales y de uso de resultados que el propio repositorio declara están en `README.md:279-283`; el texto oficial y sus condiciones comerciales: no verificado. Al quitar hy3dgen pueden faltar dependencias transitivas, porque trimesh, scipy y otras no figuran en `requirements.txt`. No se encontró archivo LICENSE.
- **Derechos a separar:** usar, vender figuras, vender el programa, redistribuir pesos. También derechos sobre personajes o fotos de terceros.
- **Sin pruebas automáticas.** No se encontró suite; `prueba_*.py` son pruebas manuales de GPU.
- **Memoria y disco.** Los modelos se deben turnar en la GPU (convivencia en 16 GB: no verificado). Pixal3D recomienda unos 80 GB libres, con ~25 GB de modelos bf16 (`INSTALAR_AMD.md:20`). `app.py:145` lee `/proc/meminfo` y no protege la RAM en Windows.
- **Otros:** los nombres llegan sin escapar al XML (`segmentar.py:330`); el giro posterior al relieve no actualiza `transformacion` (`acabados.py:191-193`), lo que puede desplazar el color recuperado del STL (inferencia).

### Contradicciones y resoluciones propuestas

| Contradicción | Resolución | Costo |
|---|---|---|
| Cerrar la malla / conservar detalle | Sólido y color separados; remallado como último escalón; mapa de desviación | Memoria; pruebas por caso |
| Holgura / resistencia | Junta en el cuerpo, eje impreso acostado o trozo de filamento (`mecanismos.py:781-787` ya lo usa), cupón PETG | Montaje manual |
| Una imagen / piezas conocidas | Pedir en la imagen pose abierta, extremidades gruesas y separadas; plantilla de esqueleto | Limita la expresión; animales raros quedan para edición manual |
| Textura / 4 filamentos | Imagen de color plano sin sombras; o segunda imagen de color para el mismo maestro | Puede bajar la calidad 3D: comparar |
| Automático / editable | Propuesta automática con confianza y correcciones guardadas como cambios sobre un maestro fijo | Reglas de invalidación |
| Calidad máxima / iterar | Muchas imágenes baratas, aprobación, pocos maestros de noche | Disco |

### Discrepancias entre revisores

- **Orden del trabajo:** zoom-out, implementation-readiness y structural proponen empezar por un GLB subido. Design empieza por cupones y una figura con 4 juntas hechas a mano. Los cinco que fijaron un orden coinciden en hacer el cupón PETG antes de la interfaz.
- **Pulido:** zoom-out lo aplaza hasta medir si el ruido supera lo que imprime la boquilla; inventive-catalyst y structural proponen una máscara de protección común. Ambas opciones se prueban sin automatizar primero.
- **Dónde se bifurca el maestro:** structural sostiene que las ramas deben separarse en la exportación, no tras generar; de otro modo se vuelve a duplicar la geometría.
- **Ciclos de prueba de las juntas:** se propusieron 50, 100 y 200. El autor debe fijar el número.

## Goals vs. Non-Goals

### Goals (propuestos)

- Un solo maestro por figura; las dos ramas divergen en la exportación.
- Todo corre sin red tras descargar los modelos; la prueba se hace con la red bloqueada. Falta que el autor confirme si se permiten descargas iniciales.
- GLB generado o subido -> sólido validado -> STL/3MF, sin errores en una versión fijada de Bambu Studio. La validación propia mide también autointersecciones y pared mínima.
- Rama (b) con 4 filamentos como máximo y colores editables.
- Al menos 4 juntas construidas y probadas en PETG, con ángulo, ciclos y repeticiones definidos. Una junta pedida que no quepa falla la ejecución.
- Cortes y juntas como plan editable antes de ejecutar; cada cambio invalida solo lo posterior.
- Conservar ojos, dedos y bordes dentro de un desplazamiento máximo en mm.
- Registro de licencia por modelo y biblioteca.

### Non-Goals (propuestos para la primera versión; confirmar con el autor)

- Eliminar líneas de capa mediante pulido de malla.
- Dedos articulados a 10 cm.
- Resina como objetivo principal.
- Instalador comercial y otras GPU.
- VLM local como requisito. Pulido global.
- Entrenar un generador 3D desde cero.
- Animación y acabados decorativos como trabajo nuevo (decidir si se conservan los actuales).

La trompa y la mano siguen siendo la apuesta diferencial. Lo propuesto es probar su tamaño mínimo antes de comprometerlas, no descartarlas.

## Alternatives Weighed

| Alternativa | Estado | Motivo |
|---|---|---|
| Dos programas separados | Descartada; unificado como un maestro con dos ramas | Duplicaría código |
| Generación en la nube | Rejected by author | Todo debe correr en su PC |
| Entrenar un generador 3D desde cero | Planteado por el asistente | 16 GB no alcanzan |
| Pulido global | Rejected by author | Borra ojos, dedos y bordes |
| Hunyuan como base | Planteado por el asistente; **confirmar con el autor** | Licencia (límites en `README.md:279-283`) |
| API de Claude para piezas | Sustituida por análisis local (decisión tomada) | 100% local; hoy en `mecanismos.py:144-191` |
| Integrar ComfyUI dentro de la app | Propuesta en contra | Parches y versiones frágiles; proceso aparte |
| Blender sin interfaz para geometría | Abierta | Remallado y booleanas hechos; licencia GPL si se distribuye: no verificado |
| Usar Bambu Studio para cortes y pintura fina | Abierta | Menor costo propio; límites de la herramienta: no verificado |
| Zócalos paramétricos que coloca el usuario | Abierta | Más predecible que la detección automática |
| Segmentación neuronal de piezas | Aplazada | Solo si plantilla y geometría fallan; soporte ROCm: no verificado |
| Print-in-place o piezas separadas | Abierta | Decidir por junta; separadas facilita lijar y pintar |
| Boquilla 0.2, relieve o capas finas | Abierta | Comparar en una cabeza impresa |

## Key Decisions & Staging for Next Steps

### Decisiones tomadas

1. Un maestro con dos ramas, no dos programas.
2. Análisis de piezas local.
3. Sin APIs en la nube.
4. La rama de GitHub sirve como base.

### Decisiones propuestas

5. Orden: reparar -> analizar y proteger -> pulir -> cortar y articular -> orientar -> exportar. Validar en la carga y en la exportación.
6. Contratos: `proyecto.json`, plan de piezas (cortes y juntas), ensamblaje con padre-hijo, perfil de impresión.
7. Análisis: plantilla + geometría + edición manual; el VLM queda opcional.
8. ComfyUI como proceso HTTP local congelado; una tarea de GPU a la vez. FLUX en una copia separada.
9. La imagen se aprueba antes de gastar 20-40 min.

### Etapas

| Etapa | Contenido | Criterio de salida (umbrales por confirmar) |
|---|---|---|
| 0. Pruebas | Cupón PETG (holguras 0.15-0.5, pasadores 2-4 mm, juntas y orientaciones); GLB reales en la base y en Bambu Studio; 3MF de 1-4 colores; matriz de licencias; prueba de FLUX | Cada prueba con umbral de descarte; tamaño mínimo de junta medido |
| 1. Núcleo GLB -> sólido | Manifiesto, reparación, validación, dos exportaciones, cama de 180 mm, pruebas automáticas | Corpus de 10-20 GLB; casos fallidos con motivo |
| 2. Cortes y articulaciones | Plan editable, biblioteca de juntas, colisiones; extremidad, cadena (trompa), mano | Extremidad -> figura de 10 cm -> 3 repeticiones sin rotura |
| 3. Color y orientación | Paleta real, islas, orientación por pieza | Comparación en impresión |
| 4. Puente ComfyUI | Pixal3D, reanudación, control de cordura | 5 de 5 generaciones completas |
| 5. Foto + texto -> chibi | Modelo elegido | Imagen aprobada por el autor |
| 6. Si la medición lo pide | Pulido selectivo, VLM, resina | Definir al llegar |

Las etapas 4 y 5 pueden hacerse a mano en ComfyUI desde el inicio. Antes de refactorizar: pruebas de caracterización y marcar `0a695d5` como punto de retorno. Documentos de diseño sugeridos: general (contratos y licencias), núcleo geométrico, juntas y cortes, generación, color y exportación, pulido y orientación, editor.

### Preguntas de investigación por etapa

Todos los datos de modelo, licencia y VRAM de esta tabla son **no verificados**. Cada agente de investigación debe citar versión, fuente y fecha, y distinguir uso, venta de figuras, venta del programa y redistribución de pesos. La prueba AMD usa gfx1200, ROCm 7.2.1 y Windows.

| Etapa | Modelo o biblioteca: preguntas | Licencia | VRAM / disco | Prueba AMD |
|---|---|---|---|---|
| Imagen estilizada | ¿FLUX.2 klein 4B u otro admite fotos de referencia y texto, y conserva los rasgos de un perro? ¿LoRA de estilo? | ¿Apache 2.0 cubre pesos, codificador de texto y VAE? ¿Uso de las imágenes generadas? | Pico con codificador; precisión; tamaño | Carga, tiempo, comparación GPU/CPU por el fallo de 2^19 filas; 5-10 fotos |
| Varias vistas | ¿Quién genera las vistas de atrás y lados coherentes? ¿Mejora el multivista? (`INSTALAR_AMD.md:158`) | Del generador de vistas | Modelo multivista extra | Comparar 1 vista contra 4 en la misma figura |
| Preparar imagen | BiRefNet: ¿es necesaria la copia de rembg? | Pesos y código | Bajo | Recorte de fotos reales de perro |
| Maestro 3D | Pixal3D, TRELLIS.2 y alternativas: calidad de forma y textura, caras delgadas, malla cerrada | Pixal3D, TRELLIS.2, DINOv3, MoGe, kit AMD, CuMesh, cuBQL; ¿la cadena completa permite vender figuras? | Pico real medido | ¿GLB cerrado tras decimar? Repetir con 5 figuras y registrar el pico |
| Textura y color plano | ¿Cómo quitar sombras horneadas? ¿Albedo disponible? | Del modelo elegido | Medir | Textura sin magenta ni NaN |
| Reparar | MeshLib, PyMeshLab, Manifold, remallado SDF, Blender: ¿cuál conserva ojos y dedos? | Cada biblioteca; ¿GPL o uso no comercial? | CPU/RAM con 700k caras | No necesita GPU; si la necesita, soporte ROCm |
| Pulir | Filtros bilaterales con máscara; ¿hay detector de ojos, dedos y bordes para chibis animales? | Biblioteca y pesos del detector | Medir | Detector sobre vistas renderizadas en gfx1200 |
| Piezas y zonas | Plantillas contra geometría (`formas.py`) contra segmentación 3D contra VLM local; ¿precisión de coordenadas? | Pesos y datos de entrenamiento | Pico del VLM; descarga tras uso | 10 modelos etiquetados a mano; comparar con `formas.detectar` |
| Cortes | ¿Qué cortes de Bambu Studio bastan? ¿Cómo elegir costuras para lijar y pintar? | Herramientas externas | CPU | No aplica |
| Articulaciones | Diseños de junta publicados (rótula, bisagra, cadena); ¿licencias de los diseños? | Cada diseño | CPU | No aplica; cupón PETG |
| Orientación | Función de soporte, visibilidad y resistencia; ¿qué ofrece Bambu Studio? | Bibliotecas | CPU | No aplica |
| 3MF | ¿La versión fijada acepta `paint_color` de un 3MF genérico? ¿Se puede laminar por línea de comandos? | Formato y metadatos | CPU | No aplica; prueba en Bambu Studio |
| Editor | ¿Gradio basta o hace falta visor 3D propio? | Bibliotecas del visor | CPU/RAM del navegador | No aplica; prototipo de edición de una junta |
| Resina (secundaria) | ¿Qué reglas cambian? | No aplica | No aplica | No aplica |
| Superresolución | ¿Sigue siendo necesaria Real-ESRGAN (`superres.py`)? | Pesos y spandrel | Medir | Carga en gfx1200 |

### Preguntas abiertas para el autor

1. ¿Se confirma el criterio de éxito? ¿Qué cuenta como "girar sin romperse" (ángulo, ciclos, repeticiones)?
2. ¿Qué 4 articulaciones de la perrita? ¿Qué se hace si una no cabe: engrosar, agrandar la figura u omitir?
3. ¿Montadas o separadas? ¿La rama (a) también se corta y articula? ¿Y con juntas pintadas a mano?
4. ¿Mano y trompa son obligatorias en la primera versión? ¿Acepta una mano a mayor escala?
5. ¿Primero articulaciones sobre cualquier GLB, o primero foto -> chibi?
6. ¿Se elimina Hunyuan? ¿La boquilla es 0.4 y puede usar 0.2? ¿Qué PETG y qué versión de Bambu Studio?
7. ¿Se permiten descargas de pesos en la instalación? ¿La venta del programa condiciona la elección desde ahora?
8. ¿Quién juzga el parecido y cuántos intentos acepta? ¿Altura de 10 cm con orejas y base?
9. ¿El plan se conserva si se regenera el maestro? Propuesta: queda ligado a su huella (hash).
10. ¿Un GLB sin textura lleva paleta manual o sale en un color?
