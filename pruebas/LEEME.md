# Pruebas 1 y 2

Los archivos salen en `pruebas/salida` (`python pruebas/cubo_color.py [modelo.glb]` y `python pruebas/cupon_petg.py`).

## 1. Color en Bambu Studio (sin imprimir al principio)
1. Abre `1_placa_4_colores.3mf` con la A1 mini 0.4 y 4 filamentos. Comprueba que los 4 lados tienen filamentos 1 a 4 y que arriba hay tres filas de islas (0.2, 0.4, 0.8 y 1.2 mm). Lamina y mira qué islas sobreviven en la vista por capas.
2. Guarda el proyecto, ciérralo y ábrelo: ¿siguen los colores?
3. Abre `ComfyUI_00025_4colores.3mf` (nuestro) y, aparte, `ComfyUI_00025_original.glb` a 100 mm con la conversión de color de Bambu a 4 colores. Compara ojos y gafas, manchas, gramos purgados, cambios y tiempo.

## 2. Cupón PETG (capa 0.12, sin soportes, una placa por tarjeta)
Seca el PETG. No lijes ni lubriques en la primera ronda. Anota en `resultados_cupon.csv`.
- **2A giro:** prueba cada eje (2, 2.5, 3, 4) en los agujeros de su fila. La columna es la holgura en centésimas de mm por lado. Retira el eje con cuidado: es una pieza fina.
- **2B bisagra:** une las dos hojas del mismo número con 1.75 mm de filamento (mide su diámetro real). 1-3: ancho 8; 4-6: ancho 10; 7-9: ancho 12; dentro de cada grupo, holgura 0.20, 0.30, 0.40. Debe topar cerca de 30 grados.
- **2C rótula:** bola 6 en copas 1-4, bola 8 en 5-8, bola 10 en 9-12; dentro de cada grupo, holgura 0.15, 0.25, 0.35, 0.45. La boca está en el CSV. Anota si entra, si se sale sola, si sostiene la pose y si se agrieta.

Las juntas que pasen se repiten (10 copias, 200 ciclos) antes de fijar la holgura de la app.

## Prueba 1b: limpiar manchas de color

    venv\Scripts\python pruebas\limpiar_color.py pruebas\salida\ComfyUI_00025_original.glb 3 20

Une las manchas de color de menos de 3 (o 20) mm² con el color vecino. Resultado
laminado en Bambu Studio (A1 mini, PETG, 0.2 mm, purga fija de 280 mm³, sin torre):

| Archivo | Cambios | Filamento total | Tiempo |
|---|---|---|---|
| `ComfyUI_00025_4colores.3mf` (sin limpiar) | 1164 | 391 g | 41 h 42 |
| `ComfyUI_00025_original_limpio3.3mf` | 1090 | 367 g | 39 h 04 |
| `ComfyUI_00025_original_limpio20.3mf` | 986 | 334 g | 35 h 25 |

La app ya limpia a 3 mm² por defecto (`multicolor.AREA_MANCHA_MM2`).

## Prueba 1c: cortar por color (resultado negativo)

    venv\Scripts\python pruebas\corte_color.py pruebas\salida\ComfyUI_00025_original.glb

Estimación de cambios (capas de 0.2 mm; la estimación de pie dio 1146 y Bambu 1090):

| Opción | De pie (Z) | Tumbada (Y) |
|---|---|---|
| 4 colores | 1146 | 563 |
| 4 colores, mejor combinación de hasta 7 cortes planos | — | 563 (no mejora) |
| 4 colores, recolorear el 3 % de superficie más "cara" | 982 | 489 |
| 3 colores | 820 | 438 |
| 2 colores | 497 | 245 |

Los cortes planos no ayudan en esta figura: los 4 colores aparecen a toda altura.
Lo que más cambia el total es la orientación y la cantidad de colores.

## Prueba 3: medidor de orientación

    venv\Scripts\python pruebas\orientacion.py pruebas\salida\ComfyUI_00025_original.glb

Prueba 214 orientaciones y exporta 5 (`_orient1`, `_orient2`, `_orientB` = base plana,
`_orientZ` = de pie como sale del modelo, `_orientmY` = boca arriba), todas con la base
recortada 0.6 mm. Laminado en Bambu Studio con soporte de árbol (PETG, 0.2 mm, purga fija
280 mm³, sin torre):

| Orientación | Cambios | Tiempo | Filamento total | Soporte | Apoyo en cama | Soporte sobre el frente (estimado) | Escalones en el frente (estimado) |
|---|---|---|---|---|---|---|---|
| B: de pie, base plana | 1045 | 38 h 33 | 367 g | 32 g | 1519 mm² | 707 mm² | 419 (el mejor) |
| Z: de pie, base inclinada 9° | 1090 | 40 h 23 | 385 g | 39 g | 41 mm² | 772 mm² | 432 |
| 1: de costado, cara arriba | 741 | 28 h 29 | 271 g | 22 g | 84 mm² | 0 mm² | 652 |
| 2: de costado, cara arriba | 802 | 30 h 35 | 292 g | 24 g | 49 mm² | 13 mm² | 606 |
| -Y: boca arriba | 544 | 22 h 08 | 213 g | 27 g | 19 mm² | 0 mm² | 748 |

"Soporte" sale del G-code (líneas Support y Support interface).

Decisión: la calidad del frente manda, así que la app imprime de pie con base plana
(`acabados.a_impresion`, `base_plana=True` por defecto). Laminado de la figura así:
1044 cambios, 38 h 26, 4 colores.

## Prueba 4: cortes para lijar y pintar

    venv\Scripts\python pruebas\cortes_pintura.py pruebas\salida\ComfyUI_00025_original.glb 8

Corta por "costuras escondidas": bordes de color (la línea de pegado queda en la
línea de pintura) y pliegues hundidos (queda en sombra). Cortes locales (no corta
lo que pasa por el mismo plano en otro sitio), agujero de 3 mm en las dos piezas y
pasador suelto de 3 x 4 mm con 0.2 mm de holgura (ver prueba 5). Tarda ~12 min (la búsqueda de planos, 3 min; con `CACHE_CORTES=archivo.pkl`
se guarda para repetir).

Resultado con la figura de prueba: 8 piezas (gorro, cabeza con pelo, torso con brazo
izquierdo, brazo derecho, piernas, base, robot arriba y abajo), costura escondida
75-94 %. Laminado en un color (PETG, 0.2 mm, soporte de árbol, como salen del corte,
sin orientar): 2 placas, ~4 h y ~44.5 g en total, 0 cambios de filamento.
Pendiente: orientar cada pieza; un corte (patas del robot) sin espiga; la pieza de
las piernas se lleva una franja de la base.

## Prueba 5: orientación de cada pieza

Ahora todo está en `pintura.py` (la app lo usa en Dividir → "Para lijar y
pintar"); `cortes_pintura.py` lo ejecuta y deja `_pintura.3mf` y `.zip` con las
piezas ya orientadas. La orientación tarda ~5 s; el total, ~14 min (la búsqueda de
zonas ~3 min y los cortes ~10: cada corte cambia la pieza grande y hay que volver a
probar casi todas las zonas; repartirlo en procesos solo ganaba 1.3x porque
manifold3d ya usa varios núcleos).

Para cada pieza prueba ~200 direcciones más las caras planas y las normales de sus
cortes. Las caras de corte y de los agujeros no se ven montadas, así que el soporte
y los escalones sobre ellas no cuentan. Si la pieza puede apoyar 100 mm² (o el 80 %
del mejor apoyo posible) es obligatorio; entre esas gana la de menos escalones y
soporte en lo visible (frente y cabeza pesan más) y con los cortes horizontales
(agujeros redondos).

Cambio de unión: con espiga macho, la cara de corte de esa pieza no podía apoyar
(solo tocaba la punta de la espiga, 14 mm²) y la espiga quedaba acostada. Ahora
agujero en las dos piezas y pasador suelto, que se imprime de pie.

Resultado (Bambu, PETG 0.2 mm, soporte de árbol, un color, 14 pasadores incluidos):

| | Placas | Tiempo | Filamento | Parte de soporte (G-code) |
|---|---|---|---|---|
| Como salen del corte | 3 | 4 h 13 | 45.0 g | 32 % |
| Orientadas | 2 | 3 h 11 | 37.7 g | 17 % |

Apoyo por pieza: base 1507 mm², cabeza 976 (sobre el corte del gorro, boca abajo),
piernas 673, torso 207, patas del robot 168, brazo 46. El gorro y el cuerpo del
robot no tienen ninguna cara plana sin algo colgando debajo: van enteros sobre
soporte (las marcas caen en el ala del gorro y en la espalda del robot).

Arreglo: al gorro y al cuerpo del robot les sobresalían de la cara de corte puntas
de 1.3-1.4 mm (pelo o malla que cruza el plano; 1 y 5 mm³). Ahora, a las piezas
que apoyan menos de 100 mm² se les prueba a quitar lo que cruza su cara de corte
si es < 2 mm y < 50 mm³, y se queda solo si apoyan claramente mejor (la base, que
ya apoya por debajo, no se toca). Recortar dejaba motas sueltas de 0.02 mm³: se
descartan. Resultado: todas apoyan (gorro 979 mm², robot 176, brazo 205; el resto
igual), cerradas y de un trozo. Bambu: 2 placas, 3 h 02, 37.2 g, soporte 15 %
(antes 3 h 11, 37.7 g, 17 %).

Pasador de las patas del robot (corte 6): la sección mide 262 mm², pero las patas
se estrechan y la base es inclinada, así que la zona con material a los dos lados a
2.5 mm del corte solo admite un radio de 1.9 mm (el pasador de 3 x 4 con 1.2 mm de
margen necesita 2.7). Ahora se prueba una escalera: 3 x 4 (margen 1.2), 3 x 3
(margen 1.0, pared de 0.8 mm), 2.5 x 3 y 2 x 3. El corte 6 lleva uno de 3 x 3:
15 pasadores, todos los cortes con pasador. Comprobado desplazando cada agujero
0.6 mm hacia los lados: ninguno se sale de la pieza (pared >= 0.6 mm en el plano).
