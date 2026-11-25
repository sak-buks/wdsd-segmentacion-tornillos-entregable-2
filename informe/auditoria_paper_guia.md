# Auditoría de fuga y paper guía

Los criterios de este capítulo se escribieron antes de correr los scripts y quedaron en un commit antes de lanzar la réplica (`criterio_paper_guia_E2.md`; la fecha que cuenta es la del commit que lo agrega: 2026-11-24, 04:37 hora de Colombia, y la primera corrida de la réplica empezó a las 05:10). Aquí se informa lo que dio, también lo que no cumplió.

## Auditoría de fuga de la partición agrupada

**Qué es una fuga.** Es cuando el modelo ve en entrenamiento información que luego se usa para evaluarlo, y por eso su resultado en test sale mejor de lo que sería con imágenes realmente nuevas. En este proyecto el riesgo es que las fotos de una misma escena (mismo producto, misma mesa, misma luz) caigan unas en train y otras en test. La partición agrupada del E2 evita eso agrupando por escena; la auditoría comprueba que lo logra (`auditoria_fuga_e2.py`).

**Tabla 6.12.** Las cuatro verificaciones obligatorias del enunciado, con el criterio fijado antes.

| Verificación | Criterio de aprobación | Resultado (partición agrupada) | ¿Aprueba? |
|---|---|---|---|
| Intersección de grupos de escena | Vacía en las tres parejas (train–val, train–test, val–test) | Vacía en las tres | Sí |
| Duplicados exactos (SHA-256 de los bytes de cada imagen) | Cero colisiones entre particiones | 0 colisiones (y ningún hash repetido en las 945 imágenes de train, val y test) | Sí |
| Casi duplicados (pHash de 64 bits, distancia de Hamming ≤ 10 con alguna imagen de train) | A lo sumo el 5 % de test con casi gemelo | 12,0 % de test (13 de 108 imágenes); 6,5 % de val | **No** |
| Validación adversarial (regresión logística que intenta separar train de test, validación cruzada de 5 pliegues por grupo) | AUC no distinguible de 0,5 (p de permutación ≥ 0,05), o diferencia explicada | AUC 0,33; p de permutación 1,0 | Sí, con una nota |

El **pHash** (hash perceptual) resume una imagen en 64 bits según sus frecuencias bajas, así que dos fotos casi iguales dan hashes casi iguales. La **distancia de Hamming** cuenta en cuántos de los 64 bits difieren. La mediana de la distancia mínima de las imágenes de test a train es 16, y 13 imágenes quedan por debajo del umbral de 10.

**Lectura (observado).**
- **Lo que sí cumple:** ninguna escena aparece en dos particiones y no hay imágenes idénticas entre ellas. Eso es lo que prometía la partición agrupada.
- **Lo que no cumple:** el 12,0 % de test tiene un casi gemelo en train, más del doble de la tolerancia de 5 %. No es una fuga por escena (los grupos no se cruzan), sino escenas distintas que se ven muy parecidas (por ejemplo, el mismo tipo de producto fotografiado en condiciones similares).
- **Nota sobre la validación adversarial:** un AUC de 0,5 significa que el clasificador no distingue train de test. Aquí da 0,33, por debajo de 0,5. Eso no indica fuga (una fuga daría un AUC alto) y el criterio, que mira la cola superior de la permutación, se cumple. Pero un valor tan por debajo del azar no lo he explicado. Es posible que se deba a la validación cruzada por grupo con pocas escenas; no lo probé.

**Efecto de las 13 imágenes con casi gemelo en el desempeño.** Se recalcula el F1 por grupo de cada modelo sin ellas (95 imágenes).

**Tabla 6.13.** F1 por grupo en test con todas las imágenes y sin las 13 que tienen casi gemelo en train (media de 5 semillas).

| Modelo | Con todas (108) | Sin las 13 (95) | Diferencia |
|---|---|---|---|
| Mask R-CNN | 0,726 | 0,655 | −0,070 |
| HTC | 0,724 | 0,644 | −0,079 |
| Cascade | 0,723 | 0,636 | −0,087 |
| MS R-CNN | 0,716 | 0,643 | −0,073 |
| PointRend | 0,701 | 0,629 | −0,072 |
| YOLOv7 | 0,638 | 0,584 | −0,053 |
| CondInst | 0,607 | 0,543 | −0,064 |
| YOLO26 | 0,569 | 0,488 | −0,081 |
| YOLOv8 | 0,566 | 0,497 | −0,069 |
| Mask2Former | 0,545 | 0,475 | −0,070 |
| YOLO11 | 0,530 | 0,456 | −0,073 |
| YOLOv9 | 0,405 | 0,334 | −0,070 |
| SOLOv2 | 0,231 | 0,271 | +0,040 |

**Lectura (observado).** Sin esas imágenes el F1 baja entre 0,05 y 0,09 en 12 de los 13 modelos (SOLOv2, con un F1 muy bajo, sube 0,04). El orden entre modelos casi no cambia: solo se intercambian dos pares que ya estaban muy cerca (Cascade con MS R-CNN, y YOLO26 con YOLOv8). **Lectura (inferido):** las imágenes con casi gemelo son más fáciles para todos los modelos, lo cual es coherente con algo de fuga, pero también con que sean simplemente imágenes más fáciles (por ejemplo, escenas con mejor luz); con estos datos no se pueden separar las dos explicaciones. Como las conclusiones del benchmark comparan modelos entre sí, y el efecto es parecido en casi todos, el ranking casi no depende de estas 13 imágenes; el valor absoluto del F1 sí sale algo inflado, unos 0,07 si la causa fuera la fuga.

**Lo que no se midió en la auditoría.** La fuga por preentrenamiento (COCO e ImageNet pudieron incluir imágenes parecidas a las de test) no se calcula; sí se declara como amenaza. La prueba de etiquetas permutadas está en el capítulo anterior.

## Paper guía: réplica parcial y su limitación

**El paper.** Kalitsios et al. (2022, DOI 10.1109/ICRCV55858.2022.9953178) es el artículo original de WDSD y el único trabajo con el mismo dataset. Entrenan Mask R-CNN con ResNet-101 con la partición oficial (765/90/90 imágenes) y reportan, en test, un AP de máscara de 0,586 y un AP50 de 0,849. (El AP promedia la precisión con umbrales de IoU de 0,5 a 0,95; el AP50 solo usa 0,5, que es mucho menos exigente con el borde de la máscara.)

### Réplica

**Qué se hizo.** Mask R-CNN ResNet-101 de MMDetection, preentrenado en COCO, con la partición oficial y la misma métrica (AP de máscara de COCO en las 90 imágenes de test). Protocolo del E2: 6 épocas, lote efectivo 2, tasa por defecto, anclas de 8 a 128 px, semillas 42, 43 y 44. Diferencias conocidas con el paper: número de épocas, tamaño de lote, programa de tasa y anclas (el paper no reporta todos esos detalles en lo que citamos).

**Criterio fijado antes.** La métrica principal es el AP50. Sea D = AP50 nuestro (media de semillas) − 0,849, con IC 95 % por bootstrap de 2000 remuestreos de las imágenes de test. Réplica lograda si |D| ≤ 0,05; parcial si 0,05 < |D| ≤ 0,10; fallida si |D| > 0,10.

**Tabla 6.14.** Réplica de Kalitsios et al. 2022 (media de 3 semillas).

| Métrica | Nuestro | Paper | D | IC 95 % de D | Por semilla (42, 43, 44) |
|---|---|---|---|---|---|
| AP50 (principal) | 0,737 | 0,849 | −0,112 | [−0,183, −0,048] | 0,729; 0,739; 0,742 |
| AP | 0,288 | 0,586 | −0,298 | [−0,335, −0,260] | 0,288; 0,288; 0,288 |

**Resultado (observado): réplica fallida** según el criterio, porque |D| = 0,112 > 0,10, aunque por poco. El AP50 queda 0,11 por debajo, con un IC que va de −0,18 a −0,05. El AP es mucho más bajo (0,29 contra 0,59), y esa diferencia es grande y no depende de la semilla.

**Causas posibles (no probadas).** Hay más de una, y este trabajo no las separa.
- Las diferencias de protocolo ya declaradas: aquí 6 épocas, y el paper puede haber entrenado mucho más. Los modelos del E2 (por ejemplo los YOLO) muestran que 6 épocas dejan algunos modelos subentrenados.
- El AP castiga mucho más que el AP50 los bordes poco precisos: con tornillos de ~12 px, un error de 1–2 px en el borde cambia mucho el IoU. Que nuestro AP50 esté más cerca del paper que nuestro AP es coherente con esto, pero no lo demuestra.
- Detalles que el paper no permite comprobar (tipo de AP, tamaño de imagen de entrada, filtro de detecciones).

### Limitación del paper

**La limitación.** La partición oficial es aleatoria por imagen. Si las imágenes de test tienen casi gemelos en train, su desempeño puede estar inflado por fuga.

**Qué se hizo.** Se entrenaron con la partición oficial tres modelos del Top 10 (MS R-CNN, PointRend y YOLO26), más el ResNet-101 de la réplica, con el mismo protocolo, la tasa elegida en la búsqueda del E2 y las semillas 42–44. Para cada uno se compara el F1 de instancia por imagen (media de semillas) entre las imágenes de test con casi gemelo en train y las que no lo tienen (mismo criterio: pHash con distancia de Hamming ≤ 10). Es una prueba exploratoria; no se usa para elegir modelo.

**Criterio fijado antes.** Se confirma la limitación si en al menos 2 de los 3 modelos la diferencia (con gemelo − sin gemelo) es positiva y su IC 95 % por bootstrap por grupo de escena (2000 remuestreos) excluye 0.

**Casi gemelos en la partición oficial.** 52 de las 90 imágenes de test (57,8 %) tienen un casi gemelo en train (y el 26,7 % de val). 40 de los 54 grupos de escena de test también están en train. Con la partición agrupada del E2 eran el 12,0 %. El preregistro citaba el 48,9 %, de la exploración de datos del E1: no es el mismo cálculo (otra definición de casi gemelo) y no los reconcilié.

**Tabla 6.15.** F1 de instancia por imagen en el test oficial, con y sin casi gemelo en train.

| Modelo | F1 con gemelo (52) | F1 sin gemelo (38) | Diferencia | IC 95 % | ¿Confirma? | AP con / sin gemelo |
|---|---|---|---|---|---|---|
| MS R-CNN | 0,785 | 0,612 | +0,173 | [0,024, 0,338] | Sí | 0,351 / 0,259 |
| PointRend | 0,794 | 0,608 | +0,186 | [0,038, 0,341] | Sí | 0,346 / 0,272 |
| YOLO26 | 0,552 | 0,417 | +0,135 | [−0,012, 0,279] | No | 0,131 / 0,142 |
| Mask R-CNN R101 (réplica) | 0,726 | 0,575 | +0,151 | [0,009, 0,295] | — | 0,312 / 0,246 |

**Resultado (observado): limitación confirmada en 2 de los 3 modelos del criterio.** MS R-CNN y PointRend rinden más en las imágenes con casi gemelo, con un IC que excluye 0. En YOLO26 la diferencia es del mismo orden (+0,135) pero su IC incluye 0. En el ResNet-101 de la réplica el IC también excluye 0, pero ese modelo no entraba en el criterio.

**Salvedades importantes.**
- **El F1 global no sale inflado.** Con la partición oficial, el F1 por grupo de MS R-CNN es 0,715 y con la agrupada del E2 es 0,716; el de PointRend, 0,715 contra 0,701; el de YOLO26, 0,509 contra 0,569 (más bajo en la oficial). Por tanto, a nivel global no se ve que la partición oficial dé mejores números que la agrupada, aunque la mitad de su test tenga gemelos. Las dos particiones no tienen las mismas imágenes de test, y esa comparación no es una prueba.
- **Fuga y facilidad no se separan.** Como en la auditoría de la partición agrupada, las imágenes con casi gemelo pueden rendir más porque están en train (fuga) o porque son más fáciles. Con estos datos no se distingue.
- **El IC es ancho.** Con 90 imágenes de test que vienen de 54 grupos de escena, los ICs van de casi 0 a unos 0,3: la dirección es clara, el tamaño no.

**Conclusión.** La evidencia es consistente con la limitación señalada, pero no la prueba. Es una razón para que el E2 use la partición agrupada y no compare sus números absolutos con los del paper.
