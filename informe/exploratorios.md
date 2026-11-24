# Análisis exploratorios

El preregistro (§5 y §7) declara exploratorios a RQ3, la calibración, la robustez y la latencia: se reportan, pero **no se usan para elegir modelo** ni cambian las conclusiones de RQ1 y RQ2. Nada de este capítulo volvió a entrenar modelos.

- RQ3, calibración y tamaños de efecto salen de las predicciones de test guardadas por las 70 corridas finales (`preds_test.json`), con `analisis_extra_e2.py`.
- Calibración completa, índice de robustez, δ de Cliff, descomposición de la varianza, frontera de Pareto, errores compartidos, ensambles e incertidumbre salen de `analisis_extra2_e2.py`, con las mismas predicciones guardadas.
- Robustez y latencia salen de `robustez_latencia_e2.py`, que vuelve a predecir test con los pesos guardados. Solo cubre los 4 modelos del entorno `wdsd_dl` (Mask R-CNN, YOLOv8, YOLO11 y YOLO26).
- **Control de calidad:** al recalcular el F1 por grupo de cada corrida desde sus predicciones, la diferencia máxima con `corridas.csv` es 8 × 10⁻¹⁷. En robustez, el caso sin perturbar reproduce la tabla principal (Mask R-CNN 0,726; YOLOv8 0,566; YOLO11 0,530; YOLO26 0,569).

Tablas: `results/e2/analisis_extra/` y `results/e2/analisis_extra2/`.

## RQ3. ¿Cambia la ventaja con el tamaño del tornillo?

**Cómo se calcula.**
1. El área de un tornillo es el número de píxeles de su máscara.
2. Los cortes de tercil se fijan con los tornillos de **validación**, no de test: 75 y 157 px. En test quedan 155 tornillos pequeños, 117 medianos y 117 grandes.
3. El emparejamiento es el de la métrica primaria: húngaro uno a uno, IoU de máscara ≥ 0,5 y el umbral de score de cada corrida.
4. Cada acierto y cada tornillo perdido van al tercil del tornillo real; cada falso positivo, al tercil de su máscara predicha.
5. En cada grupo de escena y tercil, F1 = 2·TP / (2·TP + FP + FN). Luego se promedian las 5 semillas y los grupos.
6. Contraste con Mask R-CNN (el comparador de RQ1): diferencia por grupo, Wilcoxon bilateral y Benjamini–Hochberg (BH) sobre los 36 contrastes modelo × tercil. Los grupos que no tienen tornillos de un tercil no entran en ese tercil (12, 13 y 7–9 grupos).

**Tabla 6.1.** F1 por grupo en test por tercil de área (media de 5 semillas) y recall agregado de todos los tornillos.

| Modelo | F1 pequeño | F1 mediano | F1 grande | Recall pequeño | Recall mediano | Recall grande |
|---|---|---|---|---|---|---|
| Mask R-CNN (base) | **0,625** | 0,664 | 0,545 | 0,307 | 0,691 | 0,786 |
| HTC | 0,577 | 0,659 | 0,613 | 0,319 | 0,701 | 0,803 |
| Cascade | 0,565 | 0,657 | 0,621 | 0,284 | 0,687 | 0,785 |
| MS R-CNN | 0,539 | 0,665 | 0,618 | 0,295 | 0,711 | 0,752 |
| PointRend | 0,525 | 0,676 | 0,489 | 0,280 | 0,716 | 0,752 |
| YOLOv7 | 0,477 | 0,570 | 0,557 | 0,365 | 0,738 | 0,653 |
| CondInst | 0,381 | 0,624 | 0,671 | 0,160 | 0,685 | 0,817 |
| YOLO26 | 0,468 | 0,553 | 0,375 | 0,172 | 0,597 | 0,692 |
| YOLOv8 (base) | 0,392 | 0,524 | 0,378 | 0,164 | 0,585 | 0,627 |
| Mask2Former | 0,337 | 0,551 | 0,394 | 0,121 | 0,559 | 0,733 |
| YOLO11 | 0,352 | 0,546 | 0,379 | 0,132 | 0,535 | 0,629 |
| YOLOv9 | 0,222 | 0,436 | 0,319 | 0,097 | 0,462 | 0,309 |
| SOLOv2 (base) | 0,004 | 0,174 | 0,361 | 0,001 | 0,152 | 0,716 |

```{figure} ../results/e2/analisis_extra/rq3_terciles.png
:width: 85%
:align: center

**Figura 6.1.** F1 por grupo en test por tercil de área del tornillo.
```

**Lectura (observado).**
- **La familia de dos etapas no se separa de Mask R-CNN en ningún tamaño:** diferencias de −0,056 a +0,012 y ningún p ajustado por BH menor que 0,14.
- **Los demás pierden sobre todo en los tornillos pequeños.** Mask2Former pierde 0,26 en los pequeños y 0,06 en los grandes; SOLOv2, 0,62 contra 0,10; CondInst, 0,18 contra 0,03. Diferencias significativas tras BH (p ajustado < 0,05): Mask2Former, YOLO11, YOLOv9 y SOLOv2 en pequeños y medianos, y YOLOv8 en medianos.
- Respuesta a RQ3: **sí, la ventaja de Mask R-CNN crece cuando el tornillo es más pequeño**, salvo frente a la familia de dos etapas, que acompaña a Mask R-CNN en todos los tamaños.
- El recall de los pequeños es bajo en todos (como máximo 0,37), aunque su F1 por grupo sea alto. Las dos cifras ponderan distinto: el recall agregado pesa cada tornillo, y el F1 por grupo pesa cada escena por igual.
- En varios modelos el F1 de los grandes es menor que el de los medianos, aunque su recall sea mayor. Una explicación probable, no verificada, son falsos positivos grandes (máscaras que unen dos tornillos o toman fondo), que caen en el tercil grande.

**Lectura (inferido).** Es coherente con que la resolución de las cabezas de máscara limite a los modelos que no son de dos etapas en objetos de ~10 px, como se vio con SOLOv2 en el E1. Los YOLO y Mask2Former además estaban subentrenados (ver más abajo), así que su peor desempeño en pequeños no se puede atribuir solo a la arquitectura.

## Calibración del score

**Qué mide.** Cada predicción trae un score de confianza entre 0 y 1. El score está calibrado si, entre las predicciones con score cercano a 0,7, más o menos el 70 % son correctas. Aquí una predicción es correcta si se empareja con un tornillo real con IoU ≥ 0,5. El emparejamiento recorre las predicciones de mayor a menor score, como COCO, sobre todas las predicciones guardadas (score ≥ 0,05).

El **ECE** (error de calibración esperado) reparte las predicciones en 10 intervalos de score y promedia la distancia entre el score medio y la fracción de aciertos de cada intervalo, pesando por el número de predicciones. 0 es calibración perfecta.

**Tabla 6.2.** Calibración en test (media de 5 semillas; ECE ± DE entre semillas).

| Modelo | ECE | Score medio | Fracción de aciertos | Lectura |
|---|---|---|---|---|
| YOLO11 | 0,041 ± 0,010 | 0,226 | 0,244 | bien calibrado |
| YOLO26 | 0,045 ± 0,009 | 0,234 | 0,247 | bien calibrado |
| YOLOv8 | 0,051 ± 0,015 | 0,233 | 0,264 | bien calibrado |
| MS R-CNN | 0,075 ± 0,005 | 0,327 | 0,296 | sobreconfiado |
| CondInst | 0,098 ± 0,010 | 0,150 | 0,204 | infraconfiado |
| Cascade | 0,100 ± 0,007 | 0,333 | 0,234 | sobreconfiado |
| SOLOv2 | 0,107 ± 0,017 | 0,248 | 0,257 | mixto |
| YOLOv7 | 0,107 ± 0,047 | 0,256 | 0,362 | infraconfiado |
| HTC | 0,109 ± 0,019 | 0,296 | 0,187 | sobreconfiado |
| PointRend | 0,133 ± 0,013 | 0,363 | 0,230 | sobreconfiado |
| YOLOv9 | 0,151 ± 0,029 | 0,161 | 0,309 | infraconfiado |
| Mask R-CNN | 0,166 ± 0,006 | 0,386 | 0,220 | sobreconfiado |
| Mask2Former | 0,217 ± 0,012 | 0,405 | 0,188 | sobreconfiado |

```{figure} ../results/e2/analisis_extra/calibracion.png
:width: 60%
:align: center

**Figura 6.2.** Diagrama de fiabilidad: por debajo de la diagonal el modelo es sobreconfiado; por encima, infraconfiado.
```

**Lectura (observado).** Los YOLO de ultralytics son los mejor calibrados. Los de dos etapas y Mask2Former dan scores más altos que su tasa real de acierto: Mask R-CNN tiene score medio 0,39 y acierta 0,22. CondInst, YOLOv7 y YOLOv9 hacen lo contrario.

**Consecuencia.** El F1 usa un umbral de score elegido en validación, así que la mala calibración no lo baja directamente. Pero el score de Mask R-CNN no se puede leer como probabilidad de acierto.

### Otras medidas de calibración y recalibración

- **ECE adaptativo:** igual que el ECE, pero los 10 intervalos tienen el mismo número de predicciones en vez del mismo ancho de score. Se reporta porque el ECE depende de cómo se corten los intervalos.
- **Brier:** media de (score − acierto)², con acierto = 1 o 0. Mezcla calibración y capacidad de separar aciertos de errores; 0 es perfecto.
- **Pendiente e intercepto de calibración:** se ajusta una regresión logística del acierto sobre el logit del score, donde logit(p) = ln(p / (1 − p)). Un score perfecto da pendiente 1 e intercepto 0. Intercepto negativo: el score es más alto que la tasa de acierto. Pendiente mayor que 1: los scores están demasiado juntos y el modelo no se atreve a dar valores extremos.
- **Temperature scaling:** divide el logit del score entre una constante T y vuelve a aplicar la sigmoide. Solo cambia la escala, no corrige un sesgo.
- **Platt:** la misma regresión logística de la pendiente, usada para transformar el score. Corrige escala y sesgo a la vez.

**Desviación declarada.** El enunciado pide ajustar la recalibración solo en validación, pero las corridas no guardaron sus predicciones de validación. Se ajustó con validación cruzada por grupos sobre test: para cada uno de los 14 grupos de escena, T o Platt se estiman con los otros 13 y se aplican a ese grupo. El grupo evaluado nunca participa en su propio ajuste. Ninguna de las dos cambia el F1, porque son transformaciones que conservan el orden de los scores.

**Tabla 6.2b.** Calibración en test (media de 5 semillas). Las dos últimas columnas son el ECE después de recalibrar.

| Modelo | ECE | ECE adaptativo | Brier | Pendiente | Intercepto | ECE con T | ECE con Platt |
|---|---|---|---|---|---|---|---|
| YOLO11 | 0,041 | 0,050 | 0,140 | 1,26 | 0,39 | 0,051 | 0,057 |
| YOLO26 | 0,045 | 0,057 | 0,141 | 1,29 | 0,38 | 0,050 | 0,056 |
| YOLOv8 | 0,051 | 0,060 | 0,147 | 1,22 | 0,42 | 0,047 | 0,041 |
| MS R-CNN | 0,075 | 0,080 | 0,134 | 1,20 | −0,13 | 0,069 | 0,066 |
| CondInst | 0,098 | 0,101 | 0,116 | 2,35 | 2,50 | 0,096 | 0,016 |
| Cascade | 0,100 | 0,100 | 0,110 | 0,96 | −0,93 | 0,093 | 0,023 |
| SOLOv2 | 0,107 | 0,110 | 0,120 | 2,17 | 1,04 | 0,072 | 0,032 |
| YOLOv7 | 0,107 | 0,108 | 0,178 | 1,15 | 0,82 | 0,078 | 0,040 |
| HTC | 0,109 | 0,109 | 0,103 | 1,03 | −1,03 | 0,089 | 0,025 |
| PointRend | 0,133 | 0,133 | 0,134 | 0,83 | −1,19 | 0,141 | 0,027 |
| YOLOv9 | 0,151 | 0,149 | 0,217 | 0,81 | 0,61 | 0,251 | 0,239 |
| Mask R-CNN | 0,166 | 0,167 | 0,135 | 0,86 | −1,54 | 0,176 | 0,022 |
| Mask2Former | 0,217 | 0,217 | 0,150 | 1,05 | −2,29 | 0,218 | 0,023 |

**Lectura (observado).**
- El ECE adaptativo da casi el mismo orden que el de intervalos fijos: solo se intercambian modelos vecinos que difieren en 0,002 o menos. La lectura no depende de cómo se corten los intervalos.
- El problema de los de dos etapas (salvo MS R-CNN) y de Mask2Former es un sesgo, no una escala: intercepto entre −0,9 y −2,3 con pendiente cercana a 1. Por eso la temperatura casi no los mejora y Platt los deja en ECE ≤ 0,03. Mask R-CNN pasa de 0,166 a 0,022 y su Brier de 0,135 a 0,089 (`calibracion_completa.csv`).
- CondInst y SOLOv2 tienen el problema contrario (pendiente > 2, intercepto positivo), y Platt también los corrige.
- Los tres YOLO de ultralytics ya estaban calibrados, y recalibrarlos empeora un poco su ECE: el ajuste solo agrega ruido.
- YOLOv9 empeora con las dos recalibraciones (inferido: guarda pocas predicciones por corrida, unas 320, y el ajuste por grupos queda inestable).

## Robustez a perturbaciones

**Qué se hizo.** Se volvieron a predecir las 108 imágenes de test con cada perturbación, con los pesos de las 5 semillas y sin reajustar el umbral de score.
- Ruido gaussiano aditivo con σ = 0,02, 0,05 y 0,10 sobre la escala 0–1 de la imagen.
- Desenfoque gaussiano con σ = 1, 2 y 4 px.
- El ruido depende solo de la imagen y del nivel, así que los cuatro modelos ven exactamente el mismo.

**Tabla 6.3.** F1 por grupo en test (media de 5 semillas) y, entre paréntesis, la fracción que conserva del F1 sin perturbar.

| Modelo | Limpio | Ruido 0,02 | Ruido 0,05 | Ruido 0,10 | Desenfoque 1 | Desenfoque 2 | Desenfoque 4 |
|---|---|---|---|---|---|---|---|
| Mask R-CNN | 0,726 | 0,705 (97 %) | 0,628 (87 %) | 0,484 (67 %) | 0,672 (93 %) | 0,252 (35 %) | 0,016 (2 %) |
| YOLOv8 | 0,566 | 0,528 (93 %) | 0,366 (65 %) | 0,138 (24 %) | 0,469 (83 %) | 0,041 (7 %) | 0,001 (0 %) |
| YOLO11 | 0,530 | 0,509 (96 %) | 0,350 (66 %) | 0,100 (19 %) | 0,421 (79 %) | 0,035 (7 %) | 0,002 (0 %) |
| YOLO26 | 0,569 | 0,556 (98 %) | 0,408 (72 %) | 0,163 (29 %) | 0,447 (79 %) | 0,067 (12 %) | 0,001 (0 %) |

```{figure} ../results/e2/analisis_extra/robustez.png
:width: 95%
:align: center

**Figura 6.3.** F1 por grupo en test según el nivel de perturbación.
```

**Lectura (observado).** Mask R-CNN es el más robusto. Con ruido σ = 0,10 conserva el 67 % de su F1, y los YOLO entre el 19 % y el 29 %. Un desenfoque de 2 px le quita casi dos tercios del F1 a Mask R-CNN y más del 85 % a los YOLO; con 4 px ningún modelo detecta tornillos.

**Índice de robustez.** Es el área bajo la curva de retención (F1 perturbado dividido por F1 limpio) cuando los niveles 0, 1, 2 y 3 se ponen a igual distancia, dividida por 3. Vale 1 si el modelo no pierde nada y 0 si lo pierde todo desde el primer nivel. Por ejemplo, para Mask R-CNN con ruido las retenciones son 1; 0,97; 0,87 y 0,67. Con la regla del trapecio, el área es (1 + 0,97)/2 + (0,97 + 0,87)/2 + (0,87 + 0,67)/2 = 2,68, y 2,68 / 3 = 0,89. Se calcula por corrida y se promedia sobre las 5 semillas.

**Tabla 6.3b.** Índice de robustez (media ± DE entre semillas).

| Modelo | Ruido | Desenfoque |
|---|---|---|
| Mask R-CNN | 0,89 ± 0,01 | 0,60 ± 0,02 |
| YOLO26 | 0,78 ± 0,04 | 0,47 ± 0,03 |
| YOLOv8 | 0,74 ± 0,05 | 0,47 ± 0,05 |
| YOLO11 | 0,74 ± 0,04 | 0,45 ± 0,03 |

**Lectura (inferido).** Lo esperable con tornillos de ~10 px: 2 px de desenfoque borran el borde que los separa del fondo. La diferencia de robustez puede deberse en parte a que los YOLO estaban subentrenados.

**Límite.** Solo 4 de los 13 modelos. Los de MMDetection y YOLOv7/YOLOv9 necesitan su propio entorno y cargador.

## Latencia y costo

**Tabla 6.4.** Latencia de inferencia en una sola GPU (RTX 3060 Laptop, lote 1, 108 imágenes de test tras 10 de calentamiento, pesos de la semilla 42).

| Modelo | Mediana (ms) | P95 (ms) | Imágenes/s | GFLOPs (1 imagen) | Parámetros (M) | Pico GPU (GB) |
|---|---|---|---|---|---|---|
| Mask R-CNN | 88,2 | 109,8 | 11,0 | 392,4 | 43,9 | 0,68 |
| YOLOv8n-seg | 26,2 | 44,5 | 34,6 | 27,4 | 3,4 | 0,17 |
| YOLO11n-seg | 28,5 | 47,5 | 32,5 | 22,5 | 2,9 | 0,17 |
| YOLO26n-seg | 29,0 | 42,1 | 33,6 | 21,4 | 3,1 | 0,15 |

Mide desde la imagen ya en memoria hasta las máscaras en la CPU, sin leer el archivo ni codificar el resultado. Mask R-CNN corre con precisión mixta (FP16) y ultralytics en FP32, igual que cuando se calcularon las métricas. Los GFLOPs (miles de millones de operaciones) los cuenta `torch.utils.flop_counter` sobre una imagen: solo convoluciones y productos de matrices, sin operaciones como `roi_align` ni NMS, así que son aproximados.

**Tabla 6.5.** Costo de entrenamiento de todos los modelos: minutos por corrida final (6 épocas en la RTX 3060, media de 5 semillas; incluye predecir val, test y train93).

| Modelo | Parámetros (M) | Min/corrida | F1 test |
|---|---|---|---|
| Mask R-CNN | 43,9 | 17,9 | 0,726 |
| HTC | 77,2 | 27,4 | 0,724 |
| Cascade | 77,0 | 23,3 | 0,723 |
| MS R-CNN | 60,2 | 17,7 | 0,716 |
| PointRend | 55,8 | 17,5 | 0,701 |
| YOLOv7 | 37,9 | 24,6 | 0,638 |
| CondInst | 34,0 | 17,7 | 0,607 |
| YOLO26 | 3,1 | 8,1 | 0,569 |
| YOLOv8 | 3,4 | 4,6 | 0,566 |
| Mask2Former | 44,0 | 47,7 | 0,545 |
| YOLO11 | 2,9 | 5,2 | 0,530 |
| YOLOv9 | 27,6 | 26,5 | 0,405 |
| SOLOv2 | 46,2 | 17,3 | 0,231 |

**Lectura (observado).** Los YOLO nano son unas 3 veces más rápidos que Mask R-CNN, con 13 veces menos parámetros y entre 14 y 18 veces menos operaciones, a cambio de 0,16 a 0,20 de F1. HTC y Cascade cuestan más que Mask R-CNN (77 M de parámetros, 23–27 min por corrida) sin mejorar su F1. Mask2Former es el más caro de entrenar (48 min) y uno de los peores.

## Tamaños de efecto de RQ1

Los valores p dicen si una diferencia es distinguible del azar, no cuán grande es. Se dan dos medidas sobre las 14 diferencias por grupo frente a Mask R-CNN:
- **Correlación biserial de rangos** r = (W⁺ − W⁻) / (W⁺ + W⁻), donde W⁺ y W⁻ suman los rangos de las diferencias positivas y negativas del Wilcoxon. Va de −1 (el modelo es peor en todos los grupos) a +1.
- **d de Cohen pareada:** diferencia media dividida por la desviación estándar de las diferencias.

**Tabla 6.6.** Tamaños de efecto frente a Mask R-CNN (F1 por grupo en test).

| Modelo | Diferencia media | r biserial | d pareada |
|---|---|---|---|
| HTC | −0,002 | −0,06 | −0,08 |
| Cascade | −0,002 | −0,17 | −0,06 |
| MS R-CNN | −0,010 | −0,34 | −0,28 |
| PointRend | −0,025 | −0,47 | −0,46 |
| YOLOv7 | −0,088 | −0,66 | −0,71 |
| CondInst | −0,119 | −1,00 | −1,08 |
| YOLO26 | −0,157 | −1,00 | −1,48 |
| Mask2Former | −0,181 | −1,00 | −1,56 |
| YOLO11 | −0,196 | −1,00 | −1,81 |
| YOLOv9 | −0,321 | −1,00 | −1,92 |

**Lectura (observado).** HTC y Cascade tienen efectos prácticamente nulos. MS R-CNN y PointRend, pequeños a moderados y a favor de Mask R-CNN. Para los últimos cinco, r = −1: son peores en todos los grupos con diferencia distinta de cero.

Dos medidas más, que el enunciado pide en §13.3:
- **δ de Cliff:** toma los 14 F1 de grupo de cada modelo y compara cada valor de uno con cada valor del otro (14 × 14 = 196 pares, sin emparejar por grupo). δ = P(modelo > Mask R-CNN) − P(modelo < Mask R-CNN). **A12** = P(modelo > Mask R-CNN) + la mitad de los empates; 0,5 es empate.
- **Probabilidad de superioridad pareada:** fracción de los 14 grupos en que el modelo supera a Mask R-CNN en ese mismo grupo (empate = 0,5). Su IC 95 % sale de 2000 remuestreos bootstrap de los grupos.

**Tabla 6.7.** δ de Cliff, A12 y probabilidad de superioridad frente a Mask R-CNN (`cliff_superioridad.csv`).

| Modelo | δ de Cliff | A12 | P(supera a Mask R-CNN en un grupo) [IC 95 %] |
|---|---|---|---|
| HTC | −0,01 | 0,50 | 0,46 [0,21; 0,71] |
| Cascade | −0,04 | 0,48 | 0,39 [0,18; 0,64] |
| MS R-CNN | −0,05 | 0,48 | 0,39 [0,14; 0,64] |
| PointRend | −0,10 | 0,45 | 0,32 [0,11; 0,57] |
| YOLOv7 | −0,32 | 0,34 | 0,14 [0,00; 0,36] |
| CondInst | −0,30 | 0,35 | 0,04 [0,00; 0,11] |
| YOLO26 | −0,39 | 0,30 | 0,04 [0,00; 0,11] |
| Mask2Former | −0,49 | 0,26 | 0,00 [0,00; 0,00] |
| YOLO11 | −0,51 | 0,25 | 0,00 [0,00; 0,00] |
| YOLOv9 | −0,60 | 0,20 | 0,00 [0,00; 0,00] |

**Lectura (observado).** HTC le gana a Mask R-CNN en casi la mitad de los grupos, y ningún modelo le gana en la mayoría. El δ de Cliff sale más pequeño que la r biserial porque no empareja por grupo: la diferencia entre grupos (siguiente sección) es mucho mayor que la diferencia entre modelos, y al mezclar grupos se diluye.

## Descomposición de la varianza

**Qué se hizo.** El F1 de cada modelo en cada grupo y semilla (13 modelos × 14 grupos × 5 semillas = 910 valores) se descompone de dos formas.
- **Modelo lineal mixto (§13.2):** F1 = media + efecto del modelo + efecto aleatorio del grupo + efecto aleatorio de la semilla + error. Fijo quiere decir que se estima un valor por modelo; aleatorio, que solo se estima cuánto varía.
- **Sumas de cuadrados:** reparte la variación total entre modelo, grupo, semilla, la interacción modelo × grupo (modelos a los que les va distinto en distintos grupos) y el resto, que es la variación entre semillas dentro de cada modelo y grupo.

**Tabla 6.8.** Parte de la variación total del F1 por grupo (`varianza_sumas_cuadrados.csv`).

| Fuente | % de la variación |
|---|---|
| Grupo de escena | 61,5 |
| Modelo | 18,9 |
| Modelo × grupo | 12,5 |
| Semilla dentro de modelo y grupo | 7,0 |
| Semilla común a todos los modelos | 0,1 |

El modelo mixto da lo mismo: varianza del grupo 0,069 (DE 0,26), de la semilla común casi 0 y residual 0,021. La semilla común sale en 0 como se espera, porque la semilla 42 de un modelo no tiene nada que ver con la 42 de otro.

**Lectura (observado).**
- El grupo de escena pesa tres veces más que el modelo. Dos grupos casi no se resuelven con ningún modelo: el 9 (escena `eco12`, 28 imágenes; F1 medio de los 13 modelos 0,02) y el 6 (`diaplay17`, 15 imágenes; 0,12). Los demás grupos están entre 0,35 y 0,87.
- La desviación estándar entre semillas de un mismo modelo es 0,031 en promedio (0,022 sin Mask2Former), contra 0,146 entre modelos: elegir arquitectura importa mucho más que la semilla.
- Los contrastes del modelo mixto frente a Mask R-CNN con corrección de Holm (`contrastes_modelo_mixto_holm.csv`) repiten RQ1: HTC, Cascade, MS R-CNN y PointRend no se distinguen de Mask R-CNN (p de Holm = 1) y los otros ocho son peores (p de Holm ≤ 0,002).

## Frontera de Pareto entre desempeño y costo

Un modelo está **dominado** si otro tiene F1 igual o mayor con un costo igual o menor. La frontera de Pareto son los no dominados.

```{figure} ../results/e2/analisis_extra2/pareto.png
:width: 100%
:align: center

**Figura 6.4.** F1 de grupo en test frente a parámetros, minutos de entrenamiento y latencia. En rojo, los modelos no dominados.
```

**Lectura (observado, `pareto.csv`).**
- Mask R-CNN está en todas las fronteras: es el de mayor F1 y no cuesta más que los que lo empatan.
- HTC y Cascade quedan dominados en parámetros y en tiempo: tienen 77 M de parámetros contra 44 M y tardan 23–27 min contra 18, con el mismo F1.
- En el extremo barato están los YOLO nano: YOLO26 es no dominado en parámetros, tiempo, latencia y FLOPs.
- En tiempo de entrenamiento MS R-CNN y PointRend salen no dominados por segundos de diferencia con Mask R-CNN (17,5–17,7 min contra 17,9), lo que no es una diferencia real.
- El costo por unidad de mejora del enunciado, η = (F1 del modelo − F1 del mejor baseline) / (costo del modelo − costo del mejor baseline), no se calcula: ningún modelo supera al mejor baseline, así que el numerador es siempre negativo.
- La latencia y los FLOPs solo están para 4 modelos.

## Errores compartidos y ensambles

**Qué se hizo.**
- **Correlación de errores:** para cada imagen y modelo, error = 1 − F1 de la imagen (media de 5 semillas). Se calcula la correlación de Spearman (correlación de los rangos) entre cada par de modelos.
- **Errores compartidos:** un tornillo real cuenta como perdido por un modelo si no lo detecta en al menos 3 de sus 5 semillas.
- **Ensamble por voto:** por semilla, se juntan las máscaras sobre el umbral de tres modelos. Las que se solapan con IoU ≥ 0,5 se agrupan (una por modelo como máximo) y el grupo se conserva si lo detectan al menos 2 de los 3. La máscara que queda es la del modelo con mejor F1 en validación.
- **Oráculo:** en cada imagen se toma el mejor de los tres modelos. No se puede usar en la práctica, porque exige saber cuál acierta, pero marca el máximo que podría dar combinarlos.

**Lectura (observado, `errores_compartidos.json` y `ensambles.csv`).**
- Los modelos fallan en las mismas imágenes: la correlación de errores media es 0,82 (entre 0,38 y 0,97).
- De los 389 tornillos de test, 100 los pierden los 13 modelos. 76 de esos 100 son pequeños (área menor que 75 px) y 73 están en los grupos 9 y 6. Entre los tres mejores, el 74 % de los tornillos que pierde alguno los pierden los tres (146 de 196).
- El ensamble de los tres mejores (Mask R-CNN, HTC y Cascade) da 0,736 contra 0,726 de Mask R-CNN solo: +0,011 con IC 95 % [−0,002; 0,024] y Wilcoxon p = 0,14. No es una mejora demostrada. Un ensamble más variado (Mask R-CNN, CondInst y YOLOv7) da −0,007.
- El oráculo llegaría a 0,775 (+0,049): ese es el techo de combinar estos tres modelos.

**Lectura (inferido).** Como todos fallan en los mismos tornillos, el límite está en los datos (tornillos muy pequeños y dos escenas difíciles) más que en la arquitectura. Según el §12.3 del enunciado, esto apunta a dificultad intrínseca o a ruido de etiqueta. Los 158 tornillos que pierden 10 o más modelos están listados en `tornillos_dificiles_para_todos.csv`, para la taxonomía de errores.

## Incertidumbre predictiva en los tres mejores

**Qué se predice.** Para dar un intervalo hace falta una cantidad numérica por imagen. Se usa el número de tornillos de la imagen, que es lo que interesa en una inspección. El modelo predice tantos como máscaras deja sobre su umbral.

**Métodos.**
- **Predicción conforme por grupos.** Para cada imagen se calcula el error |predichos − reales|. Para evaluar un grupo se toman los errores de las imágenes de los otros 13 grupos, y q es el valor que deja por debajo el 90 % de ellos (cuantil conforme, con nivel nominal 1 − α = 0,90). El intervalo de cada imagen del grupo es [predichos − q, predichos + q]. Se deja fuera el grupo entero, y no imagen por imagen, porque las imágenes de una misma escena se parecen (el enunciado pide variantes por bloques cuando hay dependencia).
- **Ensamble de semillas.** El intervalo va del mínimo al máximo de tornillos predichos por las 5 semillas.

**Tabla 6.9.** Cobertura empírica (fracción de imágenes cuyo número real de tornillos cae dentro del intervalo) y ancho medio (cuántos valores enteros abarca el intervalo). `incertidumbre_cobertura.csv`.

| Modelo | Conforme: cobertura (nominal 0,90) | Conforme: ancho | Peor grupo | Rango de 5 semillas: cobertura | Rango: ancho |
|---|---|---|---|---|---|
| Mask R-CNN | 0,94 | 6,2 | 0,73 (grupo 6) | 0,58 | 3,2 |
| HTC | 0,90 | 6,9 | 0,59 (grupo 32) | 0,65 | 3,5 |
| Cascade | 0,93 | 5,9 | 0,79 (grupo 6) | 0,47 | 2,6 |

**Lectura (observado).**
- La predicción conforme cumple la cobertura nominal en promedio, pero con intervalos anchos, de unos ±3 tornillos, cuando las imágenes tienen 3,6 tornillos en promedio.
- No la cumple en todos los grupos (cobertura condicional): baja a 0,59–0,79 en el peor grupo, y a 0,78–0,88 en imágenes con 3 a 5 tornillos.
- El rango entre semillas cubre solo entre el 47 % y el 65 %: las semillas coinciden entre sí más que con la verdad, así que su desacuerdo subestima la incertidumbre.

## Presupuesto de 6 épocas

**Qué se observó.**
- La regla de parada del preregistro marca, en cada corrida, si el AP de máscara en validación de la última época supera al de todas las anteriores.
- YOLOv8, YOLO11, YOLO26 y Mask2Former la cumplen en 4 de 5 semillas, con ganancias en la última época de +0,016 a +0,020 de AP (YOLO) y +0,009 (Mask2Former). Los demás modelos ganan +0,003 o menos, o pierden.
- Declararlos subentrenados con el corte de 4 de 5 semillas es una decisión nuestra, tomada después de ver las curvas. Con 3 de 5 entrarían también YOLOv9, Cascade, PointRend y Mask R-CNN, con ganancias casi nulas.
- Parte de la subida final puede venir de que el coseno lleva la tasa de aprendizaje al 1 % al final, y no solo de que faltaran épocas.

**Comparación con el entregable 1.** Misma partición, mismo F1 de instancia por imagen en test.

| Modelo | E1: épocas | E1: F1 por imagen | E2: 6 épocas, F1 por imagen |
|---|---|---|---|
| YOLOv8n-seg | 40 | 0,521 | 0,310 |
| Mask R-CNN | 12 | 0,447 | 0,438 |

Con 6 épocas YOLOv8 pierde 0,21 frente al E1; Mask R-CNN casi no cambia. La comparación mezcla épocas con aumentación: en el E1 YOLOv8 usó la aumentación por defecto de ultralytics (mosaico, color y escala), y en el E2 solo volteo horizontal.

**Consecuencia para las conclusiones.** RQ1 y RQ2 valen para un presupuesto de 6 épocas. Es probable que la desventaja de los YOLO y de Mask2Former se reduzca con más entrenamiento (inferido; no se probó). Por eso la diferencia de RQ2 no se atribuye a la arquitectura. La familia de dos etapas y Mask R-CNN ya estaban casi estables, así que no se espera que el empate entre ellos cambie (inferido).

## Lo que no se hizo

- Robustez y latencia de los 9 modelos que corren fuera de `wdsd_dl`.
- Interpretabilidad (mapas de activación o análisis cualitativo de errores).
- Comparación con y sin preentrenamiento en el mejor modelo, y verificación con la partición oficial del paper guía (preregistro §7).
- Recalibración ajustada en validación: se hizo con validación cruzada por grupos sobre test, porque no se guardaron las predicciones de validación.
- Taxonomía de errores con etiquetado manual y techo de desempeño con reanotación.
