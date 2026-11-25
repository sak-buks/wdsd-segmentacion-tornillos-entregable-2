# Ablación y prueba de sanidad

Dos verificaciones sobre el mejor modelo, Mask R-CNN. La **ablación** responde qué parte de su desempeño se debe a cada decisión de diseño. La **prueba de sanidad** comprueba que el flujo de entrenamiento y evaluación no entrega resultados buenos sin que el modelo haya aprendido algo. Ambas son exploratorias: no cambian la elección del modelo (preregistro, sección de análisis exploratorios). Los umbrales de decisión de la sanidad se fijaron antes de correrla (están en el encabezado de `sanidad_e2.py`); los de la ablación no se fijaron antes, así que se reportan como estimaciones con su incertidumbre y no como pruebas con umbral.

## Ablación

**Qué se hizo.** Se quita un componente a la vez y se vuelve a entrenar con las mismas 5 semillas (42–46), las mismas 6 épocas, el mismo lote, la misma tasa elegida en la búsqueda y la misma evaluación que las corridas finales. El modelo completo no se volvió a entrenar: sus 5 corridas finales, hechas en la misma GPU, son la referencia. Son 15 corridas de unos 17 minutos cada una (`ablacion_e2.py`, resultados en `results/e2/ablacion/`).

| Componente que se quita | Qué cambia |
|---|---|
| Anclas pequeñas | Las **anclas** son las cajas de referencia que la red usa como punto de partida para proponer dónde hay un objeto. El modelo completo usa tamaños de 8 a 128 px, elegidos porque el lado mediano de un tornillo es 12 px. Esta variante usa los de torchvision por defecto, de 32 a 512 px. |
| Preentrenamiento en COCO | El modelo completo parte de un detector ya entrenado con COCO (un conjunto grande de imágenes de objetos cotidianos). Esta variante parte solo de un **backbone** (la parte de la red que extrae características de la imagen) entrenado con ImageNet (clasificación de imágenes), y las demás partes empiezan desde cero. |
| Volteo horizontal | Es la única aumentación de datos del protocolo (voltear la imagen a la izquierda o derecha con probabilidad 0,5). Esta variante entrena sin ella. |

Antes de lanzar las 15 corridas se verificó que cada variante construye lo que dice: tamaños de ancla 8–128 contra 32–512 px; en la variante ImageNet, los pesos de la cabeza de propuestas parten de otra inicialización (valor absoluto medio 0,0080 contra 0,0057); y en la última, el volteo está apagado.

**Cómo se mide.** F1 de instancia por imagen promediado por grupo de escena (la métrica primaria), media de las 5 semillas. Para cada variante se calcula la diferencia con el modelo completo en cada uno de los 14 grupos de test, y con esos 14 valores: la diferencia media, su intervalo de confianza del 95 % por bootstrap (2000 remuestreos de grupos), la prueba de Wilcoxon (que compara los rangos de las diferencias, sin suponer normalidad) con corrección de Holm por ser 3 comparaciones, y el tamaño de efecto d de Cohen pareado (diferencia media dividida por su desviación estándar).

**Tabla 6.10.** Ablación de Mask R-CNN (F1 por grupo en test, media de 5 semillas; el modelo completo da 0,726).

| Variante | F1 | Diferencia con el completo | IC 95 % | p Wilcoxon | p Holm | Grupos mejor / peor | d pareada |
|---|---|---|---|---|---|---|---|
| Anclas de 32–512 px | 0,689 | −0,037 | [−0,076, −0,002] | 0,084 | 0,169 | 3 / 9 | −0,51 |
| Solo ImageNet | 0,687 | −0,038 | [−0,073, −0,010] | 0,023 | 0,069 | 3 / 10 | −0,63 |
| Sin volteo | 0,709 | −0,017 | [−0,042, 0,006] | 0,241 | 0,241 | 6 / 8 | −0,36 |

(Los grupos que no suman 14 son los que quedaron empatados.) Por corrida, las diferencias con la corrida final de la misma semilla van de −0,073 a −0,002: ninguna de las 15 corridas supera a su referencia (`ablacion_por_corrida.csv`).

**Lectura (observado).**
- Las tres variantes quedan por debajo del modelo completo, en 8 a 10 de los 14 grupos.
- Quitar las anclas pequeñas o el preentrenamiento en COCO baja el F1 unos 0,04. Esa diferencia es de tamaño medio (d pareada de −0,5 a −0,6) y los IC del 95 % no incluyen 0, pero ninguna sobrevive a la corrección de Holm (la menor, la de ImageNet, da p = 0,069).
- Quitar el volteo baja el F1 0,017 y su IC incluye 0.

**Lectura (inferido).** El orden es el esperable: lo que más ayuda es lo que más se parece a los datos (anclas ajustadas al tamaño de los tornillos) y partir de un detector ya entrenado. Pero con 5 semillas y 14 grupos la evidencia es débil, y no se puede afirmar que algún componente sea imprescindible. El preentrenamiento aporta poco frente al ruido entre semillas (la desviación estándar del F1 por grupo de Mask R-CNN entre sus 5 semillas es 0,011).

**Límites.** Una sola configuración de tasa para todas las variantes (la elegida para el modelo completo, que puede favorecerlo). Solo se probó un componente a la vez: no se mide si dos se compensan. Solo Mask R-CNN.

## Prueba de sanidad con etiquetas permutadas

**Idea.** Si se entrena con etiquetas que no tienen relación con las imágenes, el modelo no puede aprender a encontrar tornillos, y su desempeño en test debe caer a casi cero. Si no cae, algo del flujo filtra información (por ejemplo, test mezclado con train, o una métrica que premia cualquier salida).

**Cómo se hizo.** A cada imagen de train se le asignaron los tornillos anotados de otra imagen de train del mismo tamaño y de otro grupo de escena (dentro de un mismo grupo las fotos son casi iguales y la permutación dejaría etiquetas casi correctas). Val y test no se tocaron. Es una corrida de Mask R-CNN con la semilla 42, 6 épocas y la misma tasa y evaluación que la corrida final (`sanidad_e2.py`). La mejor época y el umbral de score se eligen en val con las etiquetas verdaderas, lo que favorece un poco a la corrida permutada, así que la prueba es conservadora.

**Criterio fijado antes de correr.** Pasa si el F1 por grupo en test de la corrida permutada es ≤ 0,10 (la corrida real da 0,73).

**Tabla 6.11.** Corrida real y permutada (semilla 42).

| | Val | Test | Train (93 imágenes) |
|---|---|---|---|
| F1 por grupo, corrida real | 0,795 | 0,732 | 0,880 |
| F1 por grupo, etiquetas permutadas | 0,000 | 0,000 | 0,000 |
| AP50 (test), real / permutada | | 0,601 / 0,000 | |

**Resultado (observado): pasa.** El F1 en val, test y train es exactamente 0, con AP de 0 en todas las épocas. La pérdida de entrenamiento se estanca cerca de 0,50 (de 0,60 en la época 1 a 0,50 en la 6) y, como el F1 en val es 0 con cualquier umbral, la regla de selección deja el mínimo permitido (0,05): ninguna detección coincide con una etiqueta verdadera. Un modelo que dejara pasar información de test no daría 0.

**Lo que no prueba.** Que no haya otra fuga: por ejemplo, la que viene del preentrenamiento en COCO o ImageNet no se mide aquí. La auditoría de fuga entre particiones está en el capítulo siguiente.
