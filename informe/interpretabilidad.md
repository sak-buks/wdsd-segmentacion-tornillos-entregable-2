# Interpretabilidad de los tres mejores modelos

**Qué se pregunta.** ¿En qué zonas de la imagen se apoyan Mask R-CNN, HTC y Cascade para detectar tornillos? ¿El mapa que lo muestra es fiel a lo que hace el modelo? ¿Cambia de una semilla a otra? Es un análisis exploratorio: no cambia la elección de modelo (`interpretabilidad_e2.py`, resultados en `results/e2/interpretabilidad/`).

## Método

**Sensibilidad a la oclusión.** Se divide la imagen (1280 × 720 px) en una rejilla de 9 × 16 celdas de 80 × 80 px. Se tapa una celda a la vez con el color medio de la imagen y se mide cuánto del resultado original sobrevive. Es una técnica de caja negra: solo usa entradas y salidas, no los gradientes, así que sirve igual para modelos de marcos de software distintos. Se eligió una técnica de caja negra, y no un mapa de gradientes como Grad-CAM, porque Mask R-CNN (torchvision) y HTC y Cascade (MMDetection) no comparten una capa equivalente donde calcularlo, y así los tres se tratan exactamente igual.

**Qué se mide al tapar.** Se usa la idea de D-RISE (Petsiuk et al., 2021), pensada para detectores. Para la imagen original se toman las detecciones con score sobre el umbral elegido en validación. Para la imagen con una celda tapada, cada una de esas detecciones se empareja con la detección nueva que más se le parece (IoU de máscara ≥ 0,5) y se suma el score de esas detecciones nuevas; si ninguna se parece, aporta 0. Sea f0 esa suma sin tapar. El valor de la celda es (f0 − f con la celda tapada) / f0: la fracción del resultado que depende de esa zona. Vale 0 si taparla no cambia nada y 1 si el modelo deja de detectar lo que detectaba.

**Imágenes y semillas.** Una imagen de test por grupo de escena (14), elegida al azar con semilla 0 entre las que tienen tornillos anotados, y las semillas 42, 43 y 44 de cada modelo. Con una imagen por grupo, remuestrear imágenes equivale a remuestrear grupos. Si un modelo no detecta nada sobre el umbral en una imagen y semilla, esa combinación se omite (en la Figura 6.6, Cascade con la semilla 42 en la imagen 74).

**Fidelidad.** Un mapa es fiel si las zonas que marca como importantes son de verdad las que el modelo necesita. Se comprueba con la curva de borrado (Petsiuk et al., 2018, RISE): se tapan las celdas de más a menos importantes, de 9 en 9 (hasta cubrir toda la imagen en 16 pasos), y se mide qué fracción de f0 sobrevive tras cada paso. El área bajo esa curva (AUC de borrado, con la fracción de imagen tapada de 0 a 1) es baja si el mapa es fiel. Se compara con tapar en orden aleatorio (3 órdenes, los mismos para todos los modelos); la **ganancia de fidelidad** es el AUC aleatorio menos el AUC del mapa, y es positiva si el mapa es mejor que el azar. Intervalo de confianza del 95 % por bootstrap sobre las 14 imágenes (2000 remuestreos).

**Estabilidad entre semillas.** Para la misma imagen, se compara el mapa del mismo modelo con semillas distintas con la correlación de Spearman (compara los órdenes de importancia de las 144 celdas) y con el solapamiento de Jaccard del 10 % de celdas más importantes (celdas en común dividido por celdas en total entre los dos conjuntos de 14 celdas). Como referencia se usa la correlación entre imágenes distintas, que solo refleja la disposición típica de la escena, y la correlación entre modelos distintos con la misma imagen y semilla.

**Plausibilidad (extra).** Qué parte del valor positivo del mapa cae en celdas que contienen algún tornillo anotado, frente a la proporción de celdas con tornillo (5,2 % en estas imágenes).

## Resultados

**Tabla 6.16.** Fidelidad: AUC de la curva de borrado (más bajo = más fiel) y ganancia sobre el orden aleatorio.

| Modelo | AUC de borrado, orden del mapa | AUC de borrado, orden aleatorio | Ganancia | IC 95 % | Imágenes con ganancia > 0 |
|---|---|---|---|---|---|
| Mask R-CNN | 0,035 | 0,387 | 0,353 | [0,280, 0,422] | 14 de 14 |
| HTC | 0,035 | 0,387 | 0,353 | [0,287, 0,414] | 14 de 14 |
| Cascade | 0,034 | 0,379 | 0,345 | [0,277, 0,405] | 14 de 14 |

```{figure} ../results/e2/interpretabilidad/curvas_borrado.png
:width: 100%
:align: center

**Figura 6.5.** Fracción del resultado que sobrevive al tapar la imagen, celdas en orden del mapa (azul) y en orden aleatorio (naranja), media de 14 imágenes y 3 semillas.
```

**Tabla 6.17.** Estabilidad entre semillas y referencias.

| Modelo | Spearman entre semillas | IC 95 % | Jaccard del 10 % superior | Spearman entre imágenes distintas |
|---|---|---|---|---|
| Mask R-CNN | 0,378 | [0,313, 0,452] | 0,609 | 0,015 |
| HTC | 0,376 | [0,305, 0,439] | 0,604 | 0,012 |
| Cascade | 0,413 | [0,336, 0,481] | 0,699 | 0,026 |

Entre modelos (misma imagen y semilla), el Spearman medio es 0,258 (HTC–Cascade), 0,249 (Mask R-CNN–Cascade) y 0,214 (Mask R-CNN–HTC).

**Tabla 6.18.** Plausibilidad: parte del valor positivo del mapa que cae en celdas con tornillo anotado.

| Modelo | Parte en celdas con tornillo | Proporción de celdas con tornillo | Razón |
|---|---|---|---|
| Mask R-CNN | 80,4 % | 5,2 % | 15,4 |
| HTC | 77,1 % | 5,2 % | 14,8 |
| Cascade | 74,7 % | 5,2 % | 14,3 |

```{figure} ../results/e2/interpretabilidad/ejemplos_mapas.png
:width: 100%
:align: center

**Figura 6.6.** Tres imágenes de test (grupos 6, 12 y 27) con los tornillos anotados en verde y el mapa de oclusión de cada modelo (semilla 42; más claro = el resultado depende más de esa celda; solo se pintan las celdas con valor ≥ 0,02). En la imagen 74, Cascade no detectó nada sobre el umbral y no tiene mapa.
```

## Lectura

**Observado.**
- **Los tres modelos dependen de las celdas donde hay tornillos.** Tapar en el orden del mapa borra casi todo el resultado con el primer paso (9 de 144 celdas, 6 % de la imagen); en orden aleatorio hace falta tapar mucho más (el AUC pasa de 0,035 a unos 0,38). La ganancia es de 0,35 en los tres, con IC que no incluyen 0 y en las 14 imágenes. No hay diferencia entre modelos: los IC se solapan casi por completo.
- **Entre el 75 % y el 80 % del mapa cae en el 5 % de celdas con tornillo** (14 a 15 veces lo que daría el azar).
- **Entre semillas, los mapas coinciden de forma moderada.** El Spearman es 0,38–0,41, muy por encima del 0,01–0,03 entre imágenes distintas, pero lejos de 1. El 10 % de celdas más importantes se solapa entre el 60 % y el 70 %. Cascade sale algo más estable (0,41 y 0,70) pero su IC se solapa con el de los otros dos.
- **Entre modelos distintos coinciden menos que entre semillas del mismo modelo** (Spearman 0,21–0,26 contra 0,38–0,41).

**Inferido (no probado).**
- La fidelidad alta es poco informativa por sí sola: es casi inevitable que un modelo que busca tornillos de ~12 px deje de detectar un tornillo cuando se tapa la celda donde está. Lo que el resultado dice es que los modelos no dependen de una zona de contexto cualquiera. No dice qué rasgo del tornillo usan (borde, brillo, sombra), porque una celda de 80 × 80 px es mucho mayor que un tornillo.
- El 20–25 % del mapa fuera de celdas con tornillo anotado puede deberse a contexto cerca del tornillo, a que un tornillo cruza el borde de dos celdas, o a detecciones sobre el umbral que no corresponden a un tornillo anotado (falsos positivos). Esta última posibilidad no se separó.
- El Spearman entre semillas es moderado en parte porque la mayoría de las 144 celdas valen casi 0 en todas las semillas, y el orden entre celdas casi iguales es ruido. El solapamiento del 10 % superior es la medida más directa de estabilidad, y es de 0,6 a 0,7.
- Los modelos «miran» sitios parecidos pero no idénticos: coherente con que sus errores estén muy correlacionados en el capítulo anterior (Spearman medio 0,82 en errores por imagen), aunque aquí se miden mapas, no errores.

## Límites
- 14 imágenes, una por escena y elegidas al azar: no se escogieron a propósito los casos más difíciles (por ejemplo, los tornillos que todos los modelos pierden), y el análisis solo mira las detecciones que el modelo sí hace, no lo que deja de detectar.
- Rejilla gruesa (80 px) frente a tornillos de ~12 px: solo localiza la zona, no el rasgo.
- Tapar con el color medio crea un borde artificial; es una elección estándar pero que puede afectar a los modelos de forma distinta.
- Solo los tres modelos de mejor F1, con las semillas 42–44 y no las 5.
- No se comparó con otras técnicas (Grad-CAM, RISE completo con máscaras aleatorias), así que no se sabe cuánto de lo observado depende de la técnica.
