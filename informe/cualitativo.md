# Análisis cualitativo de predicciones

**Qué se muestra.** Cuatro imágenes de test con las predicciones de Mask R-CNN (mejor baseline), HTC (mejor del Top 10) y YOLOv8 (familia de prototipos), semilla 42, cada uno con su umbral de score elegido en val. Las imágenes se eligieron por el F1 medio de los 13 modelos: una fácil, una con muchos tornillos donde los modelos difieren, una intermedia y una del grupo 9, que ningún modelo resuelve. El emparejamiento es el de la métrica: húngaro, IoU de máscara ≥ 0,5 (`figuras_cualitativas_e2.py`, resultados en `results/e2/cualitativo/`).

```{figure} ../results/e2/cualitativo/predicciones_cualitativas.png
:width: 100%
:align: center

**Figura 6.7.** Verde: verdadero positivo (VP). Rojo: falso positivo (FP). Círculo amarillo: tornillo anotado no detectado (FN). Cada panel se recorta alrededor de los tornillos anotados; los FP fuera del recorte se cuentan en el título pero no se ven. F1 = 2·VP / (2·VP + FP + FN).
```

**Observado.**
- **Fácil** (`Micro_2_cam3_png_Color_583`, 5 tornillos): los tres aciertan todo (F1 1,00).
- **Muchos tornillos** (`screws_76`, 11): Mask R-CNN acierta todo y HTC suma un FP. YOLOv8 encuentra 10 pero añade 11 FP: con su umbral de val (0,10, frente a 0,75 de Mask R-CNN) acepta detecciones de baja confianza sobre agujeros y bordes. F1 = 2·10 / (2·10 + 11 + 1) = 0,625.
- **Intermedia** (`pc_12_661`, 3): Mask R-CNN y HTC aciertan los 3, pero marcan 5 FP, varios sobre tornillos visibles del borde izquierdo de la carcasa que no están anotados. YOLOv8 marca 25 FP, incluidos los dos conectores redondos (PS/2).
- **Grupo 9** (`eco12_363`, 2): nadie acierta. Los tornillos son planos, del mismo color que la placa y con reflejos. HTC pone una máscara sobre uno de ellos, pero su IoU con la anotación queda por debajo de 0,5 y cuenta a la vez como FP y como FN.

**Inferido (no probado).**
- Parte de los FP de los modelos de dos etapas podrían ser tornillos reales sin anotar. Si es así, el F1 de todos está subestimado, y es un argumento para estimar el techo de anotación (re-anotación pendiente).
- El problema de YOLOv8 aquí no es encontrar los tornillos (los encuentra) sino separar tornillos de otros objetos pequeños y redondos, coherente con su F1 más bajo en tornillos pequeños (RQ3) y con el subentrenamiento a 6 épocas.
- El grupo 9 falla por contraste: los tornillos casi no se distinguen del fondo. Eso apunta al dato, no a la arquitectura (los 13 modelos fallan ahí).

**Límites.** Cuatro imágenes elegidas por nosotros, una semilla: ilustran, no prueban. La taxonomía de errores con ≥ 100 casos (enunciado §12.3) está pendiente.
