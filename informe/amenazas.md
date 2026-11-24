# Amenazas a la validez

Una **amenaza a la validez** es una razón por la que una conclusión podría ser falsa aunque los números estén bien calculados. Se agrupan en cuatro tipos. Para cada una: qué es, qué hicimos y qué queda abierto.

## Validez interna (¿la diferencia se debe al modelo?)

| Amenaza | Qué se hizo | Qué queda |
|---|---|---|
| Presupuesto de entrenamiento igual (6 épocas) que favorece a los modelos que convergen rápido | Se midió la tendencia de la última época | YOLO y Mask2Former siguen subiendo en 4 de 5 semillas: RQ1 y RQ2 valen para ese presupuesto |
| Hiperparámetros por defecto que favorecen a unos modelos | Misma búsqueda para todos: 3 multiplicadores de la tasa por defecto, 2 épocas, una semilla, elegida en val | Solo se buscó la tasa; lo demás es lo de cada marco |
| La familia se confunde con marco de software, lote y preentrenamiento | Se declara en la conclusión 2 | No se puede atribuir la ventaja a la arquitectura |
| Fuga entre particiones | Grupos y SHA-256 sin cruce; etiquetas permutadas dan F1 0 | 12,0 % de test con casi gemelo (falla el límite de 5 %); el F1 absoluto puede estar inflado ~0,07 |
| Fuga por preentrenamiento (COCO, ImageNet) | Solo se declara | No medida |
| Selección del umbral y la época en val | Regla única para todos, fijada en el preregistro | Val tiene solo 93 imágenes |

## Validez estadística (¿la inferencia es correcta?)

- **Pocos grupos.** Con 14 grupos de test solo se detectan diferencias grandes; "no significativo" no es "igual". Se reportan IC y tamaños de efecto, no solo p.
- **Muchas comparaciones.** Holm en los contrastes confirmatorios y Benjamini–Hochberg en RQ3. Los exploratorios no se corrigen entre capítulos: se leen como hipótesis.
- **Semillas.** 5 semillas en el benchmark, 3 en interpretabilidad y paper guía, 1 en 9 modelos de robustez y latencia. Mask2Former tuvo una semilla colapsada y se mantuvo.
- **Umbrales no fijados antes** en la ablación: se reporta como estimación.

## Validez de constructo (¿la métrica mide lo que importa?)

- El F1 con IoU ≥ 0,5 es poco exigente con el borde; el AP (0,5–0,95) da una imagen distinta (réplica: AP50 0,737, AP 0,288).
- El F1 por grupo pesa igual cada escena, no cada tornillo.
- Errores de anotación: no medimos el techo de anotación (re-anotación pendiente), así que parte de los "errores compartidos" podría ser ruido de etiqueta.
- Los mapas de oclusión (80 px) localizan la zona, no el rasgo.

## Validez externa (¿se generaliza?)

- Un solo dataset (WDSD), una clase, escenas de laboratorio con fondo y luz parecidos.
- Variantes pequeñas de cada modelo (por ejemplo YOLO nano) y una sola GPU (RTX 3060): latencia y ranking pueden cambiar con otro tamaño de modelo o hardware.
- La réplica del paper guía falla, así que nuestros números absolutos no son comparables con los publicados.
