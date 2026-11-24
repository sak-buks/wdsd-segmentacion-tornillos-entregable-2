# Criterios a priori: paper guía y auditoría de fuga (E2)

Escrito el 2026-11-24, antes de correr `paper_guia_e2.py` y `auditoria_fuga_e2.py`. La fecha que cuenta es la del commit que agrega este archivo. Completa el preregistro §7 («Paper guía y su limitación»), que dejó la verificación como exploratoria.

## Réplica parcial (enunciado §5)

- **Resultado que se replica:** Kalitsios et al. 2022 (DOI 10.1109/ICRCV55858.2022.9953178), Mask R-CNN ResNet-101 en WDSD con la partición oficial (765/90/90): AP de máscara 0,586 y AP50 0,849 en test.
- **Nuestra réplica:** Mask R-CNN ResNet-101 de MMDetection preentrenado en COCO, partición oficial, misma métrica (AP de máscara COCO en las 90 imágenes de test). Protocolo del E2: 6 épocas, lote efectivo 2, tasa por defecto, anclas de 8–128 px, semillas 42, 43 y 44.
- **Diferencias conocidas con el paper:** número de épocas, tamaño de lote, programa de tasa y anclas. Se declaran como causas posibles de cualquier diferencia.
- **Métrica principal de la réplica:** AP50, porque es la menos sensible a la precisión del borde. El AP se reporta como secundaria.
- **Criterio de éxito:** sea D = AP50 nuestro (media de semillas) − 0,849, con IC 95 % bootstrap de 2000 remuestreos de las imágenes de test.
  - Réplica lograda si |D| ≤ 0,05.
  - Réplica parcial si 0,05 < |D| ≤ 0,10.
  - Réplica fallida si |D| > 0,10.
  - En los tres casos se reporta D con su IC y se analizan las causas.

## Limitación del paper (enunciado §5, verificada en ≥ 3 modelos del Top 10)

- **Limitación:** la partición oficial es aleatoria por imagen. Si las imágenes de test tienen casi gemelos en train, su desempeño está inflado por fuga.
- **Medición:** MS R-CNN, PointRend y YOLO26, entrenados con la partición oficial (mismo protocolo, tasa elegida en la búsqueda del E2, semillas 42–44). Para cada modelo se compara el F1 de instancia por imagen (media de semillas) entre las imágenes de test con casi gemelo en train y las imágenes sin él.
- **Casi gemelo:** pHash de 64 bits con distancia de Hamming ≤ 10 a alguna imagen de train (definido en `auditoria_fuga_e2.py`).
- **Criterio:** la limitación se confirma si en al menos 2 de los 3 modelos la diferencia (con gemelo − sin gemelo) es positiva y su IC 95 % bootstrap por grupo de escena (2000 remuestreos) excluye 0. Además se reporta el AP de cada subconjunto y la comparación descriptiva con el F1 de test en la partición agrupada.

## Auditoría de fuga de la partición agrupada (enunciado §10.1)

| Verificación | Criterio de aprobación |
|---|---|
| Intersección de grupos de escena | Vacía en las tres parejas (train–val, train–test, val–test). |
| Duplicados exactos (SHA-256) | Cero colisiones entre particiones. |
| Casi duplicados (pHash, Hamming ≤ 10) | A lo sumo el 5 % de test con casi gemelo en train. Se reporta además el F1 con y sin esas imágenes. |
| Validación adversarial (regresión logística, validación cruzada por grupo) | AUC no distinguible de 0,5 (p de permutación ≥ 0,05), o diferencia explicada. |
