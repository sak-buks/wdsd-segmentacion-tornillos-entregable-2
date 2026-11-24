# Trazabilidad hallazgo–decisión del EDA

Plantilla A.1 del enunciado. El EDA es el del E1 (notebook 02 del repositorio del E1). Esta tabla une cada hallazgo con la decisión que tomamos en el E2 por su causa.

| Hallazgo | Evidencia (figura, prueba) | Decisión de preprocesamiento | Impacto en arquitectura o validación | Riesgo y fuga |
|---|---|---|---|---|
| Objetos muy pequeños: diámetro equivalente mediano 10,6 px, el 98,8 % es "pequeño" según COCO (área < 32² px) | Histograma de tamaños, EDA del E1 §5 | Entrenar a resolución completa (1280 × 720), sin reducir la imagen | Anclas de 8 a 128 px en los modelos de dos etapas (ablación: −0,037 de F1 con las de defecto); RQ3 por tercil de tamaño; IoU ≥ 0,5 como umbral principal porque con ~12 px un error de 1–2 px de borde baja mucho el IoU | El AP de 0,5 a 0,95 castiga mucho los bordes: se reporta, pero no es la métrica primaria |
| Las fotos de una misma escena son casi iguales (mismo producto, mesa y luz) | Inspección de grupos; pHash | Partición agrupada por escena (744/93/108), heredada del E1 | Métricas promediadas por grupo; bootstrap y Wilcoxon sobre los 14 grupos de test, no sobre imágenes | Partición por imagen = fuga. Auditoría: ningún grupo cruza, pero el 12,0 % de test tiene un casi gemelo de otra escena en train |
| Pocos grupos de test (14), 8 de ellos con 1 o 2 imágenes | Conteo por grupo | — | Potencia calculada con 14 grupos (diferencia mínima detectable 0,119); se reportan IC y tamaños de efecto | Diferencias pequeñas entre modelos no son detectables (declarado en amenazas) |
| 0 % de instancias que se tocan o se solapan | Conteo de solapes entre máscaras | Sin tratamiento especial de oclusión | Emparejamiento húngaro uno a uno con IoU de máscara; segmentación por instancias en vez de semántica más componentes conexos | Bajo |
| Una sola clase y todas las imágenes tienen tornillos | Conteo de anotaciones (4398 instancias) | Sin imágenes vacías que manejar | F1 de instancia como métrica primaria (no hace falta promediar clases) | El F1 no mide falsos positivos en imágenes sin objetos: no se puede evaluar ese caso |
| Escenas distintas pero muy parecidas entre sí | pHash, Hamming ≤ 10 (auditoría, cap. 5.2) | — | F1 recalculado sin las 13 imágenes con casi gemelo (baja ~0,07; el orden entre modelos casi no cambia) | Posible inflado del F1 absoluto; no afecta las comparaciones entre modelos |
