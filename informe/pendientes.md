# Alcance y pendientes

Estado del informe al 2026-11-24. Este libro presenta lo que ya está hecho y verificado; lo que falta se lista aquí con el mismo criterio con que el enunciado del E2 lo exige, para que ningún faltante quede implícito.

## Hecho (con evidencia en el repositorio)

- Revisión de literatura, PRISMA y Top 10 (`literatura/e2/`).
- Preregistro (`preregistro_E2.md`), prueba de humo y piloto de tiempo (`results/e2/prueba_humo/`).
- Búsqueda de hiperparámetros: 42 corridas, 14 modelos × 3 tasas (`results/e2/busqueda.csv`).
- Corridas finales: 70 de 70 (14 modelos × semillas 42–46) más la máscara vacía, sin fallos técnicos, todas en la misma GPU (`results/e2/corridas.csv`).
- Análisis estadístico de RQ1 y RQ2, Friedman con Nemenyi y todos los pares con Benjamini–Hochberg (`analisis_e2.py`, `results/e2/analisis/`, notebook de resultados).
- RQ4 (brecha entre entrenamiento y validación con IC por grupo, con dos definiciones de F1), curvas por época y declaración de subentrenamiento (`analisis_rq4.py`, secciones 4.8 y 4.9).
- Análisis exploratorios (capítulo 5): RQ3 por tercil de área, calibración del score, robustez a ruido y desenfoque, latencia y FLOPs (estos dos últimos solo en 4 modelos), costo de entrenamiento de todos y tamaños de efecto de RQ1. Además: calibración completa (ECE adaptativo, Brier, pendiente e intercepto, temperature scaling y Platt), índice de robustez, δ de Cliff y probabilidad de superioridad, descomposición de la varianza con modelo mixto, frontera de Pareto, errores compartidos con ensambles, e incertidumbre predictiva con cobertura en los tres mejores (`analisis_extra2_e2.py`).
- Registro de la búsqueda, configuración elegida y entornos de las 70 corridas, con la regla de selección de tasa verificada (capítulo 3.2).
- Fecha del preregistro: tag `e2-preregistro` en el repositorio del E1, con el mismo hash de archivo que este repositorio (portada).

## Pendiente

| Tema | Estado |
|---|---|
| Hoja de extracción y valoración de calidad de los 64 estudios incluidos (§3.1 del enunciado) | No existe |
| Referencia con DOI, diagrama, formulación e innovación de cada modelo; FLOPs; matriz comparativa; hueco de investigación (§3.3 y §3.4) | No existe |
| Latencia, FLOPs y robustez de los 9 modelos fuera del entorno `wdsd_dl`; interpretabilidad | No hechas (calibración, y robustez y latencia de 4 modelos, sí: capítulo 5) |
| Tabla de preentrenamiento por modelo (la declaración de que Cascade, MS R-CNN, PointRend, Mask2Former y CondInst corren en MMDetection ya está en el README) | Tabla pendiente de redactar |
| Resumen estructurado y discusión de amenazas a la validez | No hechos |
| Auditoría de fuga (§10), techo de desempeño con reanotación y taxonomía de errores (§12), paper guía con réplica parcial (§5), ablación (§4.6) | No hechas (la incertidumbre predictiva y los errores compartidos sí: capítulo 5) |
| Recalibración ajustada solo en validación | Hecha con validación cruzada por grupos sobre test, porque no se guardaron las predicciones de validación (desviación declarada en el capítulo 5) |
| Prueba de entorno limpio, Dockerfile opcional y checklist del Apéndice B | No hechas |
| Presentación oral | No hecha |

## Qué se sabe que limita las conclusiones

- Mask2Former tuvo una corrida inestable (semilla 46). Se reporta, no se excluye: el preregistro solo permite excluir por fallo técnico.
- Con δ = 0.20, heredado del E1 y fijado allí después de ver resultados, la regla de decisión llama «equivalentes» a modelos que son peores. Por eso las tablas muestran también la dirección y el IC de cada diferencia.
- Familia, marco de software y presupuesto de entrenamiento están mezclados en la comparación de familias (RQ2); la prueba no identifica la causa.
- Seis épocas para todos los modelos: YOLOv8, YOLO11, YOLO26 y Mask2Former siguen mejorando en la última época y se declaran subentrenados (sección 4.9).
