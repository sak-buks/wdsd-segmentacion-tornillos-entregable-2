# Alcance y pendientes

Estado del informe al 2026-11-18. Este libro presenta lo que ya está hecho y verificado; lo que falta se lista aquí con el mismo criterio con que el enunciado del E2 lo exige, para que ningún faltante quede implícito.

## Hecho (con evidencia en el repositorio)

- Revisión de literatura, PRISMA y Top 10 (`literatura/e2/`).
- Preregistro (`preregistro_E2.md`), prueba de humo y piloto de tiempo (`results/e2/prueba_humo/`).
- Búsqueda de hiperparámetros: 42 corridas, 14 modelos × 3 tasas (`results/e2/busqueda.csv`).
- Corridas finales: 70 de 70 (14 modelos × semillas 42–46) más la máscara vacía, sin fallos técnicos, todas en la misma GPU (`results/e2/corridas.csv`).
- Análisis estadístico de RQ1 y RQ2, Friedman con Nemenyi y todos los pares con Benjamini–Hochberg (`analisis_e2.py`, `results/e2/analisis/`, notebook de resultados).

## Pendiente

| Tema | Estado |
|---|---|
| Hoja de extracción y valoración de calidad de los 64 estudios incluidos (§3.1 del enunciado) | No existe |
| Referencia con DOI, diagrama, formulación e innovación de cada modelo; FLOPs; matriz comparativa; hueco de investigación (§3.3 y §3.4) | No existe |
| RQ3 (F1 por tercil de área del tornillo, exploratoria) y RQ4 (brecha train − val con IC por grupo) | No calculadas; los datos por imagen de val y train están en `results/e2/corridas/` si se copiaron |
| Curvas de entrenamiento y validación por modelo, latencia de inferencia en una sola GPU | No hechas |
| Tabla de preentrenamiento por modelo y declaración de que Cascade, MS R-CNN, PointRend, Mask2Former y CondInst corren en MMDetection y no en el código de sus autores | Pendiente de redactar |
| Auditoría de fuga (§10), techo e incertidumbre (§12), paper guía con réplica parcial (§5) | No hechas |
| Prueba de entorno limpio, Dockerfile opcional y checklist del Apéndice B | No hechas |
| Presentación oral | No hecha |

## Qué se sabe que limita las conclusiones

- Mask2Former tuvo una corrida inestable (semilla 46). Se reporta, no se excluye: el preregistro solo permite excluir por fallo técnico.
- Con δ = 0.20, heredado del E1 y fijado allí después de ver resultados, la regla de decisión llama «equivalentes» a modelos que son peores. Por eso las tablas muestran también la dirección y el IC de cada diferencia.
- Familia, marco de software y presupuesto de entrenamiento están mezclados en la comparación de familias (RQ2); la prueba no identifica la causa.
- Seis épocas para todos los modelos: en varios la mejor época fue la última, y se declara subentrenamiento según la regla de parada del preregistro.
