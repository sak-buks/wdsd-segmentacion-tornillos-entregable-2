# Reproducibilidad

## Qué se puede regenerar sin GPU

Las tablas y figuras de resultados salen de `results/e2/corridas/<modelo>_s<semilla>/test_por_imagen.csv` y `results/e2/corridas.csv`, ya versionados:

```bash
pip install -r requirements_analisis.txt
python analisis_e2.py results/e2/corridas results/e2/corridas.csv results/e2/analisis
python analisis_rq4.py results/e2/corridas results/e2/analisis
jupyter-book build .
```

El script comprueba el cálculo: recalcula el F1 por grupo desde los CSV por imagen y lo compara con la columna `test_F1_grupo` de `corridas.csv` (diferencia máxima 1e-16).

## Entornos

| Entorno | Modelos | Archivo de dependencias |
|---|---|---|
| `wdsd_dl` | YOLO11, YOLO26, YOLOv8, Mask R-CNN y el baseline clásico | `requirements_wdsd_dl_congelado.txt` |
| `wdsd_mmdet` | Mask2Former, Cascade, HTC, PointRend, MS R-CNN, CondInst, SOLOv2 | `requirements_wdsd_mmdet_congelado.txt` |
| `wdsd_yolo9` | YOLOv7 y YOLOv9 (repos oficiales parchados) | `requirements_wdsd_yolo9_congelado.txt` |

Los parches de los repos oficiales y los commits usados están en `results/e2/prueba_humo/repos_oficiales/README.md`.

## Entrenamiento

1. Obtener el dataset (ver el README del repositorio).
2. `python entrenar_e2.py configurar` y `datos`; luego `piloto`, `buscar`, `final` y `resumen`, con los comandos de [Protocolo de entrenamiento](../ENTRENAMIENTO_E2).
3. Los resultados quedan en `results/e2/corridas/` y `results/e2/busqueda/`.

## Qué deja cada corrida

| Archivo | Contenido |
|---|---|
| `resumen.json` | Configuración, entorno, tiempos, memoria y métricas de val, test y train |
| `test_por_imagen.csv`, `val_por_imagen.csv`, `train93_por_imagen.csv` | F1, PQ y conteos por imagen |
| `historial.csv` | Pérdida y AP de validación por época |
| `umbral_val.csv` | Umbral de score elegido en validación |
| `preds_*.json` | Predicciones en formato COCO (las de más de 5 MB no se versionan) |

Los pesos entrenados no se versionan.

## Prueba de entorno limpio

**Qué es.** El enunciado (§8.1) pide ejecutar el proyecto desde cero en un entorno nuevo y reproducir la tabla principal, reportando la discrepancia máxima. Sirve para comprobar que el resultado no depende de algo que solo existe en el computador donde se hizo (un archivo sin versionar, una librería instalada a mano).

**Cómo se hizo (2026-11-24).** En una máquina distinta (Linux, contenedor en la nube; las corridas se hicieron en Windows):
1. `git clone` del repositorio público, commit `283a6d9`.
2. Entorno virtual nuevo con Python 3.13 y `pip install -r requirements_analisis.txt` (versiones exactas).
3. `python analisis_e2.py results/e2/corridas results/e2/corridas.csv <carpeta nueva>` y `python analisis_rq4.py results/e2/corridas <carpeta nueva>`.
4. Comparación celda a celda de los 9 CSV generados con los versionados en `results/e2/analisis/` (tabla principal, F1 por semilla y por grupo, RQ1, pares, Friedman, RQ4, subentrenamiento).

**Resultado.** Discrepancia máxima 0 en todas las celdas numéricas y 0 celdas de texto distintas. El análisis tarda 4 s.

**Qué no cubre.** Parte de las predicciones guardadas de cada corrida, no re-entrena: re-entrenar los 70 modelos tomaría unas 20 horas de GPU, y como MMDetection no corre en modo determinista, un re-entrenamiento no daría exactamente los mismos números. La variación esperable por esa causa es del orden de la que hay entre semillas (desviación estándar del F1 por grupo de Mask R-CNN: 0,011).

## Lista de verificación del Apéndice B

✅ cumple · 🟡 cumple en parte · ❌ no cumple · — no aplica.

| Ítem del enunciado | Estado | Evidencia o motivo |
|---|---|---|
| Preregistro fechado y versionado antes del primer entrenamiento | ✅ | Tag `e2-preregistro` del repo del E1: commit 2026-11-17 00:10, primera corrida de búsqueda 00:24 (hora de Colombia) |
| Hash del dataset y del conjunto de prueba registrados | ✅ | SHA-256 del dataset y de train, val y test en el preregistro |
| Auditoría de fuga ejecutada y anexa, con todos los criterios aprobados | 🟡 | Ejecutada (cap. 5.2). Falla un criterio: 12,0 % de test con casi gemelo, con límite de 5 % |
| Particiones y semillas guardadas y regenerables | ✅ | Partición agrupada del E1 (744/93/108) y semillas 42–46 en `resumen.json` de cada corrida |
| Un único pipeline por modelo; ningún ajuste fuera de train | ✅ | `entrenar_e2.py`; tasa, época y umbral elegidos en val |
| Mismo presupuesto de ajuste; configuraciones registradas | ✅ | 3 valores de tasa por modelo y 2 épocas, en `results/e2/busqueda/` |
| ≥ 5 ejecuciones por modelo (6 con pruebas por rangos) y resultados por ejecución | 🟡 | 5 semillas por modelo, con resultados por corrida. Se usa Wilcoxon (prueba por rangos) con 5, no con 6; el contraste es por grupo (n = 14), no por semilla |
| Análisis de potencia y justificación del número de ejecuciones | ✅ | Preregistro §6 (diferencia mínima detectable 0,119 con 14 grupos) |
| Baselines trivial y clásico fuerte, y los tres modelos base del E1, con el mismo presupuesto | ✅ | `vacio`, baseline clásico, Mask R-CNN, YOLOv8 y SOLOv2 |
| Versiones de software, hardware y determinismo documentados | 🟡 | Entornos congelados y commits de los repos oficiales. Hardware: RTX 3060 y 3050. El determinismo en GPU no está garantizado: ultralytics corre en modo determinista y MMDetection no (`deterministic=False`) |
| Pruebas de las Tablas A, B y C con p exactos, ajustados, IC y tamaños de efecto | 🟡 | Todos los valores están en el cap. 4; falta el formato de la plantilla A.3 |
| Familias de comparaciones y corrección múltiple declaradas a priori | ✅ | Preregistro §5: Holm con m = 10 para RQ1, Benjamini–Hochberg para RQ3 |
| Techo de desempeño estimado y distancia γ | ❌ | Requiere re-anotar una muestra (pendiente del equipo) |
| Modelos con arquitectura no definida (búsqueda en dos fases) | — | Los 13 modelos tienen arquitectura publicada |
| Arquitectura ganadora elegida solo con validación | — | Igual que el anterior; el mejor baseline se eligió con val |
| Robustez con ≥ 3 niveles; interpretabilidad con fidelidad y estabilidad | ✅ | Cap. 5 (13 modelos, 3 niveles) y cap. 5.3 |
| Incertidumbre en los tres mejores con cobertura empírica | ✅ | Intervalo conforme por grupos, cobertura 0,90–0,94 |
| Un solo acceso a test por ejecución, con modelo y umbrales ya fijados | ✅ | Época y umbral se eligen en val antes de evaluar test |
| Réplica parcial del paper guía con criterio a priori y limitación | ✅ | Cap. 5.2. La réplica falla según el criterio, y se reporta |
| Reproducción independiente de la tabla principal con discrepancia máxima | 🟡 | Prueba de entorno limpio (abajo): discrepancia máxima 0 al regenerar desde los CSV por corrida. No se re-entrenó |
| Código de extremo a extremo con un solo comando y entorno reproducible | 🟡 | El análisis se regenera con tres comandos; el entrenamiento va por pasos. No hay Dockerfile |
