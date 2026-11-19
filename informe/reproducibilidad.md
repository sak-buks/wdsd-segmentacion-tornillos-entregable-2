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

## Pendiente

Prueba de entorno limpio (reproducir la tabla principal en una máquina nueva y reportar la discrepancia), Dockerfile opcional y la lista de verificación del Apéndice B del enunciado.
