# WDSD: benchmark de 10 modelos de segmentación de instancias de tornillos, entregable 2 de Deep Learning

**Autores:** Ivan Berrio, Jose David Castillo y Jaime Besada.

Segmentación de instancias de tornillos en el WEEE Disassembly Screw Dataset (WDSD): cada tornillo es un objeto que hay que encontrar y delimitar. El entregable 2 compara un Top 10 de modelos de la literatura contra los tres modelos base del [entregable 1](https://github.com/BESADAJ/wdsd-segmentacion-tornillos-entregable-1), un baseline no profundo y uno trivial, con preregistro, 5 semillas por modelo y pruebas estadísticas por grupo de escena. **El entregable está en construcción:** el entrenamiento y el análisis de RQ1, RQ2 y RQ4 están terminados; lo que falta está en [Pendientes](#pendientes).

**Empieza por [`index.md`](index.md)** (portada del libro) **y por [`notebooks/resultados_e2.ipynb`](notebooks/resultados_e2.ipynb).** Es el notebook de resultados, ya ejecutado, así que se puede leer sin descargar los datos:

1. Control de calidad del cálculo.
2. Tabla principal: F1 por grupo en test, parámetros y tiempo por corrida.
3. RQ1: Top 10 contra el mejor baseline (Wilcoxon, Holm, IC bootstrap).
4. RQ2: familia de dos etapas frente a prototipos.
5. Comparación global (Friedman, Nemenyi) y pares exploratorios.
6. Estabilidad entre semillas y mejor época.
7. RQ4 (sobreajuste), curvas por época y subentrenamiento.

El mismo informe se publica como **Jupyter Book** (`_config.yml` y `_toc.yml` en la raíz). Capítulos:

- [`informe/literatura.md`](informe/literatura.md): revisión de literatura con PRISMA 2020 y selección del Top 10.
- [`preregistro_E2.md`](preregistro_E2.md): lo que se fijó antes de entrenar.
- [`ENTRENAMIENTO_E2.md`](ENTRENAMIENTO_E2.md) y [`results/e2/prueba_humo/README.md`](results/e2/prueba_humo/README.md): protocolo común, prueba de humo y desviaciones declaradas.
- [`notebooks/registro_e2.ipynb`](notebooks/registro_e2.ipynb): búsqueda de hiperparámetros, configuración elegida por modelo, regla de selección verificada y entornos registrados de las 70 corridas.
- [`informe/conclusiones.md`](informe/conclusiones.md): conclusiones provisionales.
- [`informe/reproducibilidad.md`](informe/reproducibilidad.md): qué se regenera sin GPU, entornos y qué deja cada corrida.
- [`informe/pendientes.md`](informe/pendientes.md): alcance y pendientes.

Para construirlo:

```bash
pip install -r requirements_analisis.txt
jupyter-book build .
```

El libro queda en `_build/html/index.html`. Usa las salidas guardadas en los notebooks, así que no hace falta el dataset ni una GPU.

## Estado del entregable

| Sección del enunciado | Peso | Estado |
|---|---|---|
| Problema, hipótesis, preregistro y EDA | 15 % | Preregistro y prueba de humo hechos; el EDA del E1 se hereda y falta ampliarlo (auditoría de fuga) |
| Revisión de literatura, paper guía y gap | 15 % | Revisión, PRISMA y Top 10 hechos (`literatura/e2/`); faltan la hoja de extracción, el texto por modelo, el paper guía y el gap |
| Implementación de los modelos y rigor experimental | 25 % | Hecho: búsqueda de hiperparámetros y 70 corridas finales (`results/e2/`) |
| Benchmark e inferencia estadística | 20 % | Hecho para RQ1 y RQ2 (`analisis_e2.py`, `results/e2/analisis/`) |
| Ruta por tipo de problema (ajuste, sobreajuste, calibración, robustez, incertidumbre, interpretabilidad) | 15 % | Parcial: sobreajuste (RQ4), curvas y subentrenamiento hechos; faltan calibración, robustez, incertidumbre e interpretabilidad |
| Reproducibilidad, redacción y defensa | 10 % | Parcial: código, entornos congelados y análisis reproducible; faltan prueba de entorno limpio y checklist |

**Resultados (test, F1 de instancia por grupo, media de 5 semillas):**
- Mask R-CNN (base del E1, mejor baseline por validación): 0,726.
- HTC: 0,724. Cascade Mask R-CNN: 0,723. MS R-CNN: 0,716. PointRend: 0,701.
- YOLOv7-seg: 0,638. CondInst: 0,607. YOLO26: 0,569. YOLOv8 (base): 0,566. Mask2Former: 0,545. YOLO11: 0,530. YOLOv9: 0,405. SOLOv2 (base): 0,231.

Ningún modelo del Top 10 supera a Mask R-CNN (RQ1, Holm p = 1 en los 10 contrastes). La familia de dos etapas supera a la de prototipos por +0,181 de F1, en 13 de 14 grupos (RQ2, Wilcoxon p = 0,00024); la comparación mezcla familia, marco de software y presupuesto, así que no identifica la causa. Una de las cinco semillas de Mask2Former colapsó (F1 0,30 frente a 0,53–0,66).

## Modelos

- [`ENTRENAMIENTO_E2.md`](ENTRENAMIENTO_E2.md) tiene el protocolo, los comandos y los entornos.
- `entrenar_e2.py` es el orquestador (solo biblioteca estándar): lanza cada corrida en el entorno de su marco (`wdsd_dl`, `wdsd_mmdet` o `wdsd_yolo9`, con los `requirements_wdsd_*_congelado.txt`). `src/wdsd_e2/` tiene un trabajador por marco.
- Top 10: Mask2Former, YOLOv7-seg, YOLOv9-seg, YOLO26n-seg, YOLO11n-seg, Cascade Mask R-CNN, HTC, PointRend, Mask Scoring R-CNN y CondInst. Modelos base del E1 reentrenados con el mismo presupuesto: Mask R-CNN, YOLOv8n-seg y SOLOv2. Baselines: regresión logística por píxel y máscara vacía.
- `analisis_e2.py` hace el análisis del preregistro §5: Wilcoxon con Holm, bootstrap por grupos, Friedman con Nemenyi y pares con Benjamini–Hochberg. `analisis_rq4.py` calcula la brecha entre entrenamiento y validación (RQ4) y las curvas por época.
- Cada corrida deja en `results/e2/corridas/<modelo>_s<semilla>/` sus métricas, el F1 por imagen y las predicciones. Los pesos no se versionan.
- Todas las corridas finales se hicieron en una RTX 3060 Laptop de 6 GB.

## Lo que debes saber antes de reproducir

1. **Mismo presupuesto para todos:** 6 épocas, lote efectivo 2 (372 pasos por época), calentamiento y coseno, AMP, solo volteo horizontal y pesos de COCO. La tasa de aprendizaje se eligió por modelo entre 3 valores (por defecto × 0,5, 1 y 2) con la semilla 42.
2. **Métrica primaria:** F1 de instancia por imagen con IoU de máscara ≥ 0,5, promediado por grupo de escena. La unidad de independencia es el grupo; test tiene 14. δ = 0,20, heredado del E1 (allí se fijó después de ver resultados; se declara).
3. **La partición es la agrupada del E1** (744 / 93 / 108 imágenes). No uses la partición oficial como principal: tiene fuga por escena.
4. **No reduzcas la imagen.** El diámetro mediano de un tornillo es 10,6 px; los modelos con RPN usan anclas de 8 a 128 px.
5. **Cinco modelos corren en MMDetection y no en el código de sus autores:** Cascade, MS R-CNN, PointRend, Mask2Former y CondInst. YOLOv7 y YOLOv9 usan los repos oficiales con parches declarados en `results/e2/prueba_humo/repos_oficiales/README.md`.
6. **Ninguna corrida se excluyó.** Mask2Former tiene una semilla inestable que se reporta; el preregistro solo permite excluir por fallo técnico.

## Pendientes

- [ ] Hoja de extracción y valoración de calidad de los 64 estudios incluidos.
- [ ] Texto por modelo: DOI, diagrama, innovación, FLOPs, matriz comparativa y gap de investigación.
- [ ] RQ3 (F1 por tercil de área; necesita el área de cada tornillo).
- [ ] Latencia en una sola GPU, FLOPs, calibración, robustez e interpretabilidad.
- [ ] Tabla de preentrenamiento por modelo, resumen estructurado y discusión de amenazas a la validez.
- [ ] Auditoría de fuga, techo e incertidumbre, y paper guía con réplica parcial.
- [ ] Prueba de entorno limpio, Dockerfile opcional y checklist del Apéndice B.
- [x] Preregistro, prueba de humo y piloto de tiempo.
- [x] Revisión de literatura y Top 10.
- [x] Búsqueda de hiperparámetros y 70 corridas finales.
- [x] Análisis estadístico de RQ1 y RQ2.
- [x] RQ4 (brecha entre entrenamiento y validación) y curvas por época.
- [x] Registro de la búsqueda y de los entornos, con la regla de selección de tasa verificada (capítulo 3.2).
- [x] Fecha del preregistro: tag `e2-preregistro` en el repositorio del E1, mismo hash de archivo (ver Preregistro).

## Preregistro

El preregistro se publicó antes del primer entrenamiento en el repositorio del entregable 1 (rama `e2-revision-prueba-humo`, release `e2-preregistro`): <https://github.com/BESADAJ/wdsd-segmentacion-tornillos-entregable-1/tree/e2-revision-prueba-humo>. [`preregistro_E2.md`](preregistro_E2.md) es el mismo documento; este repositorio es posterior a las corridas. El archivo es idéntico en el tag y aquí (hash de git `e688df6f9852f7622ef3f88dd632ed1ac3d73048`); GitHub fecha el tag el 17-nov-2026 a las 00:10 (hora de Colombia) y la primera corrida de la búsqueda de hiperparámetros es de las 00:24. Detalle y límites en [`index.md`](index.md).

## Datos

El dataset no está en este repositorio porque pesa 2,7 GB y no tiene licencia declarada. Descarga la carpeta `WDSD` desde la [página oficial](https://vcl.iti.gr/dataset/weee-disassembly-screw-dataset/) (dos archivos zip; hacen falta los dos) y extráelos junto a este repositorio. Debes tener 945 pares y 4.398 instancias COCO. `entrenar_e2.py datos` genera `datos_inst/` (COCO y YOLO con la partición agrupada, y la submuestra fija `train93`), que no se versiona.

## Ejecución

Las tablas y figuras de resultados se regeneran sin GPU ni dataset:

```bash
pip install -r requirements_analisis.txt
python analisis_e2.py results/e2/corridas results/e2/corridas.csv results/e2/analisis
python analisis_rq4.py results/e2/corridas results/e2/analisis
```

El notebook de resultados ejecuta ese mismo script y no entrena nada: lee `results/e2/corridas/`. Para reentrenar, sigue `ENTRENAMIENTO_E2.md`.

## Estructura del repositorio

```text
index.md, _config.yml, _toc.yml   libro (Jupyter Book)
informe/                          capítulos: literatura y pendientes
notebooks/resultados_e2.ipynb     resultados e inferencia, ya ejecutado
preregistro_E2.md                 preregistro
ENTRENAMIENTO_E2.md               protocolo y comandos de entrenamiento
entrenar_e2.py, src/              código de entrenamiento (wdsd_e2, wdsd_inst, wdsd_e1)
analisis_e2.py                    análisis estadístico
literatura/e2/                    búsquedas, cribado, PRISMA y puntaje
results/e2/                       corridas, búsqueda, prueba de humo y análisis
```
