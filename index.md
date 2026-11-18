# WDSD: benchmark de 10 modelos de segmentación de instancias (Entregable 2)

**Autores:** Ivan Berrio, Jose David Castillo y Jaime Besada.

Segmentación de instancias de tornillos en el WEEE Disassembly Screw Dataset (WDSD): 945 imágenes de 1280×720, con tornillos de unos 10 px de diámetro. Este informe compara un Top 10 de modelos de la literatura contra tres modelos base, un baseline no profundo y uno trivial, con preregistro, 5 semillas por modelo y pruebas estadísticas por grupo de escena.

```{note}
**Estado: informe en construcción (2026-11-18).** Están terminados la revisión de literatura, el Top 10, el preregistro, el entrenamiento completo (70 corridas finales) y el análisis estadístico de RQ1 y RQ2. Lo que falta está listado, sin omisiones, en [Alcance y pendientes](informe/pendientes.md). El Entregable 1 sigue publicado en su propio libro y no se modificó.
```

## Resultado principal

- **RQ1.** Ningún modelo del Top 10 supera al mejor baseline fuerte, Mask R-CNN (F1 por grupo en test 0.726; elegido por validación). HTC (0.724), Cascade (0.723), MS R-CNN (0.716) y PointRend (0.701) son indistinguibles de él; los otros seis son peores, por entre 0.09 y 0.32.
- **RQ2.** La familia de dos etapas supera a la de prototipos por +0.181 de F1, en 13 de 14 grupos (Wilcoxon p = 0.00024). Familia, marco y presupuesto están mezclados, así que no se atribuye a la arquitectura.
- **Mask2Former** fue inestable: una de sus cinco semillas colapsó (F1 0.30 frente a 0.53–0.66).

Detalle, definiciones de cada prueba y tablas en [Resultados](notebooks/resultados_e2.ipynb).

## Cómo leer este libro

1. [Revisión de literatura y Top 10](informe/literatura.md): PRISMA, puntaje y selección.
2. [Preregistro](preregistro_E2.md): lo que se fijó antes de entrenar.
3. [Entrenamiento](ENTRENAMIENTO_E2.md) y [prueba de humo](results/e2/prueba_humo/README.md): protocolo común y desviaciones declaradas.
4. [Resultados](notebooks/resultados_e2.ipynb): benchmark e inferencia estadística.
5. [Alcance y pendientes](informe/pendientes.md).

## Fechado del preregistro

El preregistro se publicó antes del primer entrenamiento en el repositorio del Entregable 1, en la rama `e2-revision-prueba-humo` y la release `e2-preregistro`: <https://github.com/BESADAJ/wdsd-segmentacion-tornillos-entregable-1/tree/e2-revision-prueba-humo>. Este repositorio nuevo contiene el mismo documento (`preregistro_E2.md`) y es posterior a las corridas; la fecha que cuenta es la de aquel push y aquella release.

## Reproducir las tablas

```bash
pip install -r requirements_analisis.txt
python analisis_e2.py results/e2/corridas results/e2/corridas.csv results/e2/analisis
jupyter-book build .
```

El análisis lee los CSV por imagen de test de `results/e2/corridas/` y no necesita GPU ni el dataset. El entrenamiento está en `entrenar_e2.py` y `src/` (instrucciones en [Entrenamiento](ENTRENAMIENTO_E2.md)); los entornos congelados están en `requirements_wdsd_*_congelado.txt`.
