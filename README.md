# WDSD: benchmark de 10 modelos de segmentación de instancias (Entregable 2, Deep Learning)

Autores: Ivan Berrio, Jose David Castillo y Jaime Besada. Informe en Jupyter Book; empieza por [`index.md`](index.md).

- `preregistro_E2.md`: preregistro (fechado en la release `e2-preregistro` del repositorio del Entregable 1, antes de entrenar).
- `entrenar_e2.py` y `src/`: entrenamiento de los 14 modelos (instrucciones en `ENTRENAMIENTO_E2.md`).
- `analisis_e2.py`: análisis estadístico del preregistro §5.
- `notebooks/resultados_e2.ipynb`: resultados, ya ejecutado.
- `literatura/e2/`: búsquedas, cribado, PRISMA y puntaje.
- `results/e2/`: tablas de corridas y búsqueda, resultados por corrida (sin pesos) y análisis.
- `informe/`: capítulos del libro.

Construir el libro: `pip install -r requirements_analisis.txt` y `jupyter-book build .`.
