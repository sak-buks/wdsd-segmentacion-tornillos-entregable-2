# Resumen estructurado

**Contexto.** Segmentar por instancias tornillos pequeños (lado mediano ~12 px) en el dataset WDSD (945 imágenes). El E1 comparó tres modelos; el E2 pregunta si los diez mejores modelos de la literatura reciente superan a un baseline fuerte.

**Métodos.** Revisión de literatura y selección de un Top 10. Partición agrupada por escena (744/93/108 imágenes; 14 grupos en test). Trece modelos (Top 10 y tres baselines), 5 semillas, 6 épocas. Métrica primaria preregistrada: F1 de instancia por imagen (IoU ≥ 0,5), promediado por grupo. Bootstrap por grupos, Wilcoxon y Holm. Exploratorios: tamaño, calibración, robustez, latencia, ablación, sanidad, fuga, paper guía e interpretabilidad.

**Resultados.** Ningún modelo del Top 10 supera a Mask R-CNN (F1 0,726; p de Holm = 1 en los 10 contrastes). Otros cuatro de dos etapas quedan a menos de 0,025. La familia de dos etapas supera a la de prototipos por 0,18 (IC 95 % 0,13–0,22). El grupo de escena explica el 61 % de la variación del F1 y el modelo el 19 %; 100 de 389 tornillos los pierden todos los modelos. Con etiquetas permutadas el F1 es 0. El 12,0 % de test tiene un casi gemelo en train (límite: 5 %) y sin esas imágenes el F1 baja unos 0,07. La réplica del paper guía falla (AP50 0,737 frente a 0,849).

**Conclusión.** Con este presupuesto de entrenamiento el límite está más en los datos (escenas y tornillos pequeños) que en la arquitectura. El entregable 3 debería atacar los tornillos pequeños.
