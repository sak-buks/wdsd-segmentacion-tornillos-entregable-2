# Conclusiones (provisionales)

Provisionales porque el informe está en construcción: faltan RQ3, RQ4, calibración, robustez, interpretabilidad y la auditoría de fuga (ver [Alcance y pendientes](pendientes.md)). Lo que sigue está respaldado por el análisis ya hecho, preregistrado y reproducible.

1. **Ningún modelo del Top 10 supera al mejor baseline fuerte** (Mask R-CNN, F1 por grupo en test 0,726; elegido por validación). Ningún contraste es significativo tras Holm (p = 1 en los 10), y las diferencias medias van de −0,002 a −0,321.
   - HTC, Cascade, MS R-CNN y PointRend son indistinguibles de Mask R-CNN (diferencias de −0,002 a −0,025, con intervalos que incluyen o rozan 0).
   - YOLOv7-seg y CondInst son peores por 0,09 a 0,12; YOLO26, Mask2Former, YOLO11 y YOLOv9, por 0,16 a 0,32.
2. **La familia de dos etapas supera a la de prototipos** por +0,181 de F1 (IC 95 % 0,134 a 0,222), en 13 de 14 grupos (Wilcoxon, p = 0,00024). La comparación mezcla familia, marco de software, tamaño de lote efectivo y preentrenamiento; no identifica la arquitectura como causa.
3. **La variación entre grupos de escena domina:** con 14 grupos de prueba solo se detectan diferencias grandes, y los cinco modelos de dos etapas no se distinguen entre sí (Friedman con distancia crítica de Nemenyi).
4. **Mask2Former es inestable:** una de sus cinco semillas colapsó (F1 0,30 frente a 0,53–0,66). No se excluyó, porque el preregistro solo permite excluir por fallo técnico.
5. **Presupuesto:** con 6 épocas para todos, la mejor época fue la última en varios modelos; no se descarta subentrenamiento (regla de parada del preregistro).
6. **Hipótesis para el entregable 3 (inferida, no probada):** como el modelo base más simple de la familia R queda primero entre 14, parte del límite de desempeño parece venir de los datos (tornillos de unos 10 px, 14 grupos de prueba) y no de la arquitectura. Habría que contrastarlo con el techo de anotación y la auditoría de fuga antes de afirmarlo.
