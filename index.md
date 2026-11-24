# Benchmark de 10 modelos de segmentación de instancias de tornillos en WDSD

**Autores:** Ivan Berrio, Jose David Castillo y Jaime Besada. **Entregable 2**, Deep Learning.

```{note}
**Estado: informe en construcción (2026-11-24).** Terminados: revisión de literatura, Top 10, preregistro, entrenamiento (70 corridas finales), análisis estadístico de RQ1, RQ2 y RQ4, y exploratorios (RQ3, calibración, robustez, latencia, varianza, Pareto, errores compartidos e incertidumbre). Lo que falta está listado sin omisiones en [Alcance y pendientes](informe/pendientes.md). El Entregable 1 sigue publicado en su propio libro y no se modificó.
```

## Propósito

El entregable 1 fijó una línea base con tres modelos (Mask R-CNN, YOLOv8-seg y SOLOv2) para segmentar tornillos en el WEEE Disassembly Screw Dataset (WDSD). Este entregable pregunta si modelos más recientes o más complejos de la literatura la superan. Se compara un Top 10 seleccionado con una revisión sistemática contra los modelos base, un baseline no profundo (regresión logística por píxel) y uno trivial (máscara vacía), bajo un protocolo idéntico, preregistrado y con 5 semillas por modelo.

## Datos

945 imágenes de 1280×720 con 4.398 tornillos anotados; el diámetro mediano de un tornillo es de unos 10 píxeles, así que casi todos son «pequeños» según COCO. La partición es la agrupada por escena del entregable 1 (744 / 93 / 108 imágenes), que evita la fuga entre entrenamiento y prueba. La prueba tiene 14 grupos de escena, y esos 14 grupos son la unidad de independencia de todas las pruebas estadísticas. El dataset no se redistribuye.

## Resultados principales

**Tabla 1.** F1 de instancia por grupo en test (media de 5 semillas). Es el F1 por imagen con IoU de máscara ≥ 0,5, promediado por grupo de escena.

| Modelo | Familia | F1 test | Diferencia con Mask R-CNN [IC 95 %] | Lectura |
|---|---|---|---|---|
| Mask R-CNN (base E1) | R | 0,726 | — | Mejor baseline por validación |
| HTC | R | 0,724 | −0,002 [−0,017; 0,012] | Indistinguible |
| Cascade Mask R-CNN | R | 0,723 | −0,002 [−0,022; 0,017] | Indistinguible |
| MS R-CNN | R | 0,716 | −0,010 [−0,027; 0,007] | Indistinguible |
| PointRend | R | 0,701 | −0,025 [−0,055; 0,000] | Indistinguible |
| YOLOv7-seg | P | 0,638 | −0,088 [−0,148; −0,024] | Peor |
| CondInst | K | 0,607 | −0,119 [−0,176; −0,067] | Peor |
| YOLO26n-seg | P | 0,569 | −0,157 [−0,212; −0,104] | Peor |
| Mask2Former | Q | 0,545 | −0,181 [−0,242; −0,123] | Peor (una semilla colapsó) |
| YOLO11n-seg | P | 0,530 | −0,196 [−0,251; −0,142] | Peor |
| YOLOv9-seg | P | 0,405 | −0,321 [−0,408; −0,238] | Peor |

Familias: R = dos etapas, P = prototipos (YOLO), K = kernels dinámicos, Q = consultas. YOLOv8-seg (base E1) tiene 0,566, SOLOv2 (base E1) 0,231 y el baseline clásico 0,018.

- **RQ1.** Ningún modelo del Top 10 supera al mejor baseline fuerte (Wilcoxon con corrección de Holm: p = 1 en los 10 contrastes).
- **RQ2.** La familia de dos etapas supera a la de prototipos por +0,181 de F1, en 13 de 14 grupos (p = 0,00024). Familia, marco de software y presupuesto están mezclados, así que no se atribuye a la arquitectura.

```{figure} results/e2/analisis/rq1_diferencias.png
:width: 90%
:align: center

**Figura 1.** Diferencia de F1 por grupo de cada modelo del Top 10 frente a Mask R-CNN, con IC 95 % por bootstrap de los 14 grupos. Las líneas rojas marcan ±δ = 0,20, la diferencia mínima considerada relevante.
```

## Cómo recorrer el libro

1. [Preregistro](preregistro_E2.md): lo que se fijó antes de entrenar.
2. [Revisión de literatura y Top 10](informe/literatura.md): PRISMA, puntaje y selección.
3. [Protocolo de entrenamiento](ENTRENAMIENTO_E2.md), [prueba de humo](results/e2/prueba_humo/README.md) y [búsqueda, configuración y registro de corridas](notebooks/registro_e2).
4. [Benchmark e inferencia estadística](notebooks/resultados_e2): definiciones de cada prueba, tablas y figuras.
5. [Análisis exploratorios](informe/exploratorios.md): tamaño del tornillo (RQ3), calibración, robustez, latencia, varianza, Pareto, errores compartidos, incertidumbre y presupuesto de épocas.
6. [Conclusiones](informe/conclusiones.md).
7. Anexos: [alcance y pendientes](informe/pendientes.md) y [reproducibilidad](informe/reproducibilidad.md).

## Alcance y límites de la evidencia

- Seis épocas para todos los modelos: YOLOv8, YOLO11, YOLO26 y Mask2Former siguen mejorando en la última época y se declaran subentrenados. RQ1 y RQ2 valen para ese presupuesto (ver [Presupuesto de 6 épocas](informe/exploratorios.md)).
- Con 14 grupos de prueba, la potencia es suficiente para detectar diferencias grandes, no pequeñas.
- δ = 0,20 se heredó del entregable 1, donde se fijó después de ver resultados; con ese margen la etiqueta «equivalente» incluye modelos claramente peores, por eso se muestran siempre la dirección y el intervalo.
- Una sola GPU y un solo revisor humano en la revisión de literatura (con asistencia de IA para el cribado por título y resumen).

## Fechado del preregistro

El preregistro se publicó antes de la búsqueda de hiperparámetros y de las corridas finales en el repositorio del entregable 1, en la rama `e2-revision-prueba-humo` y con el tag `e2-preregistro`: <https://github.com/BESADAJ/wdsd-segmentacion-tornillos-entregable-1/tree/e2-revision-prueba-humo>. Este repositorio contiene el mismo documento (`preregistro_E2.md`) y es posterior a las corridas; la fecha que cuenta es la de aquel repositorio.

- **Mismo archivo:** el hash de git del archivo (`git rev-parse`) es `e688df6f9852f7622ef3f88dd632ed1ac3d73048` en el tag `e2-preregistro` (objeto `aa5d7a2c396e4f18fcf594a92480ba7e54f93d6d` en GitHub) y en este repositorio.
- **Orden de los hechos (observado el 18-nov-2026):** el último commit que modificó `preregistro_E2.md` en el repositorio del E1 es del 17-nov-2026 a las 00:10 (hora de Colombia); los archivos de la primera corrida de la búsqueda de hiperparámetros (`mask2former`, ×0.5) tienen fecha del 17-nov a las 00:24. La prueba de humo y el piloto de tiempo, anteriores, no entrenan con validación ni prueba (preregistro, encabezado).
- **Fecha del tag (observado el 18-nov-2026):** la página de tags del repositorio del E1 muestra `e2-preregistro` apuntando al commit `a09c6b7`, con fecha 17-nov-2026 12:10 AM (GMT-5): <https://github.com/BESADAJ/wdsd-segmentacion-tornillos-entregable-1/tags>.
