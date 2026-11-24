"""Figura cualitativa del E2: predicciones contra verdad en 4 imágenes de test (enunciado §4.5).

Modelos: Mask R-CNN (mejor baseline), HTC (mejor del Top 10) y YOLOv8 (familia de prototipos), semilla 42, con el umbral
de score elegido en val de cada corrida (resumen.json). Imágenes elegidas por el F1 medio de los 13 modelos (semilla 42):
una fácil, una con más tornillos donde los modelos difieren, una intermedia y una del grupo 9, que ningún modelo resuelve.
Emparejamiento igual al de la métrica: húngaro con IoU de máscara >= 0,5. Verde = verdadero positivo, rojo = falso
positivo, círculo amarillo = tornillo anotado no detectado. Cada panel se recorta alrededor de los tornillos anotados: los
falsos positivos fuera del recorte no se ven, pero sí se cuentan en el título.

No usa GPU. Uso (desde Trabajo\\datos\\wdsd, entorno wdsd_dl):  python figuras_cualitativas_e2.py
Salida: results/e2/cualitativo/predicciones_cualitativas.png y .csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mu  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

RAIZ = Path(__file__).resolve().parent
CORR = RAIZ / "results" / "e2" / "corridas"
SAL = RAIZ / "results" / "e2" / "cualitativo"
MODELOS = [("maskrcnn", "Mask R-CNN"), ("htc", "HTC"), ("yolov8", "YOLOv8")]
IMAGENES = [(22, "Fácil"), (38, "Muchos tornillos"), (59, "Intermedia"), (1, "Grupo 9: todos fallan")]
SEMILLA = 42


def rle(seg, h, w):
    if isinstance(seg, str):
        seg = eval(seg) if seg.startswith("{") else seg  # noqa: S307 (archivo propio)
    if isinstance(seg, list):
        return mu.merge(mu.frPyObjects(seg, h, w))
    if isinstance(seg.get("counts"), list):
        return mu.frPyObjects(seg, h, w)
    return {"size": seg["size"], "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"]}


def emparejar(gt, pr):
    if not gt or not pr:
        return [], list(range(len(pr))), list(range(len(gt)))
    iou = mu.iou(pr, gt, [0] * len(gt))
    f, c = linear_sum_assignment(-iou)
    tp = [(i, j) for i, j in zip(f, c) if iou[i, j] >= 0.5]
    up, ug = {i for i, _ in tp}, {j for _, j in tp}
    return tp, [i for i in range(len(pr)) if i not in up], [j for j in range(len(gt)) if j not in ug]


def contorno(ax, m, color, x0, y0):
    ax.contour(np.arange(m.shape[1]) - x0, np.arange(m.shape[0]) - y0, m, levels=[0.5], colors=[color], linewidths=1.6)


def main(fondo_gris=False):
    test = json.loads((RAIZ / "datos_inst" / "coco" / "test.json").read_text(encoding="utf8"))
    info = {i["id"]: i for i in test["images"]}
    gts = {}
    for a in test["annotations"]:
        gts.setdefault(a["image_id"], []).append(a)
    preds = {}
    for m, _ in MODELOS:
        c = CORR / f"{m}_s{SEMILLA}"
        u = json.loads((c / "resumen.json").read_text(encoding="utf8"))["metricas"]["umbral_score"]
        por = {}
        for p in json.loads((c / "preds_test.json").read_text(encoding="utf8")):
            if float(p["score"]) >= u:
                por.setdefault(int(p["image_id"]), []).append(p)
        preds[m] = (u, por)
    fig, axs = plt.subplots(len(IMAGENES), len(MODELOS), figsize=(4.2 * len(MODELOS), 3.4 * len(IMAGENES)))
    filas = []
    for r, (iid, etiqueta) in enumerate(IMAGENES):
        im = info[iid]; h, w = im["height"], im["width"]
        img = np.full((h, w, 3), 128, np.uint8) if fondo_gris else np.asarray(Image.open(im["ruta"]).convert("RGB"))
        g = [rle(a["segmentation"], h, w) for a in gts[iid]]
        bb = np.array([a["bbox"] for a in gts[iid]])
        x0 = max(0, int(bb[:, 0].min()) - 120); y0 = max(0, int(bb[:, 1].min()) - 90)
        x1 = min(w, int((bb[:, 0] + bb[:, 2]).max()) + 120); y1 = min(h, int((bb[:, 1] + bb[:, 3]).max()) + 90)
        for c_, (m, nombre) in enumerate(MODELOS):
            u, por = preds[m]
            pr = [rle(p["segmentation"], h, w) for p in por.get(iid, [])]
            tp, fp, fn = emparejar(g, pr)
            f1 = 2 * len(tp) / (2 * len(tp) + len(fp) + len(fn)) if (tp or fp or fn) else 1.0
            ax = axs[r, c_]
            ax.imshow(img[y0:y1, x0:x1]); ax.set_xticks([]); ax.set_yticks([])
            for i, _ in tp:
                contorno(ax, mu.decode(pr[i]), "lime", x0, y0)
            for i in fp:
                contorno(ax, mu.decode(pr[i]), "red", x0, y0)
            for j in fn:
                x, y, bw, bh = gts[iid][j]["bbox"]
                ax.add_patch(plt.Circle((x + bw / 2 - x0, y + bh / 2 - y0), max(bw, bh) * 0.9 + 6,
                                        fill=False, ec="yellow", lw=1.6))
            ax.set_xlim(-0.5, x1 - x0 - 0.5); ax.set_ylim(y1 - y0 - 0.5, -0.5)
            ax.set_title(f"{nombre}: VP {len(tp)}, FP {len(fp)}, FN {len(fn)}, F1 {f1:.2f}", fontsize=9)
            if c_ == 0:
                ax.set_ylabel(f"{etiqueta}\n{im['archivo']} (grupo {im['grupo']})", fontsize=9)
            filas.append(dict(image_id=iid, archivo=im["archivo"], grupo=im["grupo"], modelo=m, umbral=u,
                              n_gt=len(g), vp=len(tp), fp=len(fp), fn=len(fn), f1=round(f1, 3)))
    fig.suptitle("Verde: verdadero positivo · rojo: falso positivo · círculo amarillo: tornillo no detectado "
                 f"(semilla {SEMILLA}, umbral de val de cada modelo)", fontsize=10)
    fig.tight_layout()
    SAL.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAL / "predicciones_cualitativas.png", dpi=150)
    with (SAL / "predicciones_cualitativas.csv").open("w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=list(filas[0])); wr.writeheader(); wr.writerows(filas)
    for x in filas:
        print(x)


if __name__ == "__main__":
    main(fondo_gris="--gris" in sys.argv)
