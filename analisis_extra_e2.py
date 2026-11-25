"""Análisis exploratorios del E2 (preregistro_E2.md §5 RQ3 y §7) que no necesitan GPU ni volver a entrenar.

Uso:
    python analisis_extra_e2.py <carpeta_corridas> <corridas.csv> <carpeta_coco> <carpeta_robustez_latencia> <salida>

- <carpeta_corridas>/<modelo>_s<semilla>/preds_test.json: predicciones guardadas de las 70 corridas finales.
- <carpeta_coco>/{val,test}.json: anotaciones COCO de la partición agrupada (datos_inst/coco, no versionado).
- <carpeta_robustez_latencia>: salida de robustez_latencia_e2.py (robustez/*.csv y latencia.csv).

Qué calcula (todo exploratorio; nada de esto elige modelo):
1. RQ3, F1 por tercil de área del tornillo. Cortes fijados con las áreas de VAL (no de test). Emparejamiento igual
   que la métrica primaria (húngaro uno a uno, IoU de máscara >= 0.5, umbral de score elegido en val por corrida).
   Cada verdadero positivo y falso negativo va al tercil de su tornillo real; cada falso positivo, al tercil de su
   máscara predicha. F1 de un grupo y tercil = 2TP / (2TP + FP + FN) con los conteos del grupo; luego media de las 5
   semillas y media de los grupos. Diferencia de cada modelo con el comparador de RQ1 por grupo, Wilcoxon bilateral
   y Benjamini–Hochberg sobre todos los contrastes modelo x tercil.
2. Calibración del score: todas las predicciones guardadas (score >= 0.05), emparejamiento voraz por score (como
   COCO) con IoU >= 0.5; un acierto es una predicción emparejada. ECE con 10 intervalos iguales de score.
3. Tamaños de efecto de RQ1: correlación biserial de rangos (r = (W+ - W-) / (W+ + W-)) y d de Cohen pareada.
4. Robustez y latencia: tablas a partir de las salidas de robustez_latencia_e2.py.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mu
from scipy import stats
from scipy.optimize import linear_sum_assignment

RUNS, CSV, COCO_DIR, ROB, OUT = (Path(x) for x in sys.argv[1:6])
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = range(42, 47)
TOP10 = ["mask2former", "yolov7", "yolov9", "yolo26", "yolo11", "cascade", "htc", "pointrend", "msrcnn", "condinst"]
BASE = ["maskrcnn", "yolov8", "solov2"]
MODELOS = TOP10 + BASE + ["clasico"]
COMPARADOR = "maskrcnn"  # mejor baseline fuerte por F1 de grupo en val (analisis_e2.py)
TERCILES = ["pequeño", "mediano", "grande"]
rng = np.random.default_rng(0)


def gt_rles(coco: dict):
    imgs = {i["id"]: i for i in coco["images"]}
    por_img = defaultdict(list)
    for a in coco["annotations"]:
        i = imgs[a["image_id"]]
        r = mu.frPyObjects(a["segmentation"], i["height"], i["width"])
        por_img[a["image_id"]].append(mu.merge(r) if isinstance(r, list) else r)
    return imgs, por_img


def bh(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 1.0
    for rank in range(m - 1, -1, -1):
        i = o[rank]; run = min(run, p[i] * m / (rank + 1)); adj[i] = run
    return adj


def tercil(area, cortes):
    return int(np.searchsorted(cortes, area, side="right"))  # 0, 1, 2


# ---------------------------------------------------------------- datos
test = json.loads((COCO_DIR / "test.json").read_text(encoding="utf8"))
val = json.loads((COCO_DIR / "val.json").read_text(encoding="utf8"))
cortes = np.percentile([a["area"] for a in val["annotations"]], [100 / 3, 200 / 3])
imgs, gts = gt_rles(test)
area_gt = {iid: [float(mu.area(r)) for r in rs] for iid, rs in gts.items()}
corr = pd.read_csv(CSV)
umbral = {(r.modelo, int(r.semilla)): float(r.umbral_score) for r in corr.itertuples()}
n_gt_tercil = pd.Series([tercil(a, cortes) for v in area_gt.values() for a in v]).value_counts().sort_index()
json.dump({"cortes_area_px_val": cortes.tolist(), "n_tornillos_test_por_tercil": n_gt_tercil.tolist()},
          open(OUT / "rq3_cortes.json", "w"), indent=1)

# ---------------------------------------------------------------- 1 y 2: una pasada por corrida
filas_t, filas_cal, control = [], [], []
for m in MODELOS:
    for s in SEEDS:
        preds = json.loads((RUNS / f"{m}_s{s}" / "preds_test.json").read_text())
        por_img = defaultdict(list)
        for p in preds:
            por_img[int(p["image_id"])].append(p)
        u = umbral[(m, s)]
        conteo = defaultdict(lambda: np.zeros(3))  # (grupo, tercil) -> [tp, fp, fn]
        f1_img = []
        for iid, info in imgs.items():
            g = gts.get(iid, []); ag = area_gt.get(iid, [])
            todas = sorted(por_img[iid], key=lambda p: -p["score"])
            # --- RQ3: predicciones sobre el umbral, húngaro (igual que la métrica primaria)
            d = [p for p in todas if p["score"] >= u]
            dr = [p["segmentation"] for p in d]
            emp_g, emp_d = set(), set()
            if g and dr:
                iou = mu.iou(dr, g, [0] * len(g))
                f, c = linear_sum_assignment(-iou)
                for a, b in zip(f, c):
                    if iou[a, b] >= 0.5:
                        emp_d.add(a); emp_g.add(b)
            grp = info["grupo"]
            for j, a in enumerate(ag):
                k = tercil(a, cortes)
                conteo[(grp, k)][0 if j in emp_g else 2] += 1
            for j, r in enumerate(dr):
                if j not in emp_d:
                    conteo[(grp, tercil(float(mu.area(r)), cortes))][1] += 1
            tp = len(emp_g); fp = len(dr) - tp; fn = len(g) - tp
            den = tp + 0.5 * fp + 0.5 * fn
            f1_img.append((grp, tp / den if den else 1.0))
            # --- calibración: todas las guardadas, voraz por score
            if todas:
                ar = [p["segmentation"] for p in todas]
                usado = set()
                iou_all = mu.iou(ar, g, [0] * len(g)) if g else np.zeros((len(ar), 0))
                for j, p in enumerate(todas):
                    ok = 0
                    if g:
                        fila = np.asarray(iou_all[j]).copy()
                        fila[list(usado)] = -1
                        b = int(np.argmax(fila))
                        if fila[b] >= 0.5:
                            usado.add(b); ok = 1
                    filas_cal.append((m, s, p["score"], ok))
        for (grp, k), (tp, fp, fn) in conteo.items():
            den = 2 * tp + fp + fn
            filas_t.append((m, s, grp, TERCILES[k], tp, fp, fn, 2 * tp / den if den else np.nan))
        fi = pd.DataFrame(f1_img, columns=["grupo", "f1"])
        ref = corr[(corr.modelo == m) & (corr.semilla == s)].test_F1_grupo.iloc[0]
        control.append((m, s, ref, fi.groupby("grupo").f1.mean().mean()))
    print("listo", m, flush=True)

ctrl = pd.DataFrame(control, columns=["modelo", "semilla", "corridas_csv", "recalculado"])
ctrl["dif_abs"] = (ctrl.corridas_csv - ctrl.recalculado).abs()
ctrl.to_csv(OUT / "extra_control_calidad.csv", index=False)
print("control de calidad, máx |dif| F1 grupo:", ctrl.dif_abs.max())

# ---------------------------------------------------------------- RQ3
T = pd.DataFrame(filas_t, columns=["modelo", "semilla", "grupo", "tercil", "tp", "fp", "fn", "f1"])
T.to_csv(OUT / "rq3_conteos_por_grupo.csv", index=False)
G = T.groupby(["modelo", "grupo", "tercil"]).f1.mean().reset_index()  # media de semillas
tabla = (G.groupby(["modelo", "tercil"]).f1.mean().unstack()[TERCILES]
         .reindex(MODELOS).rename(columns=lambda c: f"F1_{c}"))
rec = T.groupby(["modelo", "tercil"])[["tp", "fn"]].sum()
rec = (rec.tp / (rec.tp + rec.fn)).unstack()[TERCILES].rename(columns=lambda c: f"recall_{c}")
tabla.join(rec).to_csv(OUT / "rq3_tabla.csv")
pruebas = []
for m in TOP10 + ["yolov8", "solov2"]:
    for t in TERCILES:
        a = G[(G.modelo == m) & (G.tercil == t)].set_index("grupo").f1
        b = G[(G.modelo == COMPARADOR) & (G.tercil == t)].set_index("grupo").f1
        d = (a - b).dropna().to_numpy()
        try:
            p = stats.wilcoxon(d).pvalue
        except ValueError:
            p = 1.0
        idx = rng.integers(0, len(d), (10000, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        pruebas.append(dict(modelo=m, tercil=t, n_grupos=len(d), dif_media=d.mean(), ic95_lo=lo, ic95_hi=hi, p=p))
pr = pd.DataFrame(pruebas)
pr["p_BH"] = bh(pr.p)
pr.to_csv(OUT / "rq3_vs_comparador_BH.csv", index=False)

# ---------------------------------------------------------------- calibración
C = pd.DataFrame(filas_cal, columns=["modelo", "semilla", "score", "acierto"])
bordes = np.linspace(0.05, 1.0, 11)
cal, curva = [], []
for (m, s), d in C.groupby(["modelo", "semilla"]):
    b = np.clip(np.digitize(d.score, bordes) - 1, 0, 9)
    ece = sum((b == k).sum() / len(d) * abs(d.acierto[b == k].mean() - d.score[b == k].mean())
              for k in range(10) if (b == k).any())
    cal.append((m, s, len(d), d.acierto.mean(), d.score.mean(), ece))
for m, d in C.groupby("modelo"):
    b = np.clip(np.digitize(d.score, bordes) - 1, 0, 9)
    for k in range(10):
        if (b == k).any():
            curva.append((m, k, d.score[b == k].mean(), d.acierto[b == k].mean(), int((b == k).sum())))
cal = pd.DataFrame(cal, columns=["modelo", "semilla", "n_pred", "tasa_acierto", "score_medio", "ECE"])
cal.to_csv(OUT / "calibracion_por_corrida.csv", index=False)
(cal.groupby("modelo").agg(ECE=("ECE", "mean"), ECE_DE=("ECE", "std"), score_medio=("score_medio", "mean"),
                          tasa_acierto=("tasa_acierto", "mean"), n_pred=("n_pred", "mean"))
    .reindex(MODELOS).to_csv(OUT / "calibracion.csv"))
pd.DataFrame(curva, columns=["modelo", "intervalo", "score_medio", "tasa_acierto", "n"]).to_csv(
    OUT / "calibracion_curvas.csv", index=False)

# ---------------------------------------------------------------- tamaños de efecto de RQ1
F = pd.read_csv(OUT.parent / "analisis" / "f1_grupo_promedio_semillas.csv", index_col=0) \
    if (OUT.parent / "analisis" / "f1_grupo_promedio_semillas.csv").exists() else \
    pd.read_csv(OUT / "f1_grupo_promedio_semillas.csv", index_col=0)
ef = []
for m in TOP10:
    d = (F[m] - F[COMPARADOR]).to_numpy()
    r = stats.rankdata(np.abs(d[d != 0]))
    wp, wm = r[d[d != 0] > 0].sum(), r[d[d != 0] < 0].sum()
    ef.append(dict(modelo=m, dif_media=d.mean(), r_biserial=(wp - wm) / (wp + wm), d_cohen_pareada=d.mean() / d.std(ddof=1)))
pd.DataFrame(ef).sort_values("dif_media", ascending=False).to_csv(OUT / "rq1_tamanos_efecto.csv", index=False)

# ---------------------------------------------------------------- robustez y latencia
R = pd.concat([pd.read_csv(p) for p in sorted((ROB / "robustez").glob("*.csv"))], ignore_index=True)
Rg = R.groupby(["modelo", "semilla", "perturbacion", "nivel", "grupo"]).f1.mean().reset_index()
Rs = Rg.groupby(["modelo", "semilla", "perturbacion", "nivel"]).f1.mean().reset_index()
rob = Rs.groupby(["modelo", "perturbacion", "nivel"]).f1.agg(["mean", "std"]).reset_index()
limpio = rob[rob.perturbacion == "limpio"].set_index("modelo")["mean"]
rob["retencion"] = rob["mean"] / rob.modelo.map(limpio)
rob.rename(columns={"mean": "F1_grupo", "std": "DE_semillas"}).to_csv(OUT / "robustez_tabla.csv", index=False)
pd.read_csv(ROB / "latencia.csv").to_csv(OUT / "latencia_tabla.csv", index=False)
print(tabla.join(rec).round(3).to_string())
print(pr.round(4).to_string())
print(cal.groupby("modelo").ECE.mean().round(3).sort_values().to_string())

# ---------------------------------------------------------------- figuras
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
NOMBRE = {"maskrcnn": "Mask R-CNN", "htc": "HTC", "cascade": "Cascade", "msrcnn": "MS R-CNN", "pointrend": "PointRend",
          "yolov7": "YOLOv7", "condinst": "CondInst", "yolo26": "YOLO26", "yolov8": "YOLOv8", "mask2former": "Mask2Former",
          "yolo11": "YOLO11", "yolov9": "YOLOv9", "solov2": "SOLOv2", "clasico": "clásico"}
COL = {"pequeño": "#1b6ca8", "mediano": "#e08a1e", "grande": "#3a9b5c"}

t3 = tabla.drop(index="clasico").sort_values("F1_pequeño")
fig, ax = plt.subplots(figsize=(7, 5))
for k, t in enumerate(TERCILES):
    ax.scatter(t3[f"F1_{t}"], np.arange(len(t3)) + (k - 1) * 0.22, color=COL[t], s=28,
               label=f"{t} (≤{cortes[0]:.0f} px)" if k == 0 else (f"{t} ({cortes[0]:.0f}–{cortes[1]:.0f} px)" if k == 1
                                                                   else f"{t} (>{cortes[1]:.0f} px)"))
ax.set_yticks(range(len(t3))); ax.set_yticklabels([NOMBRE[m] for m in t3.index])
ax.set_xlabel("F1 por grupo en test (media de 5 semillas)"); ax.set_xlim(0, 1)
ax.set_title("RQ3: F1 por tercil de área del tornillo (cortes de val)")
ax.legend(title="Tercil de área", loc="lower right", fontsize=8); ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "rq3_terciles.png", dpi=200); plt.close(fig)

cv = pd.read_csv(OUT / "calibracion_curvas.csv")
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.plot([0, 1], [0, 1], color="#888", lw=1, ls="--", label="calibración perfecta")
for m, c in zip(["maskrcnn", "htc", "condinst", "mask2former", "yolov8", "yolo11"],
                ["#1b6ca8", "#5aa0d0", "#7a4fa3", "#c0392b", "#3a9b5c", "#e08a1e"]):
    d = cv[(cv.modelo == m) & (cv.n >= 20)]
    ax.plot(d.score_medio, d.tasa_acierto, marker="o", ms=3, color=c, label=NOMBRE[m])
ax.set_xlabel("Score medio del intervalo"); ax.set_ylabel("Fracción de predicciones correctas (IoU ≥ 0,5)")
ax.set_title("Calibración del score (test, 5 semillas juntas)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "calibracion.png", dpi=200); plt.close(fig)

rb = pd.read_csv(OUT / "robustez_tabla.csv")
# color por familia: dos etapas (azules), YOLO (verdes/naranja), otros (grises/morado); Mask R-CNN y los YOLO de ultralytics
# tienen 5 semillas (barras = DE), los otros 9 solo la semilla 42 (sin barras)
ESTILO = {"maskrcnn": ("#0b3d91", "-"), "htc": ("#1b6ca8", "--"), "cascade": ("#5aa0d0", "-."), "msrcnn": ("#7fb8e0", ":"),
          "pointrend": ("#2a8fbd", "-"), "yolo26": ("#3a9b5c", "-"), "yolov8": ("#7bc47f", "--"), "yolo11": ("#e08a1e", "-"),
          "yolov7": ("#b5651d", "--"), "yolov9": ("#d9a066", ":"), "condinst": ("#7a4fa3", "-"), "mask2former": ("#c0392b", "-"),
          "solov2": ("#777777", "-")}
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (tipo, etiq) in zip(axs, [("ruido", "σ del ruido gaussiano (escala 0–1)"), ("desenfoque", "σ del desenfoque gaussiano (px)")]):
    for m, (c, ls) in ESTILO.items():
        d = rb[(rb.modelo == m) & rb.perturbacion.isin(["limpio", tipo])].sort_values("nivel")
        ax.errorbar(d.nivel, d.F1_grupo, yerr=d.DE_semillas.fillna(0), marker="o", ms=3, capsize=2, color=c, ls=ls,
                    lw=1.4, label=NOMBRE[m])
    ax.set_xlabel(etiq); ax.grid(alpha=.3)
axs[0].set_ylabel("F1 por grupo en test"); axs[0].set_ylim(0, 0.8)
axs[1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.suptitle("Robustez a perturbaciones (nivel 0 = imagen limpia; barras = DE de 5 semillas solo en Mask R-CNN y YOLOv8/11/26)")
fig.tight_layout(); fig.savefig(OUT / "robustez.png", dpi=200); plt.close(fig)
print("figuras listas")
