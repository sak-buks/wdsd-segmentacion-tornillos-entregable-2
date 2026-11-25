"""Segunda tanda de análisis exploratorios del E2 sin GPU (lo que pide el enunciado y faltaba tras analisis_extra_e2.py).

Uso:
    python analisis_extra2_e2.py <carpeta_corridas> <corridas.csv> <carpeta_coco> <carpeta_robustez_latencia> \
        <carpeta_analisis> <carpeta_analisis_extra> <salida>

<carpeta_analisis> es la salida de analisis_e2.py (f1_grupo_por_semilla.csv) y <carpeta_analisis_extra> la de
analisis_extra_e2.py (latencia_tabla.csv). Todo es exploratorio: nada de esto cambia la elección de modelos.

Qué calcula (sección del enunciado entre paréntesis):
1. Calibración completa (4.5): además del ECE de 10 intervalos iguales, ECE adaptativo (10 intervalos con el mismo
   número de predicciones), Brier, pendiente e intercepto de calibración (regresión logística del acierto sobre el
   logit del score) y temperature scaling ajustado solo en validación (preds_val.json de la misma corrida) y aplicado
   a test, antes y después. Como la temperatura solo escala el logit y no corrige un sesgo, también se reporta Platt
   (logística de 2 parámetros), ajustado igual en validación. Ninguna de las dos cambia el F1: son transformaciones
   monótonas del score y el umbral se movería con ellas.
2. Índice de robustez (4.6): área bajo la curva de retención (F1 perturbado / F1 limpio) contra el nivel 0..3,
   dividida por 3, por corrida; 1 = no pierde nada.
3. δ de Cliff, A12 y probabilidad de superioridad pareada por grupo frente al comparador (13.3), con IC bootstrap.
4. Descomposición de la varianza (13.2): modelo lineal mixto y = mu + modelo + u_grupo + v_semilla + error sobre el
   F1 por grupo y semilla de los 13 modelos del benchmark, y la descomposición clásica por sumas de cuadrados.
5. Frontera de Pareto (6.2): F1 de grupo contra parámetros, tiempo de entrenamiento, latencia y FLOPs.
6. Errores compartidos, correlación de errores y ganancia de ensambles sencillos (12.3).
7. Incertidumbre predictiva con cobertura (12.2) en los 3 mejores: predicción conforme del número de tornillos por
   imagen, calibrada por grupos (deja fuera el grupo evaluado), y el rango entre las 5 semillas como ensamble.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from pycocotools import mask as mu  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import linear_sum_assignment, minimize_scalar  # noqa: E402

RUNS, CSV, COCO_DIR, ROB, AN, AX, OUT = (Path(x) for x in sys.argv[1:8])
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(42, 47))
TOP10 = ["mask2former", "yolov7", "yolov9", "yolo26", "yolo11", "cascade", "htc", "pointrend", "msrcnn", "condinst"]
BASE = ["maskrcnn", "yolov8", "solov2"]
BENCH = TOP10 + BASE  # el clásico (F1 0.02) se deja fuera de varianza, correlaciones y Pareto: solo infla todo
COMPARADOR = "maskrcnn"
TRES_MEJORES = ["maskrcnn", "htc", "cascade"]  # mejores por F1 de grupo en test (tabla_principal.csv)
ENSAMBLES = {"tres_mejores": TRES_MEJORES, "diverso": ["maskrcnn", "condinst", "yolov7"]}
B = 2000
ALFA = 0.10
rng = np.random.default_rng(0)
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ece_fijo(s, y, k=10):  # mismos bordes que analisis_extra_e2.py: 10 intervalos iguales entre 0.05 y 1
    b = np.clip(np.digitize(s, np.linspace(0.05, 1.0, k + 1)) - 1, 0, k - 1)
    return sum(abs(y[b == i].mean() - s[b == i].mean()) * (b == i).mean() for i in range(k) if (b == i).any())


def ece_adaptativo(s, y, k=10):
    o = np.argsort(s)
    return sum(abs(y[c].mean() - s[c].mean()) * len(c) / len(s) for c in np.array_split(o, k) if len(c))


def pendiente_intercepto(s, y):
    d = pd.DataFrame({"y": y, "x": logit(s)})
    r = smf.logit("y ~ x", d).fit(disp=0)
    return r.params["x"], r.params["Intercept"]


def ajustar_T(s, y):
    z = logit(s)

    def nll(logT):
        p = np.clip(1 / (1 + np.exp(-z / np.exp(logT))), EPS, 1 - EPS)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def f1_hungaro(dr, g):
    if not dr and not g:
        return 1.0, 0, 0, 0, set()
    emp_g = set(); tp = 0
    if dr and g:
        iou = mu.iou(dr, g, [0] * len(g))
        f, c = linear_sum_assignment(-iou)
        for a, b in zip(f, c):
            if iou[a, b] >= 0.5:
                tp += 1; emp_g.add(b)
    fp, fn = len(dr) - tp, len(g) - tp
    return tp / (tp + 0.5 * fp + 0.5 * fn), tp, fp, fn, emp_g


def ic_boot(x, f=np.mean):
    x = np.asarray(x)
    v = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return np.percentile(v, [2.5, 97.5])


# ---------------------------------------------------------------- datos
test = json.loads((COCO_DIR / "test.json").read_text(encoding="utf8"))
imgs = {i["id"]: i for i in test["images"]}
gts = defaultdict(list)
for a in test["annotations"]:
    i = imgs[a["image_id"]]
    r = mu.frPyObjects(a["segmentation"], i["height"], i["width"])
    gts[a["image_id"]].append(mu.merge(r) if isinstance(r, list) else r)
grupo_img = {iid: i["grupo"] for iid, i in imgs.items()}
val = json.loads((COCO_DIR / "val.json").read_text(encoding="utf8"))
vimgs = {i["id"]: i for i in val["images"]}
vgts = defaultdict(list)
for a in val["annotations"]:
    i = vimgs[a["image_id"]]
    r = mu.frPyObjects(a["segmentation"], i["height"], i["width"])
    vgts[a["image_id"]].append(mu.merge(r) if isinstance(r, list) else r)


def aciertos(preds, gt_por_img, ids):
    """Score y acierto (1/0) de cada predicción guardada; emparejamiento voraz por score con IoU >= 0.5, como COCO."""
    por_img = defaultdict(list)
    for p in preds:
        por_img[int(p["image_id"])].append(p)
    sc, ok = [], []
    for iid in ids:
        g = gt_por_img.get(iid, [])
        todas = sorted(por_img[iid], key=lambda p: -p["score"])
        if not todas:
            continue
        iou_all = mu.iou([p["segmentation"] for p in todas], g, [0] * len(g)) if g else None
        usado = set()
        for j, p in enumerate(todas):
            acierto = 0
            if g:
                fila = np.asarray(iou_all[j]).copy(); fila[list(usado)] = -1
                b = int(np.argmax(fila))
                if fila[b] >= 0.5:
                    usado.add(b); acierto = 1
            sc.append(p["score"]); ok.append(acierto)
    return np.array(sc, float), np.array(ok, float)
corr = pd.read_csv(CSV)
umbral = {(r.modelo, int(r.semilla)): float(r.umbral_score) for r in corr.itertuples()}

# ---------------------------------------------------------------- pasada por corrida
cal_filas, det = [], {}  # det[(m, s)][iid] = lista de máscaras sobre el umbral, orden por score
img_filas = []  # (m, s, iid, grupo, n_gt, n_pred, f1)
fn_tornillo = []  # (m, s, iid, j_gt, detectado)
for m in BENCH:
    for s in SEEDS:
        preds = json.loads((RUNS / f"{m}_s{s}" / "preds_test.json").read_text())
        por_img = defaultdict(list)
        for p in preds:
            por_img[int(p["image_id"])].append(p)
        u = umbral[(m, s)]; det[(m, s)] = {}
        sc, ok, gr = [], [], []
        for iid in imgs:
            g = gts.get(iid, [])
            todas = sorted(por_img[iid], key=lambda p: -p["score"])
            if todas:  # calibración: voraz por score como COCO, igual que analisis_extra_e2.py
                ar = [p["segmentation"] for p in todas]
                usado = set()
                iou_all = mu.iou(ar, g, [0] * len(g)) if g else None
                for j, p in enumerate(todas):
                    acierto = 0
                    if g:
                        fila = np.asarray(iou_all[j]).copy(); fila[list(usado)] = -1
                        b = int(np.argmax(fila))
                        if fila[b] >= 0.5:
                            usado.add(b); acierto = 1
                    sc.append(p["score"]); ok.append(acierto); gr.append(grupo_img[iid])
            dr = [p["segmentation"] for p in todas if p["score"] >= u]
            det[(m, s)][iid] = dr
            f1, tp, fp, fn, emp_g = f1_hungaro(dr, g)
            img_filas.append((m, s, iid, grupo_img[iid], len(g), len(dr), f1))
            fn_tornillo += [(m, s, iid, j, int(j in emp_g)) for j in range(len(g))]
        sc, ok, gr = np.array(sc), np.array(ok, float), np.array(gr)
        # recalibración ajustada solo en validación y aplicada a test
        sv, okv = aciertos(json.loads((RUNS / f"{m}_s{s}" / "preds_val.json").read_text()), vgts, vimgs)
        T = ajustar_T(sv, okv)
        s_ts = 1 / (1 + np.exp(-logit(sc) / T))
        b1, b0 = pendiente_intercepto(sv, okv)  # Platt: corrige escala y sesgo
        s_pl = 1 / (1 + np.exp(-(b0 + b1 * logit(sc))))
        pe, it = pendiente_intercepto(sc, ok)
        pe2, it2 = pendiente_intercepto(s_ts, ok)
        cal_filas.append(dict(modelo=m, semilla=s, n_pred=len(sc), n_pred_val=len(sv), T=T, ECE_val=ece_fijo(sv, okv),
                              ECE=ece_fijo(sc, ok), ECE_adapt=ece_adaptativo(sc, ok), Brier=np.mean((sc - ok) ** 2),
                              pendiente=pe, intercepto=it,
                              ECE_TS=ece_fijo(s_ts, ok), ECE_adapt_TS=ece_adaptativo(s_ts, ok),
                              Brier_TS=np.mean((s_ts - ok) ** 2), pendiente_TS=pe2, intercepto_TS=it2,
                              ECE_Platt=ece_fijo(s_pl, ok), ECE_adapt_Platt=ece_adaptativo(s_pl, ok),
                              Brier_Platt=np.mean((s_pl - ok) ** 2)))
    print("listo", m, flush=True)

cal = pd.DataFrame(cal_filas)
cal.to_csv(OUT / "calibracion_completa_por_corrida.csv", index=False)
cols = ["ECE", "ECE_TS", "ECE_adapt", "ECE_adapt_TS", "Brier", "Brier_TS", "pendiente", "pendiente_TS",
        "intercepto", "intercepto_TS", "T", "ECE_val", "ECE_Platt", "ECE_adapt_Platt", "Brier_Platt"]
calm = cal.groupby("modelo")[cols].mean().sort_values("ECE")
calm.to_csv(OUT / "calibracion_completa.csv")
IMG = pd.DataFrame(img_filas, columns=["modelo", "semilla", "image_id", "grupo", "n_gt", "n_pred", "f1"])

# ---------------------------------------------------------------- 2. índice de robustez
R = pd.concat([pd.read_csv(p) for p in sorted((ROB / "robustez").glob("*.csv"))], ignore_index=True)
fg = R.groupby(["modelo", "semilla", "perturbacion", "nivel", "grupo"]).f1.mean() \
      .groupby(["modelo", "semilla", "perturbacion", "nivel"]).mean().reset_index()
filas = []
for (m, s), d in fg.groupby(["modelo", "semilla"]):
    limpio = d[d.perturbacion == "limpio"].f1.iloc[0]
    for pert in ["ruido", "desenfoque"]:
        ret = np.r_[1.0, d[d.perturbacion == pert].sort_values("nivel").f1.values / limpio]
        filas.append((m, s, pert, np.trapezoid(ret, dx=1) / (len(ret) - 1)))
ir = pd.DataFrame(filas, columns=["modelo", "semilla", "perturbacion", "indice"])
irt = ir.groupby(["modelo", "perturbacion"]).indice.agg(["mean", "std"]).unstack()
irt.columns = [f"{a}_{b}" for a, b in irt.columns]
irt.to_csv(OUT / "indice_robustez.csv")

# ---------------------------------------------------------------- 3. δ de Cliff y probabilidad de superioridad
fgs = pd.read_csv(AN / "f1_grupo_por_semilla.csv")
fgs = fgs[fgs.modelo.isin(BENCH)]
fgm = fgs.groupby(["modelo", "grupo"]).f1_grupo.mean().unstack()  # modelo x grupo, media de semillas
base = fgm.loc[COMPARADOR].values
filas = []
for m in BENCH:
    if m == COMPARADOR:
        continue
    x = fgm.loc[m].values
    mayor = (x[:, None] > base[None, :]).mean(); menor = (x[:, None] < base[None, :]).mean()
    ps = np.mean((x > base) + 0.5 * (x == base))
    lo, hi = ic_boot(np.c_[x, base], lambda z: np.mean((z[:, 0] > z[:, 1]) + 0.5 * (z[:, 0] == z[:, 1])))
    filas.append(dict(modelo=m, delta_cliff=mayor - menor, A12=mayor + 0.5 * (1 - mayor - menor),
                      P_superioridad_pareada=ps, ic95_lo=lo, ic95_hi=hi, grupos=len(x)))
pd.DataFrame(filas).to_csv(OUT / "cliff_superioridad.csv", index=False)

# ---------------------------------------------------------------- 4. descomposición de la varianza
d = fgs.copy(); d["semilla"] = d.semilla.astype(str); d["grupo"] = d.grupo.astype(str); d["uno"] = 1
mm = smf.mixedlm(f"f1_grupo ~ C(modelo, Treatment('{COMPARADOR}'))", d, groups="uno",
                 vc_formula={"grupo": "0 + C(grupo)", "semilla": "0 + C(semilla)"}).fit(reml=True)
vc = dict(zip(mm.model.exog_vc.names, mm.vcomp))
var_modelo = fgm.mean(axis=1).var(ddof=1)  # varianza entre medias de modelo (efecto fijo)
comp = pd.DataFrame({"componente": ["modelo (fijo, varianza de las medias)", "grupo de escena (aleatorio)",
                                    "semilla (aleatorio)", "residual"],
                     "varianza": [var_modelo, vc["grupo"], vc["semilla"], mm.scale]})
comp["porcentaje"] = 100 * comp.varianza / comp.varianza.sum()
comp.to_csv(OUT / "varianza_modelo_mixto.csv", index=False)
fe = mm.fe_params.drop("Intercept"); ci = mm.conf_int().loc[fe.index]; pv = mm.pvalues.loc[fe.index]
holm = np.minimum(1, np.maximum.accumulate(np.sort(pv.values) * (len(pv) - np.arange(len(pv)))))
holm = pd.Series(holm, index=pv.sort_values().index).loc[fe.index]
sd_res = np.sqrt(mm.scale + vc["grupo"] + vc["semilla"])
contr = pd.DataFrame({"modelo": [i.split("T.")[-1].rstrip("]") for i in fe.index], "dif_vs_comparador": fe.values,
                      "ic95_lo": ci[0].values, "ic95_hi": ci[1].values, "p": pv.values, "p_holm": holm.values,
                      "dif_estandarizada": fe.values / sd_res})
contr.to_csv(OUT / "contrastes_modelo_mixto_holm.csv", index=False)
# clásica, diseño balanceado modelo x grupo x semilla
y = fgs.pivot_table(index=["modelo", "grupo"], columns="semilla", values="f1_grupo")
A = y.values.reshape(len(BENCH), -1, len(SEEDS))  # modelo, grupo, semilla (modelos en orden alfabético)
gm = A.mean()
ss = {"modelo": A.shape[1] * A.shape[2] * ((A.mean((1, 2)) - gm) ** 2).sum(),
      "grupo": A.shape[0] * A.shape[2] * ((A.mean((0, 2)) - gm) ** 2).sum(),
      "semilla": A.shape[0] * A.shape[1] * ((A.mean((0, 1)) - gm) ** 2).sum()}
mg = A.mean(2)
ss["modelo x grupo"] = A.shape[2] * ((mg - mg.mean(1, keepdims=True) - mg.mean(0, keepdims=True) + gm) ** 2).sum()
ss["resto (semilla dentro de modelo y grupo)"] = ((A - gm) ** 2).sum() - sum(ss.values())
cl = pd.DataFrame({"fuente": list(ss), "suma_cuadrados": list(ss.values())})
cl["porcentaje"] = 100 * cl.suma_cuadrados / cl.suma_cuadrados.sum()
cl.to_csv(OUT / "varianza_sumas_cuadrados.csv", index=False)
de_sem = fgs.groupby(["modelo", "semilla"]).f1_grupo.mean().groupby("modelo").std()
json.dump({"DE_entre_semillas_media": float(de_sem.mean()), "DE_entre_semillas_sin_mask2former":
           float(de_sem.drop("mask2former").mean()), "DE_entre_modelos": float(fgm.mean(axis=1).std()),
           "varianzas_aleatorias": {k: float(v) for k, v in vc.items()}, "residual": float(mm.scale)},
          open(OUT / "varianza_resumen.json", "w"), indent=1)

# ---------------------------------------------------------------- 5. Pareto
c = corr[corr.modelo.isin(BENCH)].groupby("modelo").agg(F1=("test_F1_grupo", "mean"), parametros=("parametros", "first"),
                                                         entrenamiento_min=("s_total", lambda v: v.mean() / 60))
lat = pd.read_csv(AX / "latencia_tabla.csv").set_index("modelo")
c = c.join(lat[["mediana_ms", "gflops_una_imagen"]])


def no_dominados(df, costo):
    df = df.dropna(subset=[costo]); nd = []
    for m, r in df.iterrows():
        dom = ((df[costo] <= r[costo]) & (df.F1 >= r.F1) & ((df[costo] < r[costo]) | (df.F1 > r.F1))).any()
        nd.append((m, not dom))
    return dict(nd)


for costo in ["parametros", "entrenamiento_min", "mediana_ms", "gflops_una_imagen"]:
    c[f"pareto_{costo}"] = pd.Series(no_dominados(c, costo))
c.sort_values("F1", ascending=False).to_csv(OUT / "pareto.csv")
DESPLAZA = {"maskrcnn": (-48, 3), "htc": (4, 5), "cascade": (4, -9), "msrcnn": (-40, -4), "pointrend": (4, -10),
            "yolo26": (4, 4), "yolov8": (-36, 5), "yolo11": (4, -11), "yolov7": (4, 2)}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
for a, costo, et, lg in [(ax[0], "parametros", "parámetros (millones)", True),
                         (ax[1], "entrenamiento_min", "minutos de entrenamiento por corrida (6 épocas)", False),
                         (ax[2], "mediana_ms", "latencia mediana en la 3060 (ms/imagen)", False)]:
    dd = c.dropna(subset=[costo]); x = dd[costo] / (1e6 if costo == "parametros" else 1)
    nd = dd[f"pareto_{costo}"].astype(bool)
    a.scatter(x[~nd], dd.F1[~nd], c="0.6"); a.scatter(x[nd], dd.F1[nd], c="C3")
    o = np.argsort(x[nd].values); a.plot(x[nd].values[o], dd.F1[nd].values[o], c="C3", lw=1)
    for m in dd.index:
        a.annotate(m, (x[m], dd.F1[m]), fontsize=7, xytext=DESPLAZA.get(m, (4, 2)), textcoords="offset points")
    a.set_xlabel(et); a.set_ylabel("F1 de grupo en test")
    if lg:
        a.set_xscale("log"); a.set_xticks([3, 10, 30, 80], ["3", "10", "30", "80"])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[2].set_title(f"{int(c.mediana_ms.notna().sum())} modelos con latencia medida (YOLOv7 y YOLOv9 usan otro método: no entran)", fontsize=8)
fig.suptitle("Frontera de Pareto: en rojo los modelos no dominados"); fig.tight_layout()
fig.savefig(OUT / "pareto.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 6. errores compartidos y ensambles
err = IMG.groupby(["modelo", "image_id"]).f1.mean().unstack(0)  # imagen x modelo
err = 1 - err[BENCH]
corr_err = err.corr(method="spearman")
corr_err.to_csv(OUT / "correlacion_errores_spearman.csv")
tri = corr_err.values[np.triu_indices(len(BENCH), 1)]
FN = pd.DataFrame(fn_tornillo, columns=["modelo", "semilla", "image_id", "j", "detectado"])
tasa = FN.groupby(["image_id", "j", "modelo"]).detectado.mean().unstack()  # tornillo x modelo, fracción de semillas
fallan = (tasa[BENCH] < 0.5).sum(axis=1)
cortes = json.loads((AX / "rq3_cortes.json").read_text())["cortes_area_px_val"]  # cuántos de los 13 lo pierden en la mayoría de semillas
compart = {"n_tornillos_test": int(len(tasa)),
           "perdidos_por_al_menos_un_modelo": int((fallan >= 1).sum()),
           "perdidos_por_10_o_mas_de_13": int((fallan >= 10).sum()),
           "perdidos_por_los_13": int((fallan == 13).sum()),
           "correlacion_errores_media": float(tri.mean()), "correlacion_errores_min": float(tri.min()),
           "correlacion_errores_max": float(tri.max())}
t3 = tasa[TRES_MEJORES] < 0.5
compart["tres_mejores_perdidos_por_alguno"] = int(t3.any(axis=1).sum())
compart["tres_mejores_perdidos_por_los_tres"] = int(t3.all(axis=1).sum())
compart["tres_mejores_fraccion_compartida"] = compart["tres_mejores_perdidos_por_los_tres"] / max(
    1, compart["tres_mejores_perdidos_por_alguno"])
json.dump(compart, open(OUT / "errores_compartidos.json", "w"), indent=1)
dificiles = tasa[fallan >= 10].reset_index()[["image_id", "j"]]
dificiles["archivo"] = dificiles.image_id.map(lambda i: imgs[i]["file_name"])
dificiles["grupo"] = dificiles.image_id.map(grupo_img)
dificiles["modelos_que_lo_pierden"] = fallan[fallan >= 10].values
dificiles["area_px"] = [float(mu.area(gts[i][j])) for i, j in zip(dificiles.image_id, dificiles.j)]
dificiles["tercil"] = [["pequeño", "mediano", "grande"][int(np.searchsorted(cortes, a, side="right"))]
                       for a in dificiles.area_px]
dificiles.to_csv(OUT / "tornillos_dificiles_para_todos.csv", index=False)


def ensamble(miembros, s):
    """Voto por mayoría: une las máscaras de los miembros que se solapan (IoU >= 0.5, a lo sumo una por miembro) y
    conserva el grupo si lo detectan al menos 2; la máscara que queda es la del miembro de mejor F1 en val."""
    salida = {}
    for iid in imgs:
        cands = [(k, r) for k, mm_ in enumerate(miembros) for r in det[(mm_, s)][iid]]
        usados, fusion = set(), []
        for a, (ka, ra) in enumerate(cands):
            if a in usados:
                continue
            grupo_c = {ka: ra}; usados.add(a)
            for b, (kb, rb) in enumerate(cands):
                if b in usados or kb in grupo_c:
                    continue
                if mu.iou([ra], [rb], [0])[0][0] >= 0.5:
                    grupo_c[kb] = rb; usados.add(b)
            if len(grupo_c) >= 2:
                fusion.append(grupo_c[min(grupo_c)])
        salida[iid] = fusion
    return salida


filas = []
for nombre, miembros in ENSAMBLES.items():
    for s in SEEDS:
        e = ensamble(miembros, s)
        f = [(grupo_img[i], f1_hungaro(e[i], gts.get(i, []))[0]) for i in imgs]
        fe_g = pd.DataFrame(f, columns=["grupo", "f1"]).groupby("grupo").f1.mean()
        for g, v in fe_g.items():
            filas.append((nombre, s, g, v))
EN = pd.DataFrame(filas, columns=["ensamble", "semilla", "grupo", "f1_grupo"])
res = []
for nombre, miembros in ENSAMBLES.items():
    eg = EN[EN.ensamble == nombre].groupby("grupo").f1_grupo.mean()
    solo = fgm.loc[miembros].T  # grupo x miembro
    mejor = solo.mean().idxmax()
    oraculo = IMG[IMG.modelo.isin(miembros)].groupby(["semilla", "image_id"]).f1.max().reset_index()
    oraculo["grupo"] = oraculo.image_id.map(grupo_img)
    orc = oraculo.groupby(["semilla", "grupo"]).f1.mean().groupby("semilla").mean().mean()
    dif = eg.values - solo[mejor].loc[eg.index].values
    res.append(dict(ensamble=nombre, miembros="+".join(miembros), F1_ensamble=eg.mean(), mejor_miembro=mejor,
                    F1_mejor_miembro=solo[mejor].mean(), ganancia=dif.mean(),
                    ic95_lo=ic_boot(dif)[0], ic95_hi=ic_boot(dif)[1],
                    p_wilcoxon=stats.wilcoxon(dif).pvalue if np.any(dif) else 1.0,
                    F1_oraculo_por_imagen=orc, ganancia_potencial_oraculo=orc - solo[mejor].mean()))
pd.DataFrame(res).to_csv(OUT / "ensambles.csv", index=False)
fig, a = plt.subplots(figsize=(7, 6))
im = a.imshow(corr_err.values, vmin=0, vmax=1, cmap="viridis")
a.set_xticks(range(len(BENCH)), BENCH, rotation=90, fontsize=8); a.set_yticks(range(len(BENCH)), BENCH, fontsize=8)
fig.colorbar(im, label="correlación de Spearman de 1 - F1 por imagen")
a.set_title("Correlación de errores por imagen entre modelos"); fig.tight_layout()
fig.savefig(OUT / "correlacion_errores.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 7. incertidumbre predictiva (conteo de tornillos)
filas = []
for m in TRES_MEJORES:
    dm = IMG[IMG.modelo == m]
    for s in SEEDS:
        d = dm[dm.semilla == s].copy(); d["r"] = (d.n_pred - d.n_gt).abs()
        for g0 in d.grupo.unique():
            cal_ = d[d.grupo != g0].r.values; n = len(cal_)
            q = np.sort(cal_)[min(n - 1, int(np.ceil((n + 1) * (1 - ALFA))) - 1)]
            t = d[d.grupo == g0]
            for r in t.itertuples():
                lo, hi = max(0, r.n_pred - q), r.n_pred + q
                filas.append((m, "conforme_por_grupos", s, r.image_id, g0, r.n_gt, lo <= r.n_gt <= hi, hi - lo + 1))
    rango = dm.groupby("image_id").n_pred.agg(["min", "max"])
    for iid, r in rango.iterrows():
        ng = len(gts.get(iid, []))
        filas.append((m, "rango_5_semillas", -1, iid, grupo_img[iid], ng, r["min"] <= ng <= r["max"],
                      r["max"] - r["min"] + 1))
U = pd.DataFrame(filas, columns=["modelo", "metodo", "semilla", "image_id", "grupo", "n_gt", "cubre", "ancho"])
U["estrato_n_gt"] = pd.cut(U.n_gt, [-1, 2, 5, 100], labels=["1-2", "3-5", "6+"])
res = []
for (m, met), d in U.groupby(["modelo", "metodo"]):
    pg = d.groupby("grupo").cubre.mean()
    lo, hi = ic_boot(pg.values)
    fila = dict(modelo=m, metodo=met, cobertura_nominal=1 - ALFA if met.startswith("conforme") else np.nan,
                cobertura=d.cubre.mean(), ic95_lo_por_grupos=lo, ic95_hi_por_grupos=hi,
                ancho_medio_tornillos=d.ancho.mean(), cobertura_min_grupo=pg.min(), grupo_peor=pg.idxmin())
    for e, de in d.groupby("estrato_n_gt", observed=True):
        fila[f"cobertura_n_gt_{e}"] = de.cubre.mean()
    res.append(fila)
pd.DataFrame(res).to_csv(OUT / "incertidumbre_cobertura.csv", index=False)
U.groupby(["modelo", "metodo", "grupo"]).cubre.mean().unstack(0).to_csv(OUT / "incertidumbre_cobertura_por_grupo.csv")
print("hecho:", OUT)
