"""Analisis estadistico del E2 segun preregistro_E2.md, seccion 5.

Uso:  python analisis_e2.py <carpeta_corridas> <corridas.csv> <carpeta_salida>

Entrada: <carpeta_corridas>/<modelo>_s<semilla>/test_por_imagen.csv (columnas image_id, grupo, f1, ...).
Unidad de analisis: el grupo de escena (14 en test). F1 por grupo = media del F1 de sus imagenes;
luego promedio de las 5 semillas -> un valor por modelo y grupo.
"""
import sys, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

RUNS, CSV, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = range(42, 47)
TOP10 = ["mask2former", "yolov7", "yolov9", "yolo26", "yolo11",
         "cascade", "htc", "pointrend", "msrcnn", "condinst"]
BASE = ["maskrcnn", "yolov8", "solov2"]
NOPROF = ["clasico"]
R_FAM = ["cascade", "htc", "pointrend", "msrcnn"]
P_FAM = ["yolov7", "yolov9", "yolo26", "yolo11"]
DELTA, ALPHA, B = 0.20, 0.05, 10000
rng = np.random.default_rng(0)

# ---- 1. F1 por grupo, por modelo y semilla --------------------------------
rows = []
for m in TOP10 + BASE + NOPROF:
    for s in SEEDS:
        d = pd.read_csv(RUNS / f"{m}_s{s}" / "test_por_imagen.csv")
        g = d.groupby("grupo")["f1"].mean()
        for grupo, v in g.items():
            rows.append((m, s, grupo, v, int((d.grupo == grupo).sum())))
por_semilla = pd.DataFrame(rows, columns=["modelo", "semilla", "grupo", "f1_grupo", "n_img"])
por_semilla.to_csv(OUT / "f1_grupo_por_semilla.csv", index=False)
F = por_semilla.pivot_table(index="grupo", columns="modelo", values="f1_grupo", aggfunc="mean")
F.to_csv(OUT / "f1_grupo_promedio_semillas.csv")
n_img = por_semilla.drop_duplicates("grupo").set_index("grupo")["n_img"]
assert F.shape[0] == 14, F.shape

# ---- 2. Verificacion contra corridas.csv ----------------------------------
c = pd.read_csv(CSV)
c = c[c.semilla.between(42, 46)]
chk = c.groupby("modelo")["test_F1_grupo"].mean()
mine = por_semilla.groupby(["modelo", "semilla"])["f1_grupo"].mean().groupby("modelo").mean()
verif = pd.DataFrame({"corridas_csv": chk, "recalculado": mine}).dropna()
verif["dif_abs"] = (verif.corridas_csv - verif.recalculado).abs()
verif.to_csv(OUT / "verificacion_vs_corridas_csv.csv")
print("max |dif| contra corridas.csv:", verif.dif_abs.max())

# ---- 3. Mejor baseline fuerte (por VAL, regla preregistro 5) --------------
val = c.groupby("modelo")["val_F1_grupo"].mean()
cand = val[BASE + NOPROF]
mejor_base = cand.idxmax()
print("Mejor baseline por val:", mejor_base, round(cand.max(), 4))


def boot_ci(diff, B=B):
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj


def bh(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 1
    for rank in range(m - 1, -1, -1):
        i = o[rank]; run = min(run, p[i] * m / (rank + 1)); adj[i] = run
    return adj


def decision(p_holm, diff, lo, hi):
    if p_holm < ALPHA and diff >= DELTA and lo > 0:
        return "significativa y relevante"
    if p_holm < ALPHA:
        return "significativa sin relevancia"
    if lo > -DELTA and hi < DELTA:
        return "equivalente"
    return "no concluyente"


# ---- 4. RQ1: Top 10 vs mejor baseline, Wilcoxon unilateral + Holm ---------
rq1 = []
for m in TOP10:
    d = (F[m] - F[mejor_base]).to_numpy()
    try:
        w = stats.wilcoxon(d, alternative="greater", method="auto")
        p_w = w.pvalue
    except ValueError:  # todas las diferencias 0
        p_w = 1.0
    t = stats.ttest_rel(F[m], F[mejor_base], alternative="greater")
    sw = stats.shapiro(d)
    lo, hi = boot_ci(d)
    rq1.append(dict(modelo=m, F1_modelo=F[m].mean(), F1_base=F[mejor_base].mean(),
                    dif_media=d.mean(), dif_mediana=np.median(d), ic95_lo=lo, ic95_hi=hi,
                    p_wilcoxon=p_w, p_t_sensib=t.pvalue, shapiro_p=sw.pvalue,
                    grupos_mejores=int((d > 0).sum()), grupos_peores=int((d < 0).sum())))
rq1 = pd.DataFrame(rq1)
rq1["p_holm"] = holm(rq1.p_wilcoxon)
rq1["decision"] = [decision(r.p_holm, r.dif_media, r.ic95_lo, r.ic95_hi) for r in rq1.itertuples()]
rq1.sort_values("dif_media", ascending=False).to_csv(OUT / "rq1_top10_vs_baseline.csv", index=False)

# ---- 5. RQ2: familia R vs P, bilateral ------------------------------------
dR = F[R_FAM].mean(axis=1) - F[P_FAM].mean(axis=1)
w2 = stats.wilcoxon(dR, alternative="two-sided")
lo2, hi2 = boot_ci(dR.to_numpy())
rq2 = dict(media_R=F[R_FAM].mean(axis=1).mean(), media_P=F[P_FAM].mean(axis=1).mean(),
           dif_media=dR.mean(), dif_mediana=dR.median(), ic95_lo=lo2, ic95_hi=hi2,
           p_wilcoxon=w2.pvalue, grupos_R_mejor=int((dR > 0).sum()),
           shapiro_p=stats.shapiro(dR).pvalue, p_t_sensib=stats.ttest_1samp(dR, 0).pvalue)
json.dump(rq2, open(OUT / "rq2_R_vs_P.json", "w"), indent=1)

# ---- 6. Friedman global + Nemenyi (15 modelos sin 'vacio') ----------------
modelos = TOP10 + BASE + NOPROF
rangos = F[modelos].rank(axis=1, ascending=False)  # 1 = mejor
fr = stats.friedmanchisquare(*[F[m] for m in modelos])
k, n = len(modelos), len(F)
q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
CD = q * np.sqrt(k * (k + 1) / (6 * n))
rang_medio = rangos.mean().sort_values()
pd.DataFrame({"rango_medio": rang_medio}).to_csv(OUT / "friedman_rangos.csv")
json.dump(dict(chi2=fr.statistic, p=fr.pvalue, k=k, n_grupos=n, CD_nemenyi=CD),
          open(OUT / "friedman.json", "w"), indent=1)

# ---- 7. Exploratorio: todos los pares, Wilcoxon bilateral + BH ------------
pares = []
for a, b in itertools.combinations(modelos, 2):
    d = (F[a] - F[b]).to_numpy()
    try:
        p = stats.wilcoxon(d, alternative="two-sided").pvalue
    except ValueError:
        p = 1.0
    pares.append(dict(a=a, b=b, dif_media=d.mean(), p=p))
pares = pd.DataFrame(pares)
pares["p_BH"] = bh(pares.p)
pares.to_csv(OUT / "pares_exploratorio_BH.csv", index=False)

# ---- 8. Resumen de tabla principal ----------------------------------------
tab = pd.DataFrame({
    "F1_grupo_test_media": F[modelos].mean(),
    "DE_entre_semillas": por_semilla.groupby(["modelo", "semilla"])["f1_grupo"].mean()
                          .groupby("modelo").std(),
}).sort_values("F1_grupo_test_media", ascending=False)
tab.to_csv(OUT / "tabla_principal.csv")

# ---- 9. Figuras ------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = rq1.sort_values("dif_media")
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.errorbar(r.dif_media, range(len(r)), xerr=[r.dif_media - r.ic95_lo, r.ic95_hi - r.dif_media],
            fmt="o", capsize=3)
ax.axvline(0, color="k", lw=.8); ax.axvline(DELTA, color="r", ls="--", lw=.8)
ax.axvline(-DELTA, color="r", ls="--", lw=.8)
ax.set_yticks(range(len(r))); ax.set_yticklabels(r.modelo)
ax.set_xlabel(f"Diferencia de F1 por grupo vs {mejor_base} (IC 95 % bootstrap de grupos)")
ax.set_title("RQ1: Top 10 contra el mejor baseline (líneas rojas: ±δ = 0.20)")
fig.tight_layout(); fig.savefig(OUT / "rq1_diferencias.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(range(k), rang_medio.values, color="#4c72b0")
ax.set_yticks(range(k)); ax.set_yticklabels(rang_medio.index); ax.invert_yaxis()
ax.set_xlabel(f"Rango medio sobre {n} grupos (1 = mejor). CD de Nemenyi = {CD:.2f}")
ax.set_title("Friedman: rangos medios")
fig.tight_layout(); fig.savefig(OUT / "friedman_rangos.png", dpi=200); plt.close(fig)

print(tab.round(3)); print(rq1.round(4).to_string()); print(rq2); print(fr, CD)
