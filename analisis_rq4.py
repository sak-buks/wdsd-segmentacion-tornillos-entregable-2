"""RQ4 (sobreajuste, preregistro 5) y curvas por epoca del E2.

Uso:  python analisis_rq4.py <carpeta_corridas> <carpeta_salida>

Brecha = F1 de instancia por grupo en una submuestra fija de 93 imagenes de train (train93) menos el F1 por grupo en val,
con el mismo pipeline y el umbral de score elegido en val. Se promedian primero las 5 semillas por grupo.
IC 95 % por bootstrap de grupos: se remuestrean por separado los grupos de train93 y los de val (10 000 replicas).
Se calcula tambien con el F1 por imagen (promedio simple de imagenes, bootstrap por grupos): el preregistro dice "F1" sin
precisar si es por imagen o por grupo, y la lectura cambia, asi que se reportan las dos.
Regla de lectura (operacionalizacion nuestra; el preregistro fija solo la metrica y delta_g = 0.15):
 - "sobreajuste > delta_g": limite inferior del IC > 0.15
 - "dentro de delta_g": limite superior del IC < 0.15
 - si no, "no concluyente".
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
MODELOS = ["maskrcnn","htc","cascade","msrcnn","pointrend","yolov7","condinst","yolo26","yolov8","mask2former","yolo11","yolov9","solov2","clasico"]
SEEDS = range(42, 47); DG = 0.15; B = 10000
rng = np.random.default_rng(0)

def f1_grupo(m, nombre):
    out = []
    for s in SEEDS:
        d = pd.read_csv(RUNS / f"{m}_s{s}" / nombre)
        out.append(d.groupby("grupo")["f1"].mean().rename(s))
    return pd.concat(out, axis=1).mean(axis=1)   # promedio de semillas por grupo

def img_stats(m, nombre):
    """F1 por imagen promediado sobre semillas; devuelve suma y n por grupo."""
    d = pd.concat([pd.read_csv(RUNS / f"{m}_s{s}" / nombre)[["archivo", "grupo", "f1"]] for s in SEEDS])
    d = d.groupby(["archivo", "grupo"], as_index=False)["f1"].mean()
    g = d.groupby("grupo")["f1"].agg(["sum", "count"])
    return g["sum"].to_numpy(), g["count"].to_numpy()

def brecha_img(m):
    st, nt = img_stats(m, "train93_por_imagen.csv"); sv, nv = img_stats(m, "val_por_imagen.csv")
    gap = st.sum() / nt.sum() - sv.sum() / nv.sum()
    it = rng.integers(0, len(st), (B, len(st))); iv = rng.integers(0, len(sv), (B, len(sv)))
    dist = st[it].sum(1) / nt[it].sum(1) - sv[iv].sum(1) / nv[iv].sum(1)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return st.sum() / nt.sum(), sv.sum() / nv.sum(), gap, lo, hi

filas = []
for m in MODELOS:
    tr, va = f1_grupo(m, "train93_por_imagen.csv"), f1_grupo(m, "val_por_imagen.csv")
    gap = tr.mean() - va.mean()
    bt = rng.integers(0, len(tr), (B, len(tr))); bv = rng.integers(0, len(va), (B, len(va)))
    dist = tr.to_numpy()[bt].mean(1) - va.to_numpy()[bv].mean(1)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    # brecha por semilla (para ver la variacion)
    gs = []
    for s in SEEDS:
        a = pd.read_csv(RUNS / f"{m}_s{s}" / "train93_por_imagen.csv").groupby("grupo")["f1"].mean().mean()
        b = pd.read_csv(RUNS / f"{m}_s{s}" / "val_por_imagen.csv").groupby("grupo")["f1"].mean().mean()
        gs.append(a - b)
    lec = "sobreajuste > delta_g" if lo > DG else ("dentro de delta_g" if hi < DG else "no concluyente")
    ti, vi, gi, loi, hii = brecha_img(m)
    leci = "sobreajuste > delta_g" if loi > DG else ("dentro de delta_g" if hii < DG else "no concluyente")
    filas.append(dict(modelo=m, F1_train93=tr.mean(), F1_val=va.mean(), brecha=gap, ic95_lo=lo, ic95_hi=hi,
                      brecha_min_semilla=min(gs), brecha_max_semilla=max(gs), grupos_train93=len(tr), grupos_val=len(va), lectura=lec,
                      F1img_train93=ti, F1img_val=vi, brecha_img=gi, img_ic95_lo=loi, img_ic95_hi=hii, lectura_img=leci))
rq4 = pd.DataFrame(filas).sort_values("brecha", ascending=False)
rq4.to_csv(OUT / "rq4_brecha.csv", index=False)

# curvas por epoca (el clasico y el vacio no tienen historial)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = {}
for m in MODELOS:
    if m == "clasico": continue
    H[m] = pd.concat([pd.read_csv(RUNS / f"{m}_s{s}" / "historial.csv").assign(semilla=s) for s in SEEDS])
for col, titulo, nombre in [("val_AP", "AP de máscara 0.5:0.95 en validación", "curvas_val_AP.png"), ("perdida", "Pérdida de entrenamiento", "curvas_perdida.png")]:
    fig, axs = plt.subplots(4, 4, figsize=(12, 9), sharex=True)
    for ax, m in zip(axs.ravel(), H):
        d = H[m]
        for s, g in d.groupby("semilla"): ax.plot(g.epoca, g[col], color="#9db4d6", lw=.8)
        mu = d.groupby("epoca")[col].mean(); ax.plot(mu.index, mu.values, color="#1f4e8c", lw=2)
        ax.set_title(m, fontsize=9); ax.grid(alpha=.3)
    for ax in axs.ravel()[len(H):]: ax.axis("off")
    for ax in axs[-1]: ax.set_xlabel("época")
    fig.suptitle(titulo + " (gris: semillas; azul: media)"); fig.tight_layout(); fig.savefig(OUT / nombre, dpi=160); plt.close(fig)

# subentrenamiento: la curva de val sigue subiendo en la ultima epoca (flag val_AP_sube_al_final de resumen.json) y cuanto sube
sub = []
for m in H:
    flags = [json.load(open(RUNS / f"{m}_s{s}" / "resumen.json")).get("val_AP_sube_al_final") for s in SEEDS]
    d = H[m]; ult = d[d.epoca == d.epoca.max()].val_AP.mean(); pen = d[d.epoca == d.epoca.max() - 1].val_AP.mean()
    sub.append(dict(modelo=m, semillas_con_AP_val_subiendo=int(sum(bool(x) for x in flags)), AP_val_ultima=ult, AP_val_penultima=pen, ganancia_ultima_epoca=ult - pen))
sub = pd.DataFrame(sub).sort_values("ganancia_ultima_epoca", ascending=False)
sub.to_csv(OUT / "subentrenamiento.csv", index=False)
print(rq4[["modelo","brecha","ic95_lo","ic95_hi","lectura","F1img_train93","F1img_val","brecha_img","img_ic95_lo","img_ic95_hi","lectura_img"]].round(3).to_string()); print(sub.round(4).to_string())
