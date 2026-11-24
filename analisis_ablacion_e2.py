"""Análisis del estudio de ablación (enunciado §4.6; corridas de ablacion_e2.py).

Uso:
    python analisis_ablacion_e2.py results/e2/corridas results/e2/ablacion results/e2/analisis_ablacion

Para cada variante (Mask R-CNN sin un componente) y para el modelo completo: F1 por grupo en test con la misma
métrica primaria (F1 de instancia por imagen, promediado por grupo de escena), media de las semillas disponibles.
Inferencia sobre cada diferencia (variante − completo) con los 14 grupos de test como unidad: diferencia media,
IC 95 % bootstrap (2000 remuestreos de grupos), Wilcoxon bilateral, Holm sobre las variantes, d de Cohen pareada y
correlación biserial de rangos. Solo se usan las semillas que tienen tanto la variante como el completo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CORR, ABL, OUT = (Path(x) for x in sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
VARIANTES = {"anclas_defecto": "anclas de 32–512 px (por defecto)", "solo_imagenet": "preentrenamiento solo ImageNet",
             "sin_volteo": "sin volteo horizontal"}
rng = np.random.default_rng(0)


def f1_grupo(carpeta: Path) -> pd.Series:
    t = pd.read_csv(carpeta / "test_por_imagen.csv")
    return t.groupby("grupo").f1.mean()


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, p[i] * (m - k))); adj[i] = run
    return adj


filas, por_corrida = [], []
for v, desc in VARIANTES.items():
    semillas = sorted(int(c.name.split("_s")[-1]) for c in ABL.glob(f"{v}_s*") if (c / "resumen.json").exists())
    semillas = [s for s in semillas if (CORR / f"maskrcnn_s{s}" / "test_por_imagen.csv").exists()]
    if not semillas:
        print("sin corridas:", v); continue
    va = pd.concat([f1_grupo(ABL / f"{v}_s{s}") for s in semillas], axis=1).mean(axis=1)
    co = pd.concat([f1_grupo(CORR / f"maskrcnn_s{s}") for s in semillas], axis=1).mean(axis=1)
    for s in semillas:
        r = json.loads((ABL / f"{v}_s{s}" / "resumen.json").read_text(encoding="utf8"))
        por_corrida.append((v, s, f1_grupo(ABL / f"{v}_s{s}").mean(), f1_grupo(CORR / f"maskrcnn_s{s}").mean(),
                            r.get("mejor_epoca"), r.get("val_AP_sube_al_final"), r.get("s_total", np.nan) / 60))
    d = (va - co.loc[va.index]).values
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    w = stats.wilcoxon(d) if np.any(d) else None
    pos, neg = d[d > 0], d[d < 0]
    rk = stats.rankdata(np.abs(d[d != 0]))
    wp, wn = rk[d[d != 0] > 0].sum(), rk[d[d != 0] < 0].sum()
    filas.append(dict(variante=v, descripcion=desc, semillas=len(semillas), F1_variante=va.mean(), F1_completo=co.mean(),
                      dif_media=d.mean(), ic95_lo=np.percentile(boot, 2.5), ic95_hi=np.percentile(boot, 97.5),
                      p_wilcoxon=w.pvalue if w else 1.0, grupos_mejor=int((d > 0).sum()), grupos_peor=int((d < 0).sum()),
                      d_cohen_pareada=d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
                      r_biserial=(wp - wn) / (wp + wn) if wp + wn > 0 else 0.0))
res = pd.DataFrame(filas)
if len(res):
    res["p_holm"] = holm(res.p_wilcoxon)
    res["aporte_del_componente"] = -res.dif_media  # cuánto F1 se pierde al quitarlo
res.to_csv(OUT / "ablacion.csv", index=False)
pd.DataFrame(por_corrida, columns=["variante", "semilla", "F1_grupo_variante", "F1_grupo_completo", "mejor_epoca",
                                   "val_AP_sube_al_final", "minutos"]).to_csv(OUT / "ablacion_por_corrida.csv", index=False)
print(res.round(4).to_string())
