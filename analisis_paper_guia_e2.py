"""Análisis del paper guía (criterio_paper_guia_E2.md): réplica de Kalitsios et al. 2022 y su limitación.

Uso:
    python analisis_paper_guia_e2.py results/e2/paper_guia datos_oficial/coco/test.json \
        results/e2/auditoria_fuga/casi_gemelos_particion_oficial.csv results/e2/corridas.csv results/e2/analisis_paper_guia
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PG, TEST, GEM, CORR, OUT = (Path(x) for x in sys.argv[1:6])
OUT.mkdir(parents=True, exist_ok=True)
PAPER = {"AP": 0.586, "AP50": 0.849}
LIMITACION = ["msrcnn", "pointrend", "yolo26"]
B = 2000
rng = np.random.default_rng(0)

with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(str(TEST))
ids = sorted(gt.getImgIds())


def ap(preds, img_ids):
    """AP y AP50 de máscara COCO sobre img_ids (con repeticiones permitidas: se renumeran)."""
    g = COCO(); d = {"images": [], "annotations": [], "categories": gt.dataset["categories"]}
    pr = []
    por_img = defaultdict(list)
    for p in preds:
        por_img[p["image_id"]].append(p)
    for k, i in enumerate(img_ids, 1):
        d["images"].append({**gt.imgs[i], "id": k})
        for a in gt.imgToAnns[i]:
            d["annotations"].append({**a, "id": len(d["annotations"]) + 1, "image_id": k})
        pr += [{**p, "image_id": k} for p in por_img[i]]
    with contextlib.redirect_stdout(io.StringIO()):
        g.dataset = d; g.createIndex()
        if not pr:
            return 0.0, 0.0
        dt = g.loadRes(pr)
        e = COCOeval(g, dt, "segm"); e.evaluate(); e.accumulate(); e.summarize()
    return float(e.stats[0]), float(e.stats[1])


def preds(m, s):
    p = json.loads((PG / f"{m}_s{s}" / "preds_test.json").read_text())
    for x in p:
        x["image_id"] = int(x["image_id"]); x["score"] = float(x["score"])
    return p


def semillas(m):
    return sorted(int(c.name.rsplit("_s", 1)[1]) for c in PG.glob(f"{m}_s*") if (c / "resumen.json").exists())


res = {}
# ---------------------------------------------------------------- réplica
s_r = semillas("maskrcnn_r101")
if s_r:
    P = {s: preds("maskrcnn_r101", s) for s in s_r}
    obs = np.mean([ap(P[s], ids) for s in s_r], axis=0)
    boot = []
    for b in range(B):
        muestra = list(rng.choice(ids, len(ids)))
        boot.append(np.mean([ap(P[s], muestra) for s in s_r], axis=0))
        if b % 250 == 0:
            print(f"  bootstrap réplica {b}/{B}", flush=True)
    boot = np.array(boot)
    rep = {}
    for k, (nombre, ref) in enumerate([("AP", PAPER["AP"]), ("AP50", PAPER["AP50"])]):
        D = obs[k] - ref; lo, hi = np.percentile(boot[:, k] - ref, [2.5, 97.5])
        rep[nombre] = {"nuestro": float(obs[k]), "paper": ref, "D": float(D), "ic95_D": [float(lo), float(hi)],
                       "por_semilla": [ap(P[s], ids)[k] for s in s_r]}
    d50 = abs(rep["AP50"]["D"])
    rep["veredicto"] = "lograda" if d50 <= 0.05 else ("parcial" if d50 <= 0.10 else "fallida")
    rep["semillas"] = s_r
    res["replica"] = rep

# ---------------------------------------------------------------- limitación
gem = pd.read_csv(GEM)
gem = gem[gem.split == "test"].set_index("file_name")
corr = pd.read_csv(CORR)
filas = []
for m in LIMITACION + ["maskrcnn_r101"]:
    ss = semillas(m)
    if not ss:
        continue
    t = pd.concat([pd.read_csv(PG / f"{m}_s{s}" / "test_por_imagen.csv").assign(semilla=s) for s in ss])
    t = t.groupby(["image_id", "archivo", "grupo"]).f1.mean().reset_index()
    t["gemelo"] = t.archivo.map(gem.casi_gemelo).astype(bool)
    dif = t[t.gemelo].f1.mean() - t[~t.gemelo].f1.mean()
    grupos = t.grupo.unique(); bd = []
    for _ in range(B):
        gs = rng.choice(grupos, len(grupos))
        tt = pd.concat([t[t.grupo == g] for g in gs])
        if tt.gemelo.all() or (~tt.gemelo).all():
            continue
        bd.append(tt[tt.gemelo].f1.mean() - tt[~tt.gemelo].f1.mean())
    lo, hi = np.percentile(bd, [2.5, 97.5])
    con = [i for i in ids if gem.casi_gemelo.get(gt.imgs[i]["file_name"], False)]
    sin = [i for i in ids if i not in set(con)]
    ap_con = np.mean([ap(preds(m, s), con)[0] for s in ss]); ap_sin = np.mean([ap(preds(m, s), sin)[0] for s in ss])
    agr = corr[corr.modelo == m].test_F1_grupo.mean() if m in set(corr.modelo) else np.nan
    filas.append(dict(modelo=m, semillas=len(ss), n_con_gemelo=int(t.gemelo.sum()), n_sin_gemelo=int((~t.gemelo).sum()),
                      F1_con_gemelo=t[t.gemelo].f1.mean(), F1_sin_gemelo=t[~t.gemelo].f1.mean(), dif=dif,
                      ic95_lo=lo, ic95_hi=hi, confirma=bool(dif > 0 and lo > 0), AP_con_gemelo=ap_con,
                      AP_sin_gemelo=ap_sin, F1_grupo_test_oficial=t.groupby("grupo").f1.mean().mean(),
                      F1_grupo_test_agrupada_E2=agr))
lim = pd.DataFrame(filas)
lim.to_csv(OUT / "limitacion_casi_gemelos.csv", index=False)
top = lim[lim.modelo.isin(LIMITACION)]
res["limitacion"] = {"modelos_que_confirman": int(top.confirma.sum()), "modelos": int(len(top)),
                     "confirmada": bool(top.confirma.sum() >= 2), "proporcion_test_con_gemelo": float(gem.casi_gemelo.mean())}
(OUT / "paper_guia.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf8")
print(json.dumps(res, indent=1, ensure_ascii=False)); print(lim.round(3).to_string())
