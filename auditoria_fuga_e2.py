"""Auditoría de fuga del E2 (enunciado §10.1, tabla de verificaciones obligatorias). Solo CPU, ~2 min.

Uso (desde Trabajo\\datos\\wdsd, entorno wdsd_dl):
    python auditoria_fuga_e2.py
Lee datos_inst/coco (partición agrupada del E2), datos_oficial/coco si existe (partición oficial, ver
paper_guia_e2.py) y results/e2/corridas/*/test_por_imagen.csv. Escribe results/e2/auditoria_fuga/.

Verificaciones, con los criterios del enunciado y los umbrales fijados antes de mirar los resultados:
1. Intersección de identificadores: los grupos de escena (unidad de independencia) de train, val y test no se cruzan.
   Aprueba si la intersección es vacía en las tres parejas.
2. Duplicados exactos: SHA-256 de los bytes de cada imagen. Aprueba si no hay colisiones entre particiones.
3. Casi duplicados: hash perceptual (pHash de 64 bits: DCT de la imagen en gris a 32x32, signo de las 8x8 frecuencias
   bajas frente a su mediana). Umbral a priori: distancia de Hamming <= 10 de 64 bits. Para cada imagen de val y test,
   distancia a la imagen de train más parecida. Se reporta la proporción con casi gemelo en train y el F1 de test de
   cada modelo con y sin esas imágenes. Tolerancia declarada: 5 % de test.
4. Validación adversarial: regresión logística (miniatura en gris de 32x18 + color medio) que intenta separar train de
   test, con validación cruzada de 5 pliegues por grupo de escena. Criterio: AUC no distinguible de 0.5, o diferencia
   explicada.
Las mismas 3 y 4 se calculan para la partición oficial cuando datos_oficial/ existe, como contraste.
La fuga por preentrenamiento (COCO, ImageNet) y la prueba de etiquetas permutadas no se calculan aquí.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parent
OUT = RAIZ / "results" / "e2" / "auditoria_fuga"
UMBRAL_HAMMING = 10
TOLERANCIA = 0.05
rng = np.random.default_rng(0)


def dct_matriz(n=32):
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


DCT = dct_matriz()


def leer(ruta: str):
    b = Path(ruta).read_bytes()
    im = Image.open(ruta).convert("RGB")
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    baja = (DCT @ g @ DCT.T)[:8, :8].ravel()
    bits = baja > np.median(baja[1:])
    mini = np.asarray(im.convert("L").resize((32, 18), Image.BILINEAR), dtype=np.float64).ravel() / 255
    color = np.asarray(im, dtype=np.float64).reshape(-1, 3).mean(0) / 255
    return hashlib.sha256(b).hexdigest(), bits, np.r_[mini, color]


def cargar(coco_dir: Path) -> pd.DataFrame:
    filas = []
    for s in ("train", "val", "test"):
        for im in json.loads((coco_dir / f"{s}.json").read_text(encoding="utf8"))["images"]:
            filas.append((s, im["file_name"], im["ruta"], im["grupo"]))
    return pd.DataFrame(filas, columns=["split", "file_name", "ruta", "grupo"])


def auc(y, s):
    r = pd.Series(s).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logistica(X, y, lam=1.0, iters=500, lr=0.5):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / len(y) + lam * w / len(y)); b -= lr * g.mean()
    return w, b


def adversarial(df, feats):
    d = df[df.split.isin(["train", "test"])].reset_index(drop=True)
    X = np.stack([feats[f] for f in d.file_name]); y = (d.split == "test").values.astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    grupos = d.grupo.unique(); rng.shuffle(grupos)
    pliegue = {g: k % 5 for k, g in enumerate(grupos)}
    f = d.grupo.map(pliegue).values
    s = np.zeros(len(d))
    for k in range(5):
        tr, te = f != k, f == k
        if y[tr].min() == y[tr].max():
            s[te] = 0.5; continue
        w, b = logistica(X[tr], y[tr])
        s[te] = X[te] @ w + b
    a = auc(y, s)
    nulos = [auc(rng.permutation(y), s) for _ in range(1000)]  # AUC si las etiquetas no tuvieran relación
    return {"AUC": float(a), "AUC_nulo_p95": float(np.percentile(nulos, 95)),
            "p_permutacion": float(np.mean(np.array(nulos) >= a)), "n_train": int((y == 0).sum()), "n_test": int(y.sum())}


def casi_gemelos(df, bits):
    tr = df[df.split == "train"]
    B = np.stack([bits[f] for f in tr.file_name])
    filas = []
    for r in df[df.split != "train"].itertuples():
        dist = (B != bits[r.file_name]).sum(1)
        j = int(dist.argmin())
        filas.append((r.split, r.file_name, r.grupo, int(dist[j]), tr.file_name.iloc[j], int(tr.grupo.iloc[j])))
    c = pd.DataFrame(filas, columns=["split", "file_name", "grupo", "hamming_min", "gemelo_train", "grupo_gemelo"])
    c["casi_gemelo"] = c.hamming_min <= UMBRAL_HAMMING
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    agr = cargar(RAIZ / "datos_inst" / "coco")
    of_dir = RAIZ / "datos_oficial" / "coco"
    ofi = cargar(of_dir) if (of_dir / "test.json").exists() else None
    hashes, bits, feats = {}, {}, {}
    for k, r in enumerate(agr.itertuples()):
        hashes[r.file_name], bits[r.file_name], feats[r.file_name] = leer(r.ruta)
        if k % 200 == 0:
            print(f"  {k}/{len(agr)} imágenes", flush=True)
    res = {"umbral_hamming": UMBRAL_HAMMING, "tolerancia_casi_gemelos": TOLERANCIA}

    # 1. intersección de grupos
    g = {s: set(agr[agr.split == s].grupo) for s in ("train", "val", "test")}
    inter = {f"{a}-{b}": sorted(g[a] & g[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    res["1_interseccion_grupos"] = {"intersecciones": inter, "aprueba": all(not v for v in inter.values())}

    # 2. duplicados exactos
    agr["sha256"] = agr.file_name.map(hashes)
    cruz = agr.groupby("sha256").split.nunique()
    col = agr[agr.sha256.isin(cruz[cruz > 1].index)].sort_values("sha256")
    col.drop(columns="ruta").to_csv(OUT / "duplicados_exactos_entre_particiones.csv", index=False)
    dentro = int((agr.groupby("sha256").size() > 1).sum())
    res["2_duplicados_exactos"] = {"colisiones_entre_particiones": int(cruz.gt(1).sum()),
                                   "hashes_repetidos_en_total": dentro, "aprueba": bool(cruz.gt(1).sum() == 0)}
    agr[["split", "file_name", "grupo", "sha256"]].to_csv(OUT / "hashes_sha256.csv", index=False)

    # 3. casi duplicados
    c = casi_gemelos(agr, bits)
    c.to_csv(OUT / "casi_gemelos_particion_agrupada.csv", index=False)
    prop = c.groupby("split").casi_gemelo.mean()
    res["3_casi_duplicados"] = {"proporcion_con_casi_gemelo": prop.round(4).to_dict(),
                                "n_test_con_casi_gemelo": int(c[c.split == "test"].casi_gemelo.sum()),
                                "hamming_min_mediana_test": float(c[c.split == "test"].hamming_min.median()),
                                "aprueba": bool(prop.get("test", 0) <= TOLERANCIA)}
    # desempeño con y sin las imágenes de test con casi gemelo
    gem = set(c[(c.split == "test") & c.casi_gemelo].file_name)
    filas = []
    for p in sorted((RAIZ / "results" / "e2" / "corridas").glob("*_s*/test_por_imagen.csv")):
        m, s = p.parent.name.rsplit("_s", 1)
        t = pd.read_csv(p)
        todo = t.groupby("grupo").f1.mean().mean()
        sin = t[~t.archivo.isin(gem)].groupby("grupo").f1.mean().mean() if gem else todo
        filas.append((m, int(s), todo, sin, len(t) - int(t.archivo.isin(gem).sum())))
    if filas:
        dd = pd.DataFrame(filas, columns=["modelo", "semilla", "F1_grupo_todo", "F1_grupo_sin_casi_gemelos", "n_img_sin"])
        dd.groupby("modelo")[["F1_grupo_todo", "F1_grupo_sin_casi_gemelos", "n_img_sin"]].mean() \
          .sort_values("F1_grupo_todo", ascending=False).to_csv(OUT / "desempeno_con_y_sin_casi_gemelos.csv")

    # 4. validación adversarial
    res["4_validacion_adversarial"] = adversarial(agr, feats)

    # contraste: partición oficial
    if ofi is not None:
        co = casi_gemelos(ofi, bits)
        co.to_csv(OUT / "casi_gemelos_particion_oficial.csv", index=False)
        go = {s: set(ofi[ofi.split == s].grupo) for s in ("train", "test")}
        res["oficial"] = {"proporcion_con_casi_gemelo": co.groupby("split").casi_gemelo.mean().round(4).to_dict(),
                          "grupos_test_tambien_en_train": len(go["test"] & go["train"]), "grupos_test": len(go["test"]),
                          "validacion_adversarial": adversarial(ofi, feats)}
    (OUT / "resumen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
