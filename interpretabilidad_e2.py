"""Interpretabilidad del E2 (enunciado §4.6): mapas de oclusión de los 3 mejores modelos, con fidelidad y
estabilidad entre semillas. Exploratorio: no decide el modelo final.

Modelos: Mask R-CNN, HTC y Cascade Mask R-CNN (los 3 de mayor F1 por grupo en test), semillas 42, 43 y 44.
Imágenes: una imagen de test por grupo de escena (14), elegida al azar con semilla 0 entre las que tienen tornillos.
Como hay una imagen por grupo, el bootstrap por imagen es también un bootstrap por grupo.

Técnica: sensibilidad a la oclusión (Zeiler y Fergus 2014). Se divide la imagen en una rejilla (9 x 16 celdas de
80 x 80 px en 1280 x 720) y se tapa una celda a la vez con el color medio de la imagen. Lo que se mide en cada
oclusión es cuánto del resultado original sobrevive, con la función de D-RISE (Petsiuk et al. 2021):
    f(imagen) = suma, sobre cada detección original con score >= umbral de val, del score de la detección nueva que
                más se le parece (IoU de máscara >= 0.5); 0 si ninguna se le parece.
El valor de la celda es (f0 - f con la celda tapada) / f0: la fracción del resultado que depende de esa zona.
Funciona igual para modelos de marcos distintos porque solo usa entradas y salidas (no gradientes).

Fidelidad (¿el mapa señala lo que de verdad usa el modelo?): curva de borrado (Petsiuk et al. 2018, RISE). Se tapan
las celdas de más a menos importantes, de 9 en 9, y se mide f / f0 tras cada paso. El área bajo esa curva (AUC de
borrado) es baja si el mapa es fiel. Se compara con el borrado en orden aleatorio (3 órdenes, los mismos para todos
los modelos): ganancia de fidelidad = AUC aleatorio - AUC del mapa (> 0 es mejor que el azar).
Estabilidad entre semillas: correlación de Spearman entre los mapas de la misma imagen con semillas distintas, y
solapamiento (Jaccard) del 10 % de celdas más importantes. La oclusión es determinista, así que lo que varía es el
modelo. Referencias: la correlación entre imágenes distintas (lo que da solo la disposición típica de la escena) y
entre modelos distintos con la misma imagen.
Plausibilidad (extra): qué parte del valor positivo del mapa cae en celdas que tienen tornillos anotados, frente a la
proporción de celdas con tornillos.

Uso (desde Trabajo\\datos\\wdsd, después de la ablación y de robustez_latencia: usa la GPU, ~1-1.5 h en la 3060):
    python interpretabilidad_e2.py --entornos --prueba    # 2 imágenes, semilla 42, rejilla 3x4: que todo cargue
    python interpretabilidad_e2.py --entornos             # los 3 modelos x 3 semillas
    python interpretabilidad_e2.py --analizar             # tablas y figuras (CPU, segundos)
Se puede interrumpir y volver a lanzar: las corridas terminadas se saltan. Salida: results/e2/interpretabilidad/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))
import robustez_latencia_e2 as RL  # noqa: E402  (cargadores de modelos ya probados)

TOP3 = ["maskrcnn", "htc", "cascade"]
SEMILLAS = [42, 43, 44]
FILAS, COLS = 9, 16
PASOS = 16          # curva de borrado: 0, 9, 18, ... 144 celdas tapadas
N_ALEATORIOS = 3
IOU_MIN = 0.5
SUB = 4             # las máscaras se comparan submuestreadas 4x (320 x 180): basta para un IoU >= 0.5
SALIDA = RAIZ / "results" / "e2" / "interpretabilidad"
B = 2000


# --------------------------------------------------------------------------- imágenes
def elegir_imagenes(gt) -> list[int]:
    rng = np.random.default_rng(0)
    por_grupo: dict = {}
    for i in sorted(gt.imgs):
        if gt.imgToAnns[i]:
            por_grupo.setdefault(gt.imgs[i]["grupo"], []).append(i)
    return [int(rng.choice(v)) for g, v in sorted(por_grupo.items())]


def celdas(h: int, w: int, filas: int, cols: int):
    ys, xs = np.linspace(0, h, filas + 1).astype(int), np.linspace(0, w, cols + 1).astype(int)
    return [(ys[r], ys[r + 1], xs[c], xs[c + 1]) for r in range(filas) for c in range(cols)]


def tapar(rgb: np.ndarray, cajas, idx, relleno) -> np.ndarray:
    x = rgb.copy()
    for k in idx:
        y0, y1, x0, x1 = cajas[k]
        x[y0:y1, x0:x1] = relleno
    return x


# --------------------------------------------------------------------------- función objetivo (D-RISE)
def plano(ms: np.ndarray) -> np.ndarray:
    return ms[:, ::SUB, ::SUB].reshape(len(ms), -1).astype(np.float32)


def objetivo(ref: np.ndarray, sc: np.ndarray, ms: np.ndarray) -> float:
    """ref: máscaras originales aplanadas (K x P). Suma del mejor score que se parece a cada detección original."""
    if len(ms) == 0 or len(ref) == 0:
        return 0.0
    m = plano(ms)
    inter = ref @ m.T
    union = ref.sum(1)[:, None] + m.sum(1)[None, :] - inter
    iou = inter / np.maximum(union, 1)
    s = np.where(iou >= IOU_MIN, sc[None, :], 0.0)
    return float(s.max(1).sum())


# --------------------------------------------------------------------------- una corrida
def mapas(modelo: str, semilla: int, disp, prueba: bool):
    from wdsd_e2.comun import ruta_json
    from wdsd_inst.metricas import cargar_gt
    carpeta = RAIZ / "results" / "e2" / "corridas" / f"{modelo}_s{semilla}"
    umbral = json.loads((carpeta / "metricas.json").read_text())["umbral_score"]
    gt = cargar_gt(ruta_json("test"))
    ids = elegir_imagenes(gt)[: 2 if prueba else None]
    filas, cols, pasos = (3, 4, 4) if prueba else (FILAS, COLS, PASOS)
    n = filas * cols
    pred = RL.cargar(modelo, carpeta, disp)
    out = {k: [] for k in ("image_id", "S", "f0", "n_det", "curva_mapa", "curva_azar", "cobertura_gt")}
    t0 = time.time()
    for i in ids:
        rgb = RL.leer_rgb(gt.imgs[i]["ruta"])
        h, w = rgb.shape[:2]
        cajas = celdas(h, w, filas, cols)
        relleno = rgb.reshape(-1, 3).mean(0).round().astype(np.uint8)
        sc, ms = pred.masks(rgb)
        keep = sc >= umbral
        ref = plano(ms[keep])
        f0 = objetivo(ref, sc, ms)
        # cobertura de tornillos anotados por celda (para la plausibilidad)
        anot = np.zeros((h, w), bool)
        for a in gt.imgToAnns[i]:
            anot |= gt.annToMask(a).astype(bool)
        cob = np.array([anot[y0:y1, x0:x1].mean() for y0, y1, x0, x1 in cajas]).reshape(filas, cols)
        if f0 <= 0:
            print(f"  {modelo} s{semilla} imagen {i}: sin detecciones sobre el umbral, se omite", flush=True)
            continue
        f = lambda idx: objetivo(ref, *pred.masks(tapar(rgb, cajas, idx, relleno))) / f0  # noqa: E731
        S = np.array([1.0 - f([k]) for k in range(n)])
        ks = np.linspace(0, n, pasos + 1).round().astype(int)
        f_todo = f(range(n))

        def curva(orden):
            return [1.0] + [f(orden[:k]) for k in ks[1:-1]] + [f_todo]

        rng = np.random.default_rng([i, 7])  # mismos órdenes aleatorios para todos los modelos y semillas
        out["image_id"].append(i); out["S"].append(S.reshape(filas, cols)); out["f0"].append(f0)
        out["n_det"].append(int(keep.sum())); out["cobertura_gt"].append(cob)
        out["curva_mapa"].append(curva(np.argsort(-S, kind="stable")))
        out["curva_azar"].append([curva(rng.permutation(n)) for _ in range(N_ALEATORIOS)])
        print(f"  {modelo} s{semilla} imagen {i}: {int(keep.sum())} detecciones, f0 {f0:.2f}, "
              f"máx celda {S.max():.2f}, {time.time() - t0:.0f} s", flush=True)
    return {k: np.array(v) for k, v in out.items()}


def lanzar_entornos(a):
    py = RL.maquina()["python"]
    grupos: dict = {}
    for m in a.modelos:
        grupos.setdefault(RL.ENTORNO[RL.MODELOS[m]], []).append(m)
    for env, ms in grupos.items():
        cmd = [py[env], str(Path(__file__).resolve()), "--modelos", *ms, "--semillas", *map(str, a.semillas)]
        cmd += ["--prueba"] * a.prueba
        print(f"\n=== entorno {env}: {ms}", flush=True)
        r = subprocess.run(cmd, cwd=RAIZ, env={**os.environ, "PYTHONUTF8": "1"})
        if r.returncode:
            print(f"!! el entorno {env} terminó con código {r.returncode}; sigo con el siguiente", flush=True)


# --------------------------------------------------------------------------- análisis
def ic(x, rng):
    x = np.asarray(x, float)
    b = [rng.choice(x, len(x)).mean() for _ in range(B)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def spearman(a, b):
    ra, rb = pd.Series(np.ravel(a)).rank().values, pd.Series(np.ravel(b)).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def jaccard_top(a, b, q=0.10):
    k = max(1, int(round(q * np.size(a))))
    ta, tb = set(np.argsort(-np.ravel(a))[:k]), set(np.argsort(-np.ravel(b))[:k])
    return len(ta & tb) / len(ta | tb)


def _trap(y, x):
    return (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(y, x, axis=-1)


def auc(c):
    c = np.asarray(c, float)
    r = _trap(c, np.linspace(0, 1, c.shape[-1]))
    return float(r) if c.ndim == 1 else r


def analizar():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    D = {}
    for p in sorted((SALIDA / "mapas").glob("*_s*.npz")):
        m, s = p.stem.rsplit("_s", 1)
        z = np.load(p)
        D[(m, int(s))] = {k: z[k] for k in z.files}
    if not D:
        sys.exit("No hay mapas en results/e2/interpretabilidad/mapas/: corre primero sin --analizar.")
    modelos = [m for m in TOP3 if any(k[0] == m for k in D)]
    fid, est, pla, curvas = [], [], [], {}
    for m in modelos:
        ss = sorted(s for mm, s in D if mm == m)
        # fidelidad: por imagen, media de semillas
        filas = []
        for s in ss:
            d = D[(m, s)]
            for i, cm, ca, S, cob in zip(d["image_id"], d["curva_mapa"], d["curva_azar"], d["S"], d["cobertura_gt"]):
                pos = np.clip(S, 0, None)
                filas.append(dict(semilla=s, image_id=int(i), auc_mapa=auc(cm), auc_azar=float(auc(ca).mean()),
                                  masa_en_tornillos=float(pos[cob > 0].sum() / pos.sum()) if pos.sum() > 0 else np.nan,
                                  celdas_con_tornillo=float((cob > 0).mean())))
            curvas.setdefault(m, []).append((d["curva_mapa"].mean(0), d["curva_azar"].mean((0, 1))))
        t = pd.DataFrame(filas)
        t["ganancia"] = t.auc_azar - t.auc_mapa
        pi = t.groupby("image_id")[["auc_mapa", "auc_azar", "ganancia", "masa_en_tornillos", "celdas_con_tornillo"]].mean()
        g = ic(pi.ganancia, rng)
        fid.append(dict(modelo=m, semillas=len(ss), imagenes=len(pi), AUC_borrado_mapa=pi.auc_mapa.mean(),
                        AUC_borrado_azar=pi.auc_azar.mean(), ganancia=g[0], ic95_lo=g[1], ic95_hi=g[2],
                        imagenes_con_ganancia=float((pi.ganancia > 0).mean())))
        mt = pi.masa_en_tornillos.dropna()
        pla.append(dict(modelo=m, masa_positiva_en_celdas_con_tornillo=mt.mean(),
                        proporcion_celdas_con_tornillo=pi.celdas_con_tornillo.mean(),
                        razon=mt.mean() / pi.celdas_con_tornillo.mean()))
        # estabilidad entre semillas (misma imagen) y referencia entre imágenes distintas
        rs, js, nulos = [], [], []
        for a_ in range(len(ss)):
            for b_ in range(a_ + 1, len(ss)):
                da, db = D[(m, ss[a_])], D[(m, ss[b_])]
                ia = {int(i): k for k, i in enumerate(da["image_id"])}
                ib = {int(i): k for k, i in enumerate(db["image_id"])}
                comunes = sorted(set(ia) & set(ib))
                for i in comunes:
                    rs.append((i, spearman(da["S"][ia[i]], db["S"][ib[i]])))
                    js.append((i, jaccard_top(da["S"][ia[i]], db["S"][ib[i]])))
                    nulos += [spearman(da["S"][ia[i]], db["S"][ib[j]]) for j in comunes if j != i]
        if rs:
            r_img = pd.DataFrame(rs, columns=["i", "r"]).groupby("i").r.mean()
            j_img = pd.DataFrame(js, columns=["i", "j"]).groupby("i").j.mean()
            r = ic(r_img, rng)
            est.append(dict(modelo=m, spearman_entre_semillas=r[0], ic95_lo=r[1], ic95_hi=r[2],
                            jaccard_top10=j_img.mean(), spearman_imagenes_distintas=float(np.mean(nulos))))
    # acuerdo entre modelos (misma imagen, misma semilla)
    entre = []
    for a_ in range(len(modelos)):
        for b_ in range(a_ + 1, len(modelos)):
            for s in SEMILLAS:
                if (modelos[a_], s) in D and (modelos[b_], s) in D:
                    da, db = D[(modelos[a_], s)], D[(modelos[b_], s)]
                    ib = {int(i): k for k, i in enumerate(db["image_id"])}
                    entre += [dict(par=f"{modelos[a_]}-{modelos[b_]}", r=spearman(Sa, db["S"][ib[int(i)]]))
                              for i, Sa in zip(da["image_id"], da["S"]) if int(i) in ib]
    fid, est, pla = pd.DataFrame(fid), pd.DataFrame(est), pd.DataFrame(pla)
    entre = pd.DataFrame(entre).groupby("par").r.mean().reset_index(name="spearman_entre_modelos") if entre else pd.DataFrame()
    for nombre, t in [("fidelidad", fid), ("estabilidad", est), ("plausibilidad", pla), ("acuerdo_entre_modelos", entre)]:
        t.to_csv(SALIDA / f"{nombre}.csv", index=False)
        print(f"\n{nombre}\n{t.round(3).to_string(index=False)}")

    # figura: curvas de borrado
    fig, ax = plt.subplots(1, len(modelos), figsize=(4 * len(modelos), 3.2), sharey=True, squeeze=False)
    for k, m in enumerate(modelos):
        cm = np.mean([c[0] for c in curvas[m]], 0); ca = np.mean([c[1] for c in curvas[m]], 0)
        x = np.linspace(0, 1, len(cm))
        ax[0, k].plot(x, cm, label="orden del mapa"); ax[0, k].plot(x, ca, "--", label="orden aleatorio")
        ax[0, k].set_title(m); ax[0, k].set_xlabel("fracción de la imagen tapada")
    ax[0, 0].set_ylabel("f / f0 (resultado que sobrevive)"); ax[0, 0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(SALIDA / "curvas_borrado.png", dpi=150); plt.close(fig)

    # figura: ejemplos (3 imágenes x [imagen + 3 modelos], semilla 42) y una imagen con las 3 semillas de Mask R-CNN
    from wdsd_e2.comun import ruta_json
    from wdsd_inst.metricas import cargar_gt
    gt = cargar_gt(ruta_json("test"))
    s0 = SEMILLAS[0]
    base = D.get((modelos[0], s0))
    if base is not None:
        ejemplos = [int(i) for i in base["image_id"][: 3]]
        fig, ax = plt.subplots(len(ejemplos), 1 + len(modelos), figsize=(3.6 * (1 + len(modelos)), 2.2 * len(ejemplos)),
                               squeeze=False)
        for r, i in enumerate(ejemplos):
            rgb = RL.leer_rgb(gt.imgs[i]["ruta"])
            ax[r, 0].imshow(rgb)
            for a in gt.imgToAnns[i]:
                ax[r, 0].contour(gt.annToMask(a), levels=[0.5], colors="lime", linewidths=0.6)
            ax[r, 0].set_title(f"imagen {i}, grupo {gt.imgs[i]['grupo']}", fontsize=8)
            for c, m in enumerate(modelos, 1):
                d = D.get((m, s0))
                if d is None or i not in set(int(x) for x in d["image_id"]):
                    continue
                S = d["S"][list(int(x) for x in d["image_id"]).index(i)]
                ax[r, c].imshow(rgb, alpha=0.55)
                ax[r, c].imshow(np.ma.masked_less(S, 0.02), cmap="inferno", alpha=0.75, vmin=0, vmax=max(S.max(), 0.05), extent=(0, rgb.shape[1], rgb.shape[0], 0),
                                interpolation="nearest")
                ax[r, c].set_title(f"{m} s{s0}", fontsize=8)
        for a in ax.ravel():
            a.axis("off")
        fig.tight_layout(); fig.savefig(SALIDA / "ejemplos_mapas.png", dpi=150); plt.close(fig)
    print("\nListo. Resultados en", SALIDA)


# --------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=TOP3, choices=list(RL.MODELOS))
    ap.add_argument("--semillas", nargs="+", type=int, default=SEMILLAS)
    ap.add_argument("--entornos", action="store_true", help="relanzar en el entorno de cada modelo (maquina_e2.json)")
    ap.add_argument("--prueba", action="store_true", help="2 imágenes, semilla 42, rejilla 3x4; no guarda resultados")
    ap.add_argument("--analizar", action="store_true")
    a = ap.parse_args()
    if a.analizar:
        return analizar()
    if a.prueba:
        a.semillas = a.semillas[:1]
    if a.entornos:
        return lanzar_entornos(a)
    from wdsd_e2.comun import preparar_gpu
    disp = preparar_gpu(permitir_cpu=False)
    (SALIDA / "mapas").mkdir(parents=True, exist_ok=True)
    for m in a.modelos:
        for s in a.semillas:
            ruta = SALIDA / "mapas" / f"{m}_s{s}.npz"
            if ruta.exists() and not a.prueba:
                print(f"  [ya hecho] {ruta.name}")
                continue
            t0 = time.time()
            r = mapas(m, s, disp, a.prueba)
            print(f"-- {m} s{s}: {len(r['image_id'])} imágenes en {(time.time() - t0) / 60:.1f} min", flush=True)
            if not a.prueba:
                np.savez_compressed(ruta, **r)
    print("Listo." + (" (prueba: no se guardó nada)" if a.prueba else f" Mapas en {SALIDA / 'mapas'}"))


if __name__ == "__main__":
    main()
