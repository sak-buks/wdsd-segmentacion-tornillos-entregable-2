"""Robustez a perturbaciones, latencia y FLOPs de los modelos del E2 que corren en el entorno wdsd_dl.

Cubre Mask R-CNN (torchvision) y YOLOv8n-seg, YOLO11n-seg y YOLO26n-seg (ultralytics). No cubre los modelos de
MMDetection ni YOLOv7/YOLOv9: cada uno necesita su propio entorno y su propio cargador.

Uso (PowerShell, desde la carpeta del proyecto: la que tiene src/, datos_inst/ y results/e2/corridas/):
    conda activate wdsd_dl            # o: C:\\Users\\ivanj\\miniconda3\\Scripts\\conda.exe run -n wdsd_dl ...
    python robustez_latencia_e2.py --prueba          # 5 imágenes, Mask R-CNN, semilla 42: comprueba que todo funciona
    python robustez_latencia_e2.py                   # completo: 4 modelos x 5 semillas
    python robustez_latencia_e2.py --modelos maskrcnn --semillas 42    # solo uno

Qué mide
- Robustez: F1 de instancia por imagen (mismo cálculo que el E2, src/wdsd_inst/metricas.py) sobre las 108 imágenes de
  test, con la imagen perturbada. Se usa el umbral de score elegido en val con la imagen limpia (metricas.json de cada
  corrida): no se reajusta bajo perturbación. Perturbaciones: ruido gaussiano aditivo (sigma 0.02, 0.05, 0.10 sobre
  la escala 0-1) y desenfoque gaussiano (sigma 1, 2, 4 px). La semilla del ruido depende solo de la imagen y del
  nivel, así que todos los modelos ven el mismo ruido. El caso "limpio" se recalcula aquí y se compara con
  metricas.json como control de calidad.
- Latencia: una sola GPU, imagen por imagen (lote 1), 10 imágenes de calentamiento y las 108 de test medidas. El
  tiempo va desde la imagen ya en memoria (arreglo uint8) hasta las máscaras booleanas en CPU; no incluye leer el
  archivo ni codificar RLE. Se usa la misma precisión con que se calcularon las predicciones del E2: Mask R-CNN con
  autocast (FP16 mixto), ultralytics en FP32. Se informa la mediana y el percentil 95.
- FLOPs: torch.utils.flop_counter.FlopCounterMode sobre una imagen de test (cuenta convoluciones y multiplicaciones
  de matrices; no cuenta operaciones como roi_align ni NMS). Valor aproximado y propio de cada imagen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from wdsd_e2.comun import ruta_json, filtrar, preparar_gpu          # noqa: E402
from wdsd_inst.metricas import a_rle, cargar_gt, pq_por_imagen      # noqa: E402

MODELOS = {"maskrcnn": "torchvision", "yolov8": "ultra", "yolo11": "ultra", "yolo26": "ultra"}
RUIDO = (0.02, 0.05, 0.10)
DESENFOQUE = (1.0, 2.0, 4.0)
PERTURBACIONES = [("limpio", 0.0)] + [("ruido", s) for s in RUIDO] + [("desenfoque", s) for s in DESENFOQUE]
SALIDA = RAIZ / "results" / "e2" / "robustez_latencia"


# --------------------------------------------------------------------------- imágenes y perturbaciones
def leer_rgb(ruta: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(ruta).convert("RGB"))  # uint8 H x W x 3


def perturbar(rgb: np.ndarray, tipo: str, nivel: float, image_id: int) -> np.ndarray:
    import cv2
    if tipo == "limpio":
        return rgb
    if tipo == "ruido":
        rng = np.random.default_rng([int(image_id), int(round(nivel * 1000))])
        x = rgb.astype(np.float32) / 255.0 + rng.normal(0.0, nivel, rgb.shape).astype(np.float32)
        return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
    if tipo == "desenfoque":
        return cv2.GaussianBlur(rgb, (0, 0), sigmaX=nivel)
    raise ValueError(tipo)


# --------------------------------------------------------------------------- cargadores de modelos
class PredictorTV:
    """Mask R-CNN de torchvision, igual que src/wdsd_inst/maskrcnn.py:predecir."""

    def __init__(self, carpeta: Path, disp):
        import torch
        from wdsd_inst import maskrcnn
        self.torch, self.disp = torch, disp
        self.m = maskrcnn.crear_modelo(True).to(disp)
        self.m.load_state_dict(torch.load(carpeta / "mejor.pt", map_location=disp))
        self.m.eval()
        self.parametros = sum(p.numel() for p in self.m.parameters())

    def masks(self, rgb: np.ndarray):
        torch, disp = self.torch, self.disp
        with torch.no_grad():
            img = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(disp)
            with torch.autocast(device_type=disp.type, enabled=disp.type == "cuda"):
                out = self.m([img])[0]
            sc = out["scores"].float().cpu().numpy()
            ms = (out["masks"][:, 0] > 0.5).cpu().numpy()
        return sc, ms


class PredictorUltra:
    """ultralytics, con los mismos argumentos que src/wdsd_e2/m_ultra.py:predecir. Recibe BGR uint8 (como cv2)."""

    def __init__(self, carpeta: Path, disp):
        from ultralytics import YOLO
        self.disp = disp
        self.y = YOLO(str(carpeta / "mejor.pt"))
        self.parametros = sum(p.numel() for p in self.y.model.parameters())

    def masks(self, rgb: np.ndarray):
        r = self.y.predict(np.ascontiguousarray(rgb[:, :, ::-1]), imgsz=1280, conf=0.05, iou=0.6, max_det=100,
                           retina_masks=True, device=0 if self.disp.type == "cuda" else "cpu", verbose=False)[0]
        if r.masks is None:
            return np.zeros(0, np.float32), np.zeros((0, *rgb.shape[:2]), bool)
        return r.boxes.conf.cpu().numpy(), r.masks.data.cpu().numpy() > 0.5


def cargar(modelo: str, carpeta: Path, disp):
    return (PredictorTV if MODELOS[modelo] == "torchvision" else PredictorUltra)(carpeta, disp)


def a_preds(image_id: int, sc, ms) -> list[dict]:
    return [{"image_id": int(image_id), "category_id": 1, "segmentation": a_rle(m), "score": float(s)}
            for s, m in zip(sc, ms) if s >= 0.05 and m.any()]


# --------------------------------------------------------------------------- robustez
def robustez(modelo: str, semilla: int, disp, n_max: int | None) -> pd.DataFrame | None:
    carpeta = RAIZ / "results" / "e2" / "corridas" / f"{modelo}_s{semilla}"
    if not (carpeta / "mejor.pt").exists():
        print(f"  [salto] no existe {carpeta / 'mejor.pt'}")
        return None
    umbral = json.loads((carpeta / "metricas.json").read_text())["umbral_score"]
    gt = cargar_gt(ruta_json("test"))
    imgs = sorted(gt.imgs.values(), key=lambda i: i["id"])[:n_max]
    pred = cargar(modelo, carpeta, disp)
    filas = []
    for tipo, nivel in PERTURBACIONES:
        preds = []
        for im in imgs:
            rgb = perturbar(leer_rgb(im["ruta"]), tipo, nivel, im["id"])
            sc, ms = pred.masks(rgb)
            preds += a_preds(im["id"], sc, ms)
        t = pq_por_imagen(gt, filtrar(preds), umbral)
        t = t[t.image_id.isin([i["id"] for i in imgs])]
        t.insert(0, "nivel", nivel); t.insert(0, "perturbacion", tipo)
        t.insert(0, "semilla", semilla); t.insert(0, "modelo", modelo)
        filas.append(t)
        print(f"  {modelo} s{semilla} {tipo:10s} {nivel:<5}: F1 imagen {t.f1.mean():.3f}, "
              f"F1 grupo {t.groupby('grupo').f1.mean().mean():.3f}")
    df = pd.concat(filas, ignore_index=True)
    if n_max is None:  # control de calidad contra el cálculo del E2
        ref = json.loads((carpeta / "metricas.json").read_text())["test"]["F1_grupo"]
        ahora = df[df.perturbacion == "limpio"].groupby("grupo").f1.mean().mean()
        marca = "OK" if abs(ref - ahora) < 0.01 else "REVISAR"
        print(f"  control limpio: metricas.json {ref:.4f} vs aquí {ahora:.4f} -> {marca}")
    return df


# --------------------------------------------------------------------------- latencia y FLOPs
def latencia(modelo: str, semilla: int, disp, n_max: int | None) -> dict | None:
    import torch
    carpeta = RAIZ / "results" / "e2" / "corridas" / f"{modelo}_s{semilla}"
    if not (carpeta / "mejor.pt").exists():
        return None
    gt = cargar_gt(ruta_json("test"))
    imgs = sorted(gt.imgs.values(), key=lambda i: i["id"])[:n_max]
    arrs = [leer_rgb(i["ruta"]) for i in imgs]
    pred = cargar(modelo, carpeta, disp)
    sync = torch.cuda.synchronize if disp.type == "cuda" else (lambda: None)
    for a in arrs[:10]:
        pred.masks(a)
    sync(); torch.cuda.reset_peak_memory_stats() if disp.type == "cuda" else None
    ms_ = []
    for a in arrs:
        sync(); t0 = time.perf_counter()
        pred.masks(a)
        sync(); ms_.append((time.perf_counter() - t0) * 1000)
    gflops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            pred.masks(arrs[0])
        gflops = fc.get_total_flops() / 1e9
    except Exception as e:  # el conteo es un extra: no debe tumbar la medición de latencia
        print(f"  [aviso] FLOPs no disponibles para {modelo}: {e}")
    pico = torch.cuda.max_memory_allocated() / 2**30 if disp.type == "cuda" else None
    r = {"modelo": modelo, "semilla": semilla, "n_imagenes": len(arrs), "mediana_ms": float(np.median(ms_)),
         "p95_ms": float(np.percentile(ms_, 95)), "media_ms": float(np.mean(ms_)),
         "imagenes_por_s": 1000 / float(np.mean(ms_)), "gflops_una_imagen": gflops,
         "parametros": pred.parametros, "gpu_pico_gb": pico,
         "gpu": torch.cuda.get_device_name(0) if disp.type == "cuda" else "cpu"}
    print(f"  latencia {modelo} s{semilla}: mediana {r['mediana_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms, "
          f"GFLOPs {gflops if gflops is None else round(gflops, 1)}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=list(MODELOS), choices=list(MODELOS))
    ap.add_argument("--semillas", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--prueba", action="store_true", help="5 imágenes, maskrcnn, semilla 42; no escribe resultados finales")
    ap.add_argument("--sin-latencia", action="store_true")
    a = ap.parse_args()
    if a.prueba:
        a.modelos, a.semillas = ["maskrcnn"], [42]
    n_max = 5 if a.prueba else None
    disp = preparar_gpu(permitir_cpu=False)
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "robustez").mkdir(exist_ok=True)
    lat = []
    for m in a.modelos:
        for s in a.semillas:
            ruta = SALIDA / "robustez" / f"{m}_s{s}.csv"
            if ruta.exists() and not a.prueba:
                print(f"  [ya hecho] {ruta.name}")
            else:
                df = robustez(m, s, disp, n_max)
                if df is not None and not a.prueba:
                    df.to_csv(ruta, index=False)
            if s == a.semillas[0] and not a.sin_latencia:  # latencia: una semilla por modelo basta (mismo modelo)
                r = latencia(m, s, disp, n_max)
                if r:
                    lat.append(r)
    if lat and not a.prueba:
        pd.DataFrame(lat).to_csv(SALIDA / "latencia.csv", index=False)
    partes = sorted((SALIDA / "robustez").glob("*.csv"))
    if partes and not a.prueba:
        todo = pd.concat([pd.read_csv(p) for p in partes], ignore_index=True)
        g = todo.groupby(["modelo", "semilla", "perturbacion", "nivel", "grupo"]).f1.mean().reset_index()
        res = (g.groupby(["modelo", "perturbacion", "nivel", "grupo"]).f1.mean().reset_index()
                .groupby(["modelo", "perturbacion", "nivel"]).f1.mean().reset_index().rename(columns={"f1": "F1_grupo"}))
        res.to_csv(SALIDA / "robustez_resumen.csv", index=False)
        print(res.pivot_table(index="modelo", columns=["perturbacion", "nivel"], values="F1_grupo").round(3).to_string())
    print("Listo. Resultados en", SALIDA)


if __name__ == "__main__":
    main()
