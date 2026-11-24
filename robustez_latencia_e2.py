"""Robustez a perturbaciones, latencia y FLOPs de los 13 modelos del E2.

Cada modelo corre en su entorno: Mask R-CNN y los YOLO de ultralytics en wdsd_dl, los 7 de MMDetection en
wdsd_mmdet, YOLOv7 y YOLOv9 (repos oficiales) en wdsd_yolo9. Con --entornos el script se relanza solo en el entorno
de cada modelo (rutas de maquina_e2.json), así que basta un comando.

Uso (PowerShell, desde la carpeta del proyecto: la que tiene src/, datos_inst/ y results/e2/corridas/):
    python robustez_latencia_e2.py --prueba          # 5 imágenes, Mask R-CNN, semilla 42: comprueba que todo funciona
    python robustez_latencia_e2.py --entornos --faltantes --semillas 42 --prueba   # prueba de los 9 que faltaban
    python robustez_latencia_e2.py --entornos --faltantes --semillas 42            # los 9 que faltaban, semilla 42
    python robustez_latencia_e2.py --modelos maskrcnn --semillas 42    # solo uno, en el entorno actual

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
- YOLOv7 y YOLOv9 se predicen con segment/val.py de su repo, igual que en el E2 (m_oficial.py): las imágenes
  perturbadas se escriben en PNG (sin pérdida) en results/e2/robustez_latencia/imagenes/. Su latencia es la que
  imprime val.py (media de preproceso + inferencia + NMS por imagen, lote 1), no la mediana medida aquí, y sus GFLOPs
  los del resumen del repo (calculados a 640x640) escalados por 1280x720 / 640x640 = 2.25. Se marca en la columna
  metodo_latencia.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from wdsd_e2.comun import ruta_json, filtrar, preparar_gpu          # noqa: E402
from wdsd_inst.metricas import a_rle, cargar_gt, pq_por_imagen      # noqa: E402

MODELOS = {"maskrcnn": "torchvision", "yolov8": "ultra", "yolo11": "ultra", "yolo26": "ultra",
           "cascade": "mmdet", "htc": "mmdet", "pointrend": "mmdet", "msrcnn": "mmdet", "mask2former": "mmdet",
           "condinst": "mmdet", "solov2": "mmdet", "yolov7": "oficial", "yolov9": "oficial"}
ENTORNO = {"torchvision": "dl", "ultra": "dl", "mmdet": "mmdet", "oficial": "yolo9"}
FALTANTES = [m for m, k in MODELOS.items() if k in ("mmdet", "oficial")]
PESOS = {"torchvision": "mejor.pt", "ultra": "mejor.pt", "mmdet": "mejor.pth", "oficial": "mejor.pt"}
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


class PredictorMMDet:
    """MMDetection, con la misma configuración con que se entrenó y predijo (src/wdsd_e2/m_mmdet.py). Recibe BGR."""

    def __init__(self, modelo: str, carpeta: Path, disp):
        from mmdet.apis import init_detector
        from mmdet.utils import register_all_modules
        from wdsd_e2 import m_mmdet
        register_all_modules()
        r = json.loads((carpeta / "resumen.json").read_text(encoding="utf8"))
        cfg, _ = m_mmdet.configuracion(modelo, carpeta, int(r["semilla"]), int(r["epocas"]), float(r["valor"]),
                                       amp=False)
        self.det = init_detector(cfg, str(carpeta / "mejor.pth"), device=str(disp))
        self.parametros = sum(p.numel() for p in self.det.parameters())

    def masks(self, rgb: np.ndarray):
        from mmdet.apis import inference_detector
        r = inference_detector(self.det, np.ascontiguousarray(rgb[:, :, ::-1])).pred_instances
        return r.scores.float().cpu().numpy(), r.masks.cpu().numpy().astype(bool)


def cargar(modelo: str, carpeta: Path, disp):
    marco = MODELOS[modelo]
    if marco == "mmdet":
        return PredictorMMDet(modelo, carpeta, disp)
    return (PredictorTV if marco == "torchvision" else PredictorUltra)(carpeta, disp)


# --------------------------------------------------------------------------- YOLOv7 / YOLOv9 (repos oficiales)
def maquina() -> dict:
    return json.loads((RAIZ / "maquina_e2.json").read_text(encoding="utf8"))


def carpeta_perturbada(tipo: str, nivel: float, imgs: list[dict], prueba: bool) -> Path:
    """Escribe una vez las imágenes de test perturbadas en formato YOLO (images/test + labels/test)."""
    from PIL import Image
    from wdsd_e2.comun import DATOS
    raiz = SALIDA / "imagenes" / (f"{tipo}_{nivel:g}" + ("_prueba" if prueba else ""))
    di, dl = raiz / "images" / "test", raiz / "labels" / "test"
    if (raiz / "listo.txt").exists():
        return raiz
    di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    for im in imgs:
        stem = Path(im["file_name"]).stem
        Image.fromarray(perturbar(leer_rgb(im["ruta"]), tipo, nivel, im["id"])).save(di / f"{stem}.png")
        shutil.copy2(DATOS / "yolo" / "labels" / "test" / f"{stem}.txt", dl / f"{stem}.txt")
    (raiz / "listo.txt").write_text(str(len(imgs)), encoding="utf8")
    return raiz


def predecir_oficial(modelo: str, pesos: Path, raiz: Path, trabajo: Path, disp) -> tuple[list[dict], str]:
    """segment/val.py sobre una carpeta de imágenes, con los mismos argumentos que src/wdsd_e2/m_oficial.py."""
    import yaml
    from wdsd_e2 import registro as R
    from wdsd_e2.comun import imagenes
    from wdsd_e2.m_oficial import _correr
    repo = Path(maquina()["repos"][R.MODELOS[modelo]["repo"]])
    trabajo.mkdir(parents=True, exist_ok=True)
    data = trabajo / "data.yaml"
    data.write_text(yaml.safe_dump({"path": raiz.resolve().as_posix(), "train": "images/test", "val": "images/test",
                                    "nc": 1, "names": {0: "tornillo"}}, sort_keys=False), encoding="utf8")
    salida = trabajo / "pred"
    texto = _correr([sys.executable, "segment/val.py", "--data", str(data), "--weights", str(pesos), "--img", "1280",
                     "--batch-size", "1", "--task", "val", "--conf-thres", "0.05", "--iou-thres", "0.6",
                     "--max-det", "100", "--save-json", "--device", "0" if disp.type == "cuda" else "cpu",
                     "--project", str(trabajo), "--name", "pred", "--exist-ok"], repo, trabajo / "val.log", print)
    archivo = salida / f"{pesos.stem}_predictions.json"
    crudo = json.loads(archivo.read_text()) if archivo.exists() else []
    ids = {Path(im["file_name"]).stem: im for im in imagenes("test")}
    preds = [{"image_id": ids[str(p["image_id"])]["id"], "category_id": 1, "segmentation": p["segmentation"],
              "score": float(p["score"])} for p in crudo]
    shutil.rmtree(salida, ignore_errors=True)
    return preds, texto


def a_preds(image_id: int, sc, ms) -> list[dict]:
    return [{"image_id": int(image_id), "category_id": 1, "segmentation": a_rle(m), "score": float(s)}
            for s, m in zip(sc, ms) if s >= 0.05 and m.any()]


# --------------------------------------------------------------------------- robustez
def robustez(modelo: str, semilla: int, disp, n_max: int | None) -> pd.DataFrame | None:
    carpeta = RAIZ / "results" / "e2" / "corridas" / f"{modelo}_s{semilla}"
    pesos = carpeta / PESOS[MODELOS[modelo]]
    if not pesos.exists():
        print(f"  [salto] no existe {pesos}")
        return None
    umbral = json.loads((carpeta / "metricas.json").read_text())["umbral_score"]
    gt = cargar_gt(ruta_json("test"))
    imgs = sorted(gt.imgs.values(), key=lambda i: i["id"])[:n_max]
    oficial = MODELOS[modelo] == "oficial"
    pred = None if oficial else cargar(modelo, carpeta, disp)
    filas = []
    for tipo, nivel in PERTURBACIONES:
        preds = []
        if oficial:
            raiz = carpeta_perturbada(tipo, nivel, imgs, n_max is not None)
            preds, texto = predecir_oficial(modelo, pesos, raiz, SALIDA / "tmp" / f"{modelo}_s{semilla}", disp)
            if tipo == "limpio":
                ULTIMO_VAL[modelo] = texto
        else:
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
ULTIMO_VAL: dict = {}  # salida de val.py en la imagen limpia (YOLOv7/YOLOv9), para su latencia


def latencia_oficial(modelo: str, semilla: int, disp) -> dict | None:
    carpeta = RAIZ / "results" / "e2" / "corridas" / f"{modelo}_s{semilla}"
    texto = ULTIMO_VAL.get(modelo)
    if texto is None:  # la robustez ya estaba hecha: se corre val.py una vez sobre las imágenes limpias
        gt = cargar_gt(ruta_json("test"))
        imgs = sorted(gt.imgs.values(), key=lambda i: i["id"])
        _, texto = predecir_oficial(modelo, carpeta / "mejor.pt", carpeta_perturbada("limpio", 0.0, imgs, False),
                                    SALIDA / "tmp" / f"{modelo}_s{semilla}", disp)
    v = re.search(r"Speed: ([\d.]+)ms pre-process, ([\d.]+)ms inference, ([\d.]+)ms NMS", texto)
    g = re.findall(r"([\d.]+) GFLOPs", texto)
    r = json.loads((carpeta / "resumen.json").read_text(encoding="utf8"))
    if not v:
        print(f"  [aviso] val.py no imprimió la velocidad de {modelo}")
        return None
    ms = sum(float(x) for x in v.groups())
    fila = {"modelo": modelo, "semilla": semilla, "n_imagenes": 108, "mediana_ms": None, "p95_ms": None,
            "media_ms": ms, "imagenes_por_s": 1000 / ms,
            "gflops_una_imagen": float(g[0]) * 1280 * 720 / (640 * 640) if g else None,
            "parametros": r.get("parametros"), "gpu_pico_gb": None,
            "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU" if disp.type == "cuda" else "cpu",
            "metodo_latencia": "val.py del repo: media de preproceso + inferencia + NMS; GFLOPs del repo x 2.25"}
    print(f"  latencia {modelo} s{semilla}: media {ms:.1f} ms (val.py), GFLOPs {fila['gflops_una_imagen']}")
    return fila


def latencia(modelo: str, semilla: int, disp, n_max: int | None) -> dict | None:
    import torch
    if MODELOS[modelo] == "oficial":
        return latencia_oficial(modelo, semilla, disp)
    carpeta = RAIZ / "results" / "e2" / "corridas" / f"{modelo}_s{semilla}"
    if not (carpeta / PESOS[MODELOS[modelo]]).exists():
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
         "gpu": torch.cuda.get_device_name(0) if disp.type == "cuda" else "cpu",
         "metodo_latencia": "medida aquí: mediana, sincronizando la GPU; GFLOPs con FlopCounterMode"}
    print(f"  latencia {modelo} s{semilla}: mediana {r['mediana_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms, "
          f"GFLOPs {gflops if gflops is None else round(gflops, 1)}")
    return r


def lanzar_entornos(a):
    """Relanza este script en el entorno de cada grupo de modelos, uno detrás de otro."""
    py = maquina()["python"]
    grupos: dict = {}
    for m in a.modelos:
        grupos.setdefault(ENTORNO[MODELOS[m]], []).append(m)
    for env, ms in grupos.items():
        cmd = [py[env], str(Path(__file__).resolve()), "--modelos", *ms, "--semillas", *map(str, a.semillas)]
        cmd += ["--prueba"] * a.prueba + ["--sin-latencia"] * a.sin_latencia
        print(f"\n=== entorno {env}: {ms}", flush=True)
        r = subprocess.run(cmd, cwd=RAIZ, env={**__import__("os").environ, "PYTHONUTF8": "1"})
        if r.returncode:
            print(f"!! el entorno {env} terminó con código {r.returncode}; sigo con el siguiente", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=None, choices=list(MODELOS))
    ap.add_argument("--faltantes", action="store_true", help="los 9 modelos de MMDetection y de los repos oficiales")
    ap.add_argument("--entornos", action="store_true", help="relanzar en el entorno de cada modelo (maquina_e2.json)")
    ap.add_argument("--semillas", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--prueba", action="store_true", help="5 imágenes y una semilla; no escribe resultados finales")
    ap.add_argument("--sin-latencia", action="store_true")
    a = ap.parse_args()
    a.modelos = a.modelos or (FALTANTES if a.faltantes else list(MODELOS)[:4])
    if a.entornos:
        return lanzar_entornos(a)
    if a.prueba:
        a.semillas = a.semillas[:1]
        if a.modelos == list(MODELOS)[:4]:
            a.modelos = ["maskrcnn"]
    n_max = 5 if a.prueba else None
    import torch
    solo_oficial = all(MODELOS[m] == "oficial" for m in a.modelos)
    # los repos oficiales predicen en un proceso hijo: aquí no se abre contexto de CUDA (le quitaría VRAM)
    disp = torch.device("cuda:0") if solo_oficial and torch.cuda.is_available() else preparar_gpu(permitir_cpu=False)
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
    if lat and not a.prueba:  # se agrega a lo que ya había: cada entorno escribe solo sus modelos
        previo = pd.read_csv(SALIDA / "latencia.csv") if (SALIDA / "latencia.csv").exists() else pd.DataFrame()
        nuevo = pd.DataFrame(lat)
        if len(previo):
            previo = previo[~previo.modelo.isin(nuevo.modelo)]
        pd.concat([previo, nuevo], ignore_index=True).to_csv(SALIDA / "latencia.csv", index=False)
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
