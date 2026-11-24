"""Paper guía del E2 (enunciado §5; preregistro §7): réplica parcial de Kalitsios et al. 2022 y medición de su limitación.

Kalitsios et al. 2022 (DOI 10.1109/ICRCV55858.2022.9953178) entrenan Mask R-CNN ResNet-101 con la partición oficial
de WDSD (765/90/90 imágenes) y reportan AP de máscara 0.586 y AP50 0.849 en test.

1. Réplica: Mask R-CNN ResNet-101 (MMDetection, preentrenado en COCO) con la partición oficial, misma métrica (AP de
   máscara COCO en test). Criterio de éxito fijado antes de correr: ver criterio_paper_guia_E2.md.
2. Limitación: la partición oficial es aleatoria por imagen y casi la mitad de test tiene un casi gemelo en train.
   Se entrenan 3 modelos del Top 10 (MS R-CNN, PointRend, YOLO26) con la partición oficial y se compara su desempeño
   en las imágenes de test con y sin casi gemelo (auditoria_fuga_e2.py marca cuáles).

Todo con el protocolo de las corridas finales del E2: 6 épocas, lote efectivo 2, coseno, anclas de 8–128 px en los de
MMDetection, misma evaluación. Tasa: la elegida en la búsqueda del E2 para cada modelo; para ResNet-101, que no pasó
por la búsqueda, la tasa por defecto (factor 1). Las épocas no son las del paper (no las reporta en lo que citamos):
se declara.

Uso (desde Trabajo\\datos\\wdsd, con el entorno wdsd_dl):
    python paper_guia_e2.py datos         # crea datos_oficial/ (COCO y YOLO con la partición oficial), segundos
    python paper_guia_e2.py verificar     # construye la configuración de ResNet-101 en wdsd_mmdet, segundos
    python paper_guia_e2.py correr        # 4 modelos x semillas 42 43 44, ~3.5 h en la 3060
    python paper_guia_e2.py correr --semillas 42 43 44 45 46 --modelos maskrcnn_r101
Va semilla por semilla. Se puede interrumpir y volver a lanzar: las corridas terminadas se saltan.
Salida: results/e2/paper_guia/<modelo>_s<semilla>/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))
import entrenar_e2 as E  # noqa: E402

DATOS_OF = RAIZ / "datos_oficial"
MODELOS = ["maskrcnn_r101", "msrcnn", "pointrend", "yolo26"]
EPOCAS = 6
ESPERADO = {"train": 765, "val": 90, "test": 90}


def cmd_datos(a):
    from wdsd_inst.datos import _escribir_yolo
    origen = E.RAIZ / "datos_inst" / "coco"
    nuevo = {s: {"images": [], "annotations": [], "categories": None} for s in ESPERADO}
    for s in ("train", "val", "test"):
        d = json.loads((origen / f"{s}.json").read_text(encoding="utf8"))
        por_img = defaultdict(list)
        for x in d["annotations"]:
            por_img[x["image_id"]].append(x)
        for im in d["images"]:
            dst = nuevo[im["split_oficial"]]
            dst["categories"] = d["categories"]
            nid = len(dst["images"]) + 1
            dst["images"].append({**im, "id": nid, "split_agrupado": s})
            for x in por_img[im["id"]]:
                dst["annotations"].append({**x, "id": len(dst["annotations"]) + 1, "image_id": nid})
    n = {s: len(v["images"]) for s, v in nuevo.items()}
    if n != ESPERADO:
        sys.exit(f"La partición oficial no da 765/90/90: {n}")
    (DATOS_OF / "coco").mkdir(parents=True, exist_ok=True)
    for s, v in nuevo.items():
        (DATOS_OF / "coco" / f"{s}.json").write_text(json.dumps(v), encoding="utf8")
    _escribir_yolo(nuevo, DATOS_OF / "yolo")
    # train93: misma regla que entrenar_e2.py datos, sobre el train oficial (solo para que la corrida final no falle)
    tr = nuevo["train"]
    ids = set(random.Random(0).sample(sorted(i["id"] for i in tr["images"]), E.N_TRAIN93))
    sub = {"images": [i for i in tr["images"] if i["id"] in ids],
           "annotations": [x for x in tr["annotations"] if x["image_id"] in ids], "categories": tr["categories"]}
    (DATOS_OF / "coco" / "train93.json").write_text(json.dumps(sub), encoding="utf8")
    lista = [(DATOS_OF / "yolo" / "images" / "train" / f"{Path(i['file_name']).stem}.png").resolve().as_posix()
             for i in sub["images"]]
    (DATOS_OF / "yolo" / "train93.txt").write_text("\n".join(lista) + "\n", encoding="utf8")
    grupos = {s: len({i["grupo"] for i in v["images"]}) for s, v in nuevo.items()}
    comp = defaultdict(int)
    for i in nuevo["test"]["images"]:
        comp[i["split_agrupado"]] += 1
    print(f"datos_oficial listo: imágenes {n}, tornillos { {s: len(v['annotations']) for s, v in nuevo.items()} }, "
          f"grupos {grupos}. El test oficial viene de la partición agrupada así: {dict(comp)}")


def cmd_verificar(a):
    cfg = E.maquina()
    py = cfg["python"][E.R.MODELOS["maskrcnn_r101"]["entorno"]]
    codigo = ("import sys; from pathlib import Path; sys.path.insert(0, 'src'); from wdsd_e2 import m_mmdet; "
              "c, i = m_mmdet.configuracion('maskrcnn_r101', Path('results/e2/paper_guia/_verificar'), 42, 6, 1.0); "
              "print('backbone depth', c.model.backbone.depth, '| anclas', c.model.rpn_head.anchor_generator.scales, "
              "'| train', c.train_dataloader.dataset.ann_file); print(i)")
    env = {**os.environ, "WDSD_DATOS": str(DATOS_OF), "PYTHONUTF8": "1"}
    subprocess.run([py, "-c", codigo], cwd=RAIZ, env=env, check=True)


def cmd_correr(a):
    if not (DATOS_OF / "coco" / "train93.json").exists():
        sys.exit("Falta datos_oficial/: corre primero `python paper_guia_e2.py datos`.")
    cfg = E.maquina()
    opciones = types.SimpleNamespace(permitir_cpu=False, sin_preentrenar=False)
    valores = {m: 1.0 if m == "maskrcnn_r101" else json.loads(
        (E.SALIDA / "busqueda" / f"{m}_elegida.json").read_text(encoding="utf8"))["elegido"] for m in a.modelos}
    os.environ["WDSD_DATOS"] = str(DATOS_OF)  # lanzar() pasa el entorno al proceso hijo
    salida = E.SALIDA / "paper_guia"
    salida.mkdir(parents=True, exist_ok=True)
    print(f"Paper guía, partición oficial: {a.modelos} x semillas {a.semillas}, {EPOCAS} épocas, tasas {valores}",
          flush=True)
    for s in a.semillas:
        for m in a.modelos:
            ok = E.lanzar(m, "final", s, EPOCAS, valores[m], salida / f"{m}_s{s}", opciones, cfg)
            print(f"-- {m} s{s}: {'terminó' if ok else 'FALLÓ'}", flush=True)
    print("Paper guía terminado. Resultados en", salida)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datos"); sub.add_parser("verificar")
    c = sub.add_parser("correr")
    c.add_argument("--semillas", type=int, nargs="+", default=[42, 43, 44])
    c.add_argument("--modelos", nargs="+", default=MODELOS, choices=MODELOS)
    a = p.parse_args()
    {"datos": cmd_datos, "verificar": cmd_verificar, "correr": cmd_correr}[a.cmd](a)


if __name__ == "__main__":
    main()
