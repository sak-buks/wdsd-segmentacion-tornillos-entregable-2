"""Prueba de sanidad con etiquetas permutadas (enunciado §10.1) sobre el mejor modelo, Mask R-CNN.

Idea: si se entrena con etiquetas que no tienen relación con las imágenes, el modelo no puede aprender a encontrar
tornillos, así que su F1 en test (con las etiquetas verdaderas) debe caer cerca de 0. Si no cae, algo del flujo de
evaluación filtra información (por ejemplo, test mezclado con train o una métrica que premia cualquier salida).

Cómo se permutan: a cada imagen de train se le asignan los tornillos anotados de otra imagen de train del mismo tamaño
y de OTRO grupo de escena (dentro de un grupo las fotos son casi iguales y la permutación dejaría etiquetas casi
correctas). Val y test no se tocan. Todo lo demás es igual a la corrida final maskrcnn_s42: 6 épocas, lote 2, tasa
elegida en la búsqueda, semilla 42, misma evaluación (la mejor época y el umbral de score se eligen en val, con
etiquetas verdaderas; eso favorece un poco a la corrida permutada, así que la prueba es conservadora).

Criterio fijado antes de correr: pasa si el F1 por grupo en test de la corrida permutada es <= 0.10 (la corrida real
da 0.73). Se informa también el F1 en train93 (93 imágenes de train evaluadas contra sus etiquetas permutadas): si es
mayor que el de test, el modelo memorizó etiquetas sin sentido, lo esperable en una red de este tamaño.

Uso (desde Trabajo\\datos\\wdsd, con cualquier python: el orquestador solo usa biblioteca estándar):
    python sanidad_e2.py datos      # crea datos_permutado/coco (segundos)
    python sanidad_e2.py correr     # 1 corrida, ~18 min en la 3060
    python sanidad_e2.py analizar   # compara con maskrcnn_s42 y escribe results/e2/sanidad/sanidad.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import entrenar_e2 as E  # noqa: E402

DATOS_P = RAIZ / "datos_permutado"
SEMILLA = 42
EPOCAS = 6
UMBRAL_PASA = 0.10
CARPETA = E.SALIDA / "sanidad" / f"maskrcnn_permutado_s{SEMILLA}"


def derangement_entre_grupos(imgs: list[dict], rng: random.Random) -> dict:
    """Devuelve {id destino: id origen} con origen de otro grupo y del mismo tamaño."""
    pi = {}
    por_tam = defaultdict(list)
    for im in imgs:
        por_tam[(im["height"], im["width"])].append(im)
    for bloque in por_tam.values():
        ids = [im["id"] for im in bloque]
        grupo = {im["id"]: im["grupo"] for im in bloque}
        if len({grupo[i] for i in ids}) < 2:
            sys.exit(f"Hay un tamaño de imagen con un solo grupo: no se puede permutar entre grupos ({len(ids)} imágenes)")
        orig = ids[:]
        rng.shuffle(orig)
        malos = lambda: [k for k in range(len(ids)) if grupo[ids[k]] == grupo[orig[k]]]  # noqa: E731
        for _ in range(100):
            m = malos()
            if not m:
                break
            for k in m:  # intercambio con una posición al azar si arregla las dos
                for j in rng.sample(range(len(ids)), len(ids)):
                    if grupo[ids[k]] != grupo[orig[j]] and grupo[ids[j]] != grupo[orig[k]]:
                        orig[k], orig[j] = orig[j], orig[k]
                        break
        if malos():
            sys.exit("No se encontró una permutación entre grupos")
        pi.update(dict(zip(ids, orig)))
    return pi


def cmd_datos(a):
    origen = E.RAIZ / "datos_inst" / "coco"
    (DATOS_P / "coco").mkdir(parents=True, exist_ok=True)
    for s in ("val", "test"):
        (DATOS_P / "coco" / f"{s}.json").write_text((origen / f"{s}.json").read_text(encoding="utf8"), encoding="utf8")
    tr = json.loads((origen / "train.json").read_text(encoding="utf8"))
    por_img = defaultdict(list)
    for x in tr["annotations"]:
        por_img[x["image_id"]].append(x)
    pi = derangement_entre_grupos(tr["images"], random.Random(0))
    anns = []
    for im in tr["images"]:
        for x in por_img[pi[im["id"]]]:
            anns.append({**x, "id": len(anns) + 1, "image_id": im["id"]})
    nuevo = {**tr, "annotations": anns}
    (DATOS_P / "coco" / "train.json").write_text(json.dumps(nuevo), encoding="utf8")
    ids = set(random.Random(0).sample(sorted(i["id"] for i in tr["images"]), E.N_TRAIN93))  # misma regla que E2
    sub = {"images": [i for i in tr["images"] if i["id"] in ids],
           "annotations": [x for x in anns if x["image_id"] in ids], "categories": tr["categories"]}
    (DATOS_P / "coco" / "train93.json").write_text(json.dumps(sub), encoding="utf8")
    grupo = {im["id"]: im["grupo"] for im in tr["images"]}
    iguales = sum(grupo[i] == grupo[j] for i, j in pi.items())
    (DATOS_P / "permutacion.json").write_text(json.dumps({str(k): v for k, v in pi.items()}), encoding="utf8")
    print(f"datos_permutado listo: {len(tr['images'])} imágenes de train, {len(anns)} tornillos "
          f"(antes {len(tr['annotations'])}), imágenes con etiquetas de su mismo grupo: {iguales}. Val y test sin cambios.")


def cmd_correr(a):
    if not (DATOS_P / "coco" / "train93.json").exists():
        sys.exit("Falta datos_permutado/: corre primero `python sanidad_e2.py datos`.")
    cfg = E.maquina()
    valor = json.loads((E.SALIDA / "busqueda" / "maskrcnn_elegida.json").read_text(encoding="utf8"))["elegido"]
    os.environ["WDSD_DATOS"] = str(DATOS_P)  # lanzar() pasa el entorno al proceso hijo
    os.environ.pop("WDSD_ABLACION", None)
    opciones = types.SimpleNamespace(permitir_cpu=False, sin_preentrenar=False)
    CARPETA.parent.mkdir(parents=True, exist_ok=True)
    print(f"Sanidad: Mask R-CNN con etiquetas de train permutadas, semilla {SEMILLA}, {EPOCAS} épocas, tasa x{valor:g}")
    ok = E.lanzar("maskrcnn", "final", SEMILLA, EPOCAS, valor, CARPETA, opciones, cfg)
    print("terminó" if ok else "FALLÓ")


def cmd_analizar(a):
    real = json.loads((E.SALIDA / "corridas" / f"maskrcnn_s{SEMILLA}" / "metricas.json").read_text())
    perm = json.loads((CARPETA / "metricas.json").read_text())
    fila = lambda m: {p: {k: m.get(p, {}).get(k) for k in ("F1_grupo", "AP", "AP50")} for p in ("val", "test", "train93")}  # noqa: E731
    r = {"criterio": f"pasa si F1 por grupo en test de la corrida permutada <= {UMBRAL_PASA}",
         "real": fila(real), "permutada": fila(perm), "umbral_score": {"real": real.get("umbral_score"),
                                                                     "permutada": perm.get("umbral_score")}}
    r["pasa"] = bool(perm["test"]["F1_grupo"] <= UMBRAL_PASA)
    (CARPETA.parent / "sanidad.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf8")
    print(json.dumps(r, indent=1, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("datos", "correr", "analizar"):
        sub.add_parser(c)
    a = p.parse_args()
    {"datos": cmd_datos, "correr": cmd_correr, "analizar": cmd_analizar}[a.cmd](a)


if __name__ == "__main__":
    main()
