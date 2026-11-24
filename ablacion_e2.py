"""Estudio de ablación del E2 (enunciado §4.6) sobre el mejor modelo, Mask R-CNN.

Quita un componente a la vez y entrena con las mismas 5 semillas, las mismas 6 épocas, la misma tasa elegida en la
búsqueda y la misma evaluación que las corridas finales. El modelo completo no se vuelve a entrenar: sus 5 corridas
finales (results/e2/corridas/maskrcnn_s42..46) se hicieron en esta misma GPU y son la referencia.

Componentes (ver src/wdsd_inst/maskrcnn.py):
- anclas_defecto: anclas de 32–512 px de torchvision en vez de 8–128 px (decisión tomada por el EDA: el lado mediano
  de un tornillo es 12 px).
- solo_imagenet: backbone preentrenado solo en ImageNet en vez del detector completo preentrenado en COCO.
- sin_volteo: sin volteo horizontal (la única aumentación).

Uso (desde Trabajo\\datos\\wdsd, con cualquier python: el orquestador solo usa biblioteca estándar):
    python ablacion_e2.py --verificar         # 1 min: construye las 3 variantes en wdsd_dl y muestra qué cambia
    python ablacion_e2.py                     # 15 corridas, ~4.5 h en la 3060
    python ablacion_e2.py --semillas 42 43 44 # parcial
Va semilla por semilla (las 3 variantes de la 42, luego las de la 43...), así que si se corta quedan todas las
variantes con el mismo número de semillas. Se puede interrumpir y volver a lanzar: las corridas terminadas se saltan.
Salida: results/e2/ablacion/<variante>_s<semilla>/ (mismos archivos que una corrida final).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import types
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import entrenar_e2 as E  # noqa: E402

VARIANTES = ["anclas_defecto", "solo_imagenet", "sin_volteo"]
EPOCAS = 6  # presupuesto de las corridas finales

VERIFICAR = r"""
import os, sys
sys.path.insert(0, "src")
from wdsd_inst import maskrcnn
for v in [""] + %r:
    os.environ["WDSD_ABLACION"] = v
    m = maskrcnn.crear_modelo(True)
    tam = [s[0] for s in m.rpn.anchor_generator.sizes]
    rpn = float(m.rpn.head.conv[0][0].weight.abs().mean())
    print(f"{v or 'completo':15s} anclas {tam}  volteo {'sin_volteo' not in maskrcnn.ablacion()}  "
          f"params {sum(p.numel() for p in m.parameters()):,}  |w| conv RPN {rpn:.4f} (cambia si RPN parte de cero)")
""" % VARIANTES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--semillas", type=int, nargs="+", default=list(range(42, 47)))
    p.add_argument("--variantes", nargs="+", default=VARIANTES, choices=VARIANTES)
    p.add_argument("--verificar", action="store_true")
    a = p.parse_args()
    cfg = E.maquina()
    py = cfg["python"][E.R.MODELOS["maskrcnn"]["entorno"]]
    if a.verificar:
        subprocess.run([py, "-c", VERIFICAR], cwd=RAIZ, check=True)
        return
    valor = json.loads((E.SALIDA / "busqueda" / "maskrcnn_elegida.json").read_text(encoding="utf8"))["elegido"]
    opciones = types.SimpleNamespace(permitir_cpu=False, sin_preentrenar=False)
    salida = E.SALIDA / "ablacion"
    salida.mkdir(parents=True, exist_ok=True)
    print(f"Ablación de Mask R-CNN: {a.variantes} x semillas {a.semillas}, {EPOCAS} épocas, tasa x{valor:g}", flush=True)
    for s in a.semillas:
        for v in a.variantes:
            os.environ["WDSD_ABLACION"] = v  # lanzar() pasa el entorno al proceso hijo
            carpeta = salida / f"{v}_s{s}"
            ok = E.lanzar("maskrcnn", "final", s, EPOCAS, valor, carpeta, opciones, cfg)
            if ok:
                (carpeta / "ablacion.json").write_text(json.dumps({"sin": v}), encoding="utf8")
            print(f"-- {v} s{s}: {'terminó' if ok else 'FALLÓ'}", flush=True)
    os.environ.pop("WDSD_ABLACION", None)
    print("Ablación terminada. Resultados en", salida)


if __name__ == "__main__":
    main()
