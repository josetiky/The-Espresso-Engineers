"""
Entrena el modelo YOLO propio para Copa FutBotMX con AUMENTACIÓN FUERTE,
respaldo automático del modelo anterior e impresión de métricas (mAP).

v2 — Mejoras respecto a v1:
  · AUMENTACIÓN DE DATOS agresiva (mosaic, mixup, copy-paste, random-erasing,
    cambios de escala/brillo/perspectiva). Esto enseña al modelo a reconocer
    robots BORROSOS, OCLUIDOS y a DISTINTA ESCALA → ataca directamente el
    "flicker" (reconoce/deja de reconocer) sin necesitar más fotos.
  · NO sobrescribe el modelo anterior: el nuevo se guarda como futbotmx_v2.pt
    (o el siguiente número libre) y el viejo se respalda.
  · Lee dataset/data.yaml y muestra las clases reales (sirve para ajustar
    CONFIG['class_robot'] / CONFIG['class_ball'] en pipeline.py).
  · Al terminar imprime las métricas: mAP@50, mAP@50-95, precisión y recall.

REQUISITOS PREVIOS:
  1. En Roboflow exporta el dataset en formato **YOLOv11** (NO COCO).
  2. Descomprime el ZIP en ./dataset/ con esta estructura:
        dataset/
        ├── data.yaml
        ├── train/{images,labels}/
        ├── valid/{images,labels}/
        └── test/{images,labels}/
  3. Revisa el orden de las clases que imprime este script y ajústalo en
     pipeline.py (CONFIG['class_robot'] / CONFIG['class_ball']).

Uso:
    python entrenar_modelo.py                 # 80 épocas por defecto
    python entrenar_modelo.py --epochs 120    # más entrenamiento
    python entrenar_modelo.py --modelo yolo11m.pt   # modelo más grande
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _next_free(path_stem: str, suffix: str = ".pt") -> Path:
    """Devuelve futbotmx_v2.pt, o v3, v4… si ya existen (no sobrescribe)."""
    n = 2
    while Path(f"{path_stem}{n}{suffix}").exists():
        n += 1
    return Path(f"{path_stem}{n}{suffix}")


def _mostrar_clases(data_yaml: str):
    try:
        with open(data_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        print(f"[INFO] Dataset: {data.get('nc', len(names))} clases → {names}")
        print("[INFO] Recuerda ajustar en pipeline.py CONFIG según ESTE orden:")
        for i, nm in enumerate(names):
            print(f"         clase {i} = {nm}")
        return names
    except Exception as e:
        print(f"[WARN] No pude leer {data_yaml}: {e}")
        return []


def entrenar(epochs=80, modelo_base="yolo11s.pt", imgsz=640, batch=8):
    data_yaml = "dataset/data.yaml"
    if not Path(data_yaml).exists():
        raise FileNotFoundError(
            "No existe dataset/data.yaml. Exporta el dataset de Roboflow en "
            "formato YOLOv11 y descomprímelo en ./dataset/")

    _mostrar_clases(data_yaml)
    device = _device()
    print(f"[INFO] Dispositivo: {device.upper()} | base: {modelo_base} | "
          f"épocas: {epochs}")

    # ── Respaldo del modelo anterior (no se pierde) ────────────────────
    anterior = Path("futbotmx_v1.pt")
    if anterior.exists():
        backup = Path(f"futbotmx_v1_backup_{datetime.now():%Y%m%d_%H%M}.pt")
        shutil.copy2(anterior, backup)
        print(f"[INFO] Respaldo del modelo anterior → {backup}")

    model = YOLO(modelo_base)   # transfer learning desde COCO

    run_name = f"futbotmx_v2_{datetime.now():%Y%m%d_%H%M}"
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=20,            # early-stopping si deja de mejorar
        name=run_name,
        seed=42,                # reproducibilidad (Requisito Profesional)

        # ── AUMENTACIÓN para robustez ante blur / oclusión / escala ────
        hsv_h=0.015,            # tono (pequeño)
        hsv_s=0.7,              # saturación (cambios de cancha/iluminación)
        hsv_v=0.4,              # brillo (sombras, reflejos) → robots oscuros/claros
        degrees=5.0,           # rotación leve de la cámara
        translate=0.10,        # desplazamiento
        scale=0.5,             # ZOOM in/out → robots grandes y pequeños
        shear=2.0,
        perspective=0.0005,    # perspectiva (cámara móvil)
        fliplr=0.5,            # espejo horizontal
        flipud=0.0,            # nunca volteo vertical (la cancha tiene "arriba")
        mosaic=1.0,            # combina 4 imágenes → muchas escalas/contextos
        close_mosaic=10,       # apaga mosaic en las últimas 10 épocas (afina)
        mixup=0.15,            # mezcla imágenes → robustez general
        copy_paste=0.1,        # pega objetos → simula MÁS robots / oclusiones
        erasing=0.4,           # borra parches → simula OCLUSIÓN parcial
    )

    # ── Copiar pesos al raíz sin sobrescribir el modelo viejo ──────────
    best = Path("runs/detect") / run_name / "weights" / "best.pt"
    destino = _next_free("futbotmx_v")
    if best.exists():
        shutil.copy2(best, destino)
        print(f"\n[OK] Pesos nuevos → {destino}")
        print(f"     Actualiza pipeline.py:  CONFIG['yolo_model'] = '{destino.name}'")
    else:
        print(f"[WARN] No encontré {best}; revisa la carpeta runs/detect/{run_name}/")

    # ── Métricas finales (mAP, precisión, recall) ──────────────────────
    try:
        print("\n[INFO] Evaluando en el set de validación…")
        metrics = model.val(data=data_yaml, device=device, verbose=False)
        b = metrics.box
        print("\n──────────── MÉTRICAS DEL MODELO ────────────")
        print(f"  mAP@50      : {b.map50:.3f}")
        print(f"  mAP@50-95   : {b.map:.3f}")
        print(f"  Precisión   : {b.mp:.3f}")
        print(f"  Recall      : {b.mr:.3f}")
        print("─────────────────────────────────────────────")
        print("  (mAP@50 > 0.8 suele dar un tracking estable; "
              "recall bajo = se 'pierden' objetos → más flicker)")
    except Exception as e:
        print(f"[WARN] No se pudieron calcular métricas: {e}")

    print("\n[DONE] Entrenamiento terminado.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--modelo", default="yolo11s.pt",
                    help="pesos base: yolo11n/s/m/l.pt (más grande = más preciso y lento)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    entrenar(epochs=args.epochs, modelo_base=args.modelo,
             imgsz=args.imgsz, batch=args.batch)
