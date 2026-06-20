"""
diagnostico_deteccion.py — Encuentra la MEJOR configuración del detector.
────────────────────────────────────────────────────────────────────────
En vez de adivinar la confianza y la resolución, este script corre tu modelo
sobre varios frames del video con distintas combinaciones de:
   · resolución de inferencia (imgsz): 640, 768, 960, 1280
   · umbral de confianza:              0.15, 0.25, 0.40, 0.55
y reporta cuántos robots y balones detecta en promedio. Como hay EXACTAMENTE
4 robots, la mejor configuración es la que detecta ~4 robots de forma estable
sin inventar de más. También guarda frames anotados para que los veas.

Uso:
    python diagnostico_deteccion.py                       # usa assets/partido.mp4
    python diagnostico_deteccion.py assets/otro.mp4 12    # video + nº de frames
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Mismas clases que el pipeline (ver dataset/data.yaml): 0=Robot, 1=pelota
CLASS_ROBOT, CLASS_BALL = 0, 1
MODELO = "futbotmx_v2.pt"
IMGSZ_OPC = [640, 768, 960, 1280]
CONF_OPC = [0.15, 0.25, 0.40, 0.55]


def muestrear_frames(video_path, n=10):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(int(total * 0.1), int(total * 0.9), n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append((int(i), fr))
    cap.release()
    return frames


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "assets/partido.mp4"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if not Path(video).exists():
        raise SystemExit(f"No existe el video: {video}")
    if not Path(MODELO).exists():
        raise SystemExit(f"No existe el modelo: {MODELO}")

    from ultralytics import YOLO
    import torch
    dev = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(MODELO)
    frames = muestrear_frames(video, n)
    print(f"[INFO] {len(frames)} frames de muestra · modelo {MODELO} · device {dev}\n")

    out = Path("output/diagnostico"); out.mkdir(parents=True, exist_ok=True)
    print(f"{'imgsz':>6} {'conf':>6} {'robots/frame':>13} {'balón %frames':>14}")
    print("─" * 44)

    mejor = None
    for imgsz in IMGSZ_OPC:
        for conf in CONF_OPC:
            tot_rob, con_balon = 0, 0
            for fidx, fr in frames:
                r = model(fr, imgsz=imgsz, conf=conf, device=dev, verbose=False)[0]
                cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.array([])
                n_rob = int(np.sum(cls == CLASS_ROBOT))
                n_bal = int(np.sum(cls == CLASS_BALL))
                tot_rob += n_rob
                con_balon += (1 if n_bal > 0 else 0)
            rob_avg = tot_rob / max(len(frames), 1)
            bal_pct = con_balon / max(len(frames), 1) * 100
            # "bondad": cercanía a 4 robots + detección de balón
            score = -abs(rob_avg - 4) + bal_pct / 100
            flag = ""
            if mejor is None or score > mejor[0]:
                mejor = (score, imgsz, conf, rob_avg, bal_pct); flag = "  ←"
            print(f"{imgsz:>6} {conf:>6.2f} {rob_avg:>13.1f} {bal_pct:>13.0f}%{flag}")

    print("\n" + "═" * 44)
    _, imgsz, conf, rob_avg, bal_pct = mejor
    print(f"RECOMENDADO →  infer_imgsz={imgsz} · confidence={conf}")
    print(f"   (detecta {rob_avg:.1f} robots/frame y balón en {bal_pct:.0f}% de los frames)")
    print("Pon esos valores en CONFIG (pipeline.py): infer_imgsz y confidence.")

    # Guardar frames anotados con la mejor configuración
    for fidx, fr in frames[:5]:
        r = model(fr, imgsz=imgsz, conf=conf, device=dev, verbose=False)[0]
        cv2.imwrite(str(out / f"frame_{fidx}.jpg"), r.plot())
    print(f"\n[OK] Frames anotados (mejor config) → {out}/  (míralos para confirmar)")


if __name__ == "__main__":
    main()
