"""
sam3_demo.py — Demostración de SAM 3 con PROMPTS DE TEXTO (vocabulario abierto).
────────────────────────────────────────────────────────────────────────
Cumple la línea de innovación de la convocatoria (3.7.3): usar SAM 3 para
segmentar los elementos del partido mediante CONCEPTOS en texto, sin cajas
previas ni entrenamiento. Toma unos frames del video y segmenta:
   · "soccer robot"      → robots
   · "orange ball"       → balón
   · "green soccer field"→ cancha
Guarda imágenes con las máscaras superpuestas en output/sam3_demo/.

REQUISITOS (ver utils/sam3_backend.py):
  pip install transformers torch
  huggingface-cli login          # acceso a facebook/sam3 (modelo gated)

Uso:
    python sam3_demo.py                         # assets/partido.mp4, 4 frames
    python sam3_demo.py assets/partido.mp4 6
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import supervision as sv

PROMPTS = [
    ("soccer robot",       (60, 220, 30)),    # verde
    ("orange ball",        (0, 140, 255)),    # naranja
    ("green soccer field", (200, 120, 0)),    # azul (contorno cancha)
]


def muestrear(video, n):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(int(total * 0.2), int(total * 0.8), n).astype(int)
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append((int(i), fr))
    cap.release()
    return out


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "assets/partido.mp4"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if not Path(video).exists():
        raise SystemExit(f"No existe el video: {video}")

    try:
        from utils.sam3_backend import Sam3Backend
        sam3 = Sam3Backend()
    except Exception as e:
        raise SystemExit(
            f"[ERROR] No se pudo cargar SAM 3: {e}\n"
            "Instala 'transformers' y haz 'huggingface-cli login' con acceso a\n"
            "huggingface.co/facebook/sam3 (modelo gated).")

    out = Path("output/sam3_demo"); out.mkdir(parents=True, exist_ok=True)
    frames = muestrear(video, n)
    print(f"[INFO] {len(frames)} frames · prompts: {[p for p,_ in PROMPTS]}\n")

    for fidx, fr in frames:
        vis = fr.copy()
        resumen = []
        for text, color_bgr in PROMPTS:
            dets = sam3.segment(fr, text, threshold=0.5)
            if len(dets) > 0:
                col = sv.Color(color_bgr[2], color_bgr[1], color_bgr[0])
                # color_lookup=INDEX: las máscaras de SAM 3 no traen class_id
                ann = sv.MaskAnnotator(color=col, opacity=0.5,
                                       color_lookup=sv.ColorLookup.INDEX)
                vis = ann.annotate(vis, dets)
                resumen.append(f"{text}: {len(dets)}")
        cv2.putText(vis, " | ".join(resumen) or "sin detecciones",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ruta = out / f"sam3_frame_{fidx}.jpg"
        cv2.imwrite(str(ruta), vis)
        print(f"[OK] {ruta}  ({' | '.join(resumen) or 'sin detecciones'})")

    print(f"\n[DONE] Resultados de SAM 3 (texto) → {out}/")
    print("Úsalos en el README/video como demostración de 'segmentación por")
    print("concepto / vocabulario abierto' con SAM 3 (innovación 3.7.3).")


if __name__ == "__main__":
    main()
