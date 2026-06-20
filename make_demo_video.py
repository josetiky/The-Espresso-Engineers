"""
make_demo_video.py — Video demo lado-a-lado (Requisito 3.5.3)
────────────────────────────────────────────────────────────────────────
La convocatoria exige un video (máx. 2 min) que muestre "la vista del video
original junto al resultado segmentado (lado a lado o superpuesto)" con
"indicadores visuales claros" y una "breve explicación del enfoque".

Este script toma:
  · el video ORIGINAL          (assets/partido.mp4)
  · el video ANALIZADO         (output/partido_analizado.mp4)
y los compone LADO A LADO, recortados a una duración máxima, con títulos y
un pie de página explicativo. El resultado: output/demo_lado_a_lado.mp4

Uso:
    python make_demo_video.py
    python make_demo_video.py original.mp4 analizado.mp4 salida.mp4 --segundos 90
"""

import sys
import argparse
import cv2
import numpy as np


def _put_panel_title(frame, text, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)
    return frame


def make_demo(original_path, analizado_path, out_path="output/demo_lado_a_lado.mp4",
              max_seconds=120, panel_h=540, footer=(
                  "Pipeline: YOLO (modelo propio) -> ByteTrack -> SAM -> "
                  "Homografia -> Estadisticas en metros")):
    cap_o = cv2.VideoCapture(original_path)
    cap_a = cv2.VideoCapture(analizado_path)
    if not cap_o.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el original: {original_path}")
    if not cap_a.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el analizado: {analizado_path}")

    fps = cap_a.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(fps * max_seconds)

    # Dimensiones de cada panel (manteniendo aspecto del original)
    ow = int(cap_o.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap_o.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = panel_h / oh
    pw = int(ow * scale)
    footer_h = 46

    out_w = pw * 2
    out_h = panel_h + footer_h
    fourcc = cv2.VideoWriter_fourcc(*"mp44") if False else cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    n = 0
    while n < max_frames:
        ro, fo = cap_o.read()
        ra, fa = cap_a.read()
        if not ro or not ra:
            break
        fo = cv2.resize(fo, (pw, panel_h))
        fa = cv2.resize(fa, (pw, panel_h))
        fo = _put_panel_title(fo, "ORIGINAL")
        fa = _put_panel_title(fa, "ANALISIS (segmentacion + tracking)", (0, 255, 180))

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:panel_h, :pw] = fo
        canvas[:panel_h, pw:] = fa
        # Pie explicativo
        cv2.rectangle(canvas, (0, panel_h), (out_w, out_h), (15, 15, 15), -1)
        cv2.putText(canvas, footer, (12, panel_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
        writer.write(canvas)
        n += 1

    cap_o.release(); cap_a.release(); writer.release()
    print(f"[OK] Demo lado a lado → {out_path}  ({n} frames, {n/fps:.1f}s)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("original", nargs="?", default="assets/partido.mp4")
    ap.add_argument("analizado", nargs="?", default="output/partido_analizado.mp4")
    ap.add_argument("salida", nargs="?", default="output/demo_lado_a_lado.mp4")
    ap.add_argument("--segundos", type=int, default=120)
    args = ap.parse_args()
    make_demo(args.original, args.analizado, args.salida, max_seconds=args.segundos)
