"""
calibrar_balon.py — Encuentra el mejor rango HSV para el balón (naranja).
────────────────────────────────────────────────────────────────────────
Clave del reglamento (regla 3.7): los robots NO pueden tener naranja, amarillo
ni azul. Por eso una mancha NARANJA sobre la cancha verde es, con casi total
seguridad, el BALÓN (pelota de golf naranja brillante, 42 mm). La detección
por color es entonces muy fiable — mejor que YOLO, que aquí ve el balón en
solo ~30-40% de los frames.

Este script prueba varios rangos HSV de naranja sobre frames reales del video
y reporta en qué % de frames encuentra el balón dentro de la cancha. Imprime
el MEJOR rango listo para pegar en CONFIG (ball_hsv_lower / ball_hsv_upper).

Uso:
    python calibrar_balon.py                    # assets/partido.mp4, 30 frames
    python calibrar_balon.py assets/otro.mp4 40
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Candidatos de rango naranja (H, S, V) low/high — del más estricto al más amplio
CANDIDATOS = {
    "estricto":  (np.array([8, 150, 150]), np.array([20, 255, 255])),
    "medio":     (np.array([5, 120, 120]), np.array([22, 255, 255])),
    "amplio":    (np.array([3, 90, 90]),   np.array([25, 255, 255])),
    "muy_amplio":(np.array([2, 70, 70]),   np.array([28, 255, 255])),
}
FIELD_LOW = np.array([35, 40, 40], np.uint8)
FIELD_HIGH = np.array([85, 255, 255], np.uint8)
BALL_MIN_AREA, BALL_MAX_AREA = 15, 3000   # px² plausibles para la pelota


def field_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, FIELD_LOW, FIELD_HIGH)
    k = np.ones((9, 9), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    if cnts:
        cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return cv2.dilate(filled, np.ones((15, 15), np.uint8))


def encuentra_balon(frame, lo, hi, fmask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.bitwise_and(mask, fmask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = [c for c in cnts if BALL_MIN_AREA <= cv2.contourArea(c) <= BALL_MAX_AREA]
    if not cand:
        return None
    c = max(cand, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return (x + w // 2, y + h // 2, int(cv2.contourArea(c)))


def muestrear(video, n):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(int(total * 0.05), int(total * 0.95), n).astype(int)
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
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    if not Path(video).exists():
        raise SystemExit(f"No existe el video: {video}")
    frames = muestrear(video, n)
    print(f"[INFO] {len(frames)} frames de muestra de {video}\n")
    print(f"{'rango':>12} {'balón %frames':>14} {'área media':>11}")
    print("─" * 40)

    mejor = None
    for nombre, (lo, hi) in CANDIDATOS.items():
        hits, areas = 0, []
        for _, fr in frames:
            fm = field_mask(fr)
            r = encuentra_balon(fr, lo, hi, fm)
            if r:
                hits += 1
                areas.append(r[2])
        pct = hits / max(len(frames), 1) * 100
        amed = int(np.mean(areas)) if areas else 0
        flag = ""
        if mejor is None or pct > mejor[0]:
            mejor = (pct, nombre, lo, hi); flag = "  ←"
        print(f"{nombre:>12} {pct:>13.0f}% {amed:>11} {flag}")

    pct, nombre, lo, hi = mejor
    print("\n" + "═" * 40)
    print(f"MEJOR rango: '{nombre}' → balón en {pct:.0f}% de los frames")
    print("Pega esto en CONFIG (pipeline.py):")
    print(f'    "ball_hsv_lower": np.array({lo.tolist()}, dtype=np.uint8),')
    print(f'    "ball_hsv_upper": np.array({hi.tolist()}, dtype=np.uint8),')

    # Guardar pruebas visuales con el mejor rango
    out = Path("output/calibracion_balon"); out.mkdir(parents=True, exist_ok=True)
    for fidx, fr in frames[:6]:
        fm = field_mask(fr)
        r = encuentra_balon(fr, lo, hi, fm)
        vis = fr.copy()
        if r:
            cv2.circle(vis, (r[0], r[1]), 18, (0, 0, 255), 3)
            cv2.putText(vis, "balon", (r[0] - 20, r[1] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(out / f"frame_{fidx}.jpg"), vis)
    print(f"\n[OK] Pruebas visuales → {out}/  (confirma que el círculo cae en el balón)")
    if pct < 60:
        print("\n[NOTA] Si el % sigue bajo, el balón puede estar muy ocluido o el")
        print("       naranja se confunde con la luz. Mira los frames y, si hace")
        print("       falta, ajusta a mano S/V más bajos en el rango 'muy_amplio'.")


if __name__ == "__main__":
    main()
