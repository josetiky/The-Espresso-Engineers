"""
Prueba sintética de extremo a extremo (sin video real).
Simula un partido: 2 robots azules, 2 rojos y un balón que se mueve y
cambia de dueño, para verificar que:
  · se calculan distancias/velocidad en metros (no cero),
  · la posesión se reparte (no 0%),
  · se generan eventos (pases / intercepciones / tiros),
  · el JSON, las visualizaciones avanzadas y el dashboard se crean.

Ejecuta:  python tests/test_sintetico.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import supervision as sv
from utils.stats_tracker import StatsTracker
from utils.advanced_viz import generate_all_advanced
from dashboard import build_dashboard

CFG = {
    "frame_rate": 30,
    "field_real_m": (1.82, 2.43),
    "possession_capture_m": 0.22,
    "possession_release_m": 0.40,
    "shot_speed_kmh": 6.0,
}

# Lienzo cenital ficticio 700×440 px (como field_template_size)
CANVAS = (700, 440)
MPP = float(np.hypot(*CFG["field_real_m"]) / np.hypot(*CANVAS))

# Robots: tracker_id → (equipo, posición cenital fija aprox + jitter)
ROBOTS = {
    1: (0, np.array([180, 150])),
    2: (0, np.array([300, 300])),
    3: (1, np.array([480, 180])),
    4: (1, np.array([520, 320])),
}

def robot_dets(frame_idx):
    xyxy, tids, cls, conf = [], [], [], []
    topdown = {}
    for tid, (team, base) in ROBOTS.items():
        jit = 6 * np.sin(frame_idx / 9.0 + tid)
        pos = base + np.array([jit, jit])
        topdown[tid] = (float(pos[0]), float(pos[1]))
        # bbox de cámara ~ misma posición (no se usa para métrica aquí)
        x, y = pos
        xyxy.append([x-15, y-15, x+15, y+15]); tids.append(tid)
        cls.append(1); conf.append(0.9)
    d = sv.Detections(xyxy=np.array(xyxy, np.float32),
                      confidence=np.array(conf, np.float32),
                      class_id=np.array(cls, int))
    d.tracker_id = np.array(tids)
    return d, topdown

def ball_at(frame_idx):
    """El balón viaja: robot1 → robot2 (pase azul) → robot3 (intercepción)."""
    t = frame_idx
    if t < 30:      target = ROBOTS[1][1]
    elif t < 70:    target = ROBOTS[2][1]   # pase dentro de azul
    elif t < 110:   target = ROBOTS[3][1]   # intercepción del rojo
    else:           target = np.array([690, 220])  # disparo al borde
    # interpolación suave hacia el objetivo
    prev = ball_at._prev
    pos = prev + (target - prev) * 0.25
    ball_at._prev = pos
    x, y = pos
    d = sv.Detections(xyxy=np.array([[x-6, y-6, x+6, y+6]], np.float32),
                      confidence=np.array([0.8], np.float32),
                      class_id=np.array([2], int))
    d.tracker_id = np.array([99])   # como tras pasar por ByteTrack en el pipeline
    return d, (float(x), float(y))
ball_at._prev = ROBOTS[1][1].astype(float).copy()

def main():
    stats = StatsTracker(CFG)
    team_ids_map = {1:0,2:0,3:1,4:1}
    H, W = 440, 700
    frame = np.zeros((H, W, 3), np.uint8)

    for f in range(1, 160):
        rd, rtop = robot_dets(f)
        bd, btop = ball_at(f)
        team_ids = np.array([team_ids_map[int(t)] for t in rd.tracker_id])
        all_det = sv.Detections.merge([rd, bd])
        stats.update(frame, all_det, rd, bd, team_ids, f,
                     robot_topdown=rtop, ball_topdown=btop, meters_per_pixel=MPP)

    data = stats.export_json("output/estadisticas_TEST.json")

    print("── RESULTADOS ──")
    print("Posesión:", data["posesion"])
    print("Balón:", data["balon"])
    print("Eventos:", data["eventos_resumen"])
    print("Robots equipos:", {k: v["equipo"] for k, v in data["robots"].items()})
    print("Distancias (m):", {k: v["distancia_m"] for k, v in data["robots"].items()})
    print("Red de pases:", data["red_de_pases"][:5])

    # Aserciones clave (lo que estaba en cero antes)
    assert data["balon"]["posiciones_totales"] > 0, "balón sin posiciones"
    assert sum(data["posesion"].values()) > 0, "posesión vacía"
    assert any(v["equipo"] in (0,1) for v in data["robots"].values()), "equipos -1"
    assert len(data["eventos"]) > 0, "sin eventos"
    assert all(v["distancia_m"] > 0 for v in data["robots"].values()), "distancias en 0"

    generate_all_advanced(stats, CFG, out_dir="output/avanzadas_TEST")
    build_dashboard(data, "output/dashboard_TEST.html")
    print("\n[OK] Todas las aserciones pasaron. Artefactos en output/*_TEST*")

if __name__ == "__main__":
    main()
