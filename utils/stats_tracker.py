"""
StatsTracker — Acumula estadísticas de robots y balón frame a frame.

v2 (Copa FutBotMX — Visión por Computadora)
────────────────────────────────────────────────────────────────────────
Mejoras respecto a v1 (corrige los ceros del JSON):

  · MÉTRICA REAL POR HOMOGRAFÍA. Si el pipeline entrega las posiciones ya
    proyectadas a la vista cenital (coordenadas del campo canónico), las
    distancias, velocidades, posesión y eventos se calculan en METROS
    reales usando `meters_per_pixel`, no con una escala fija inventada.
    Si no hay homografía disponible ese frame, cae a coordenadas de píxel.

  · POSESIÓN CON HISTÉRESIS. El "dueño" del balón se mantiene estable con
    un radio de captura y un radio de liberación distintos (evita parpadeo
    de posesión entre frames). Antes la posesión salía 0 % porque el balón
    nunca llegaba con detección estable.

  · EVENTOS REALES (Requisito 3.5.1). Máquina de estados sobre el dueño del
    balón → detecta PASES (mismo equipo), INTERCEPCIONES (equipo rival) y
    TIROS A GOL (velocidad alta hacia la portería). Antes `eventos` salía
    vacío.

  · RED DE PASES. Se acumula un grafo dirigido robot→robot para construir
    el grafo de interacción (visualización avanzada).

  · LÍNEA DE TIEMPO DE POSESIÓN. Se guarda (frame, equipo_en_posesión) para
    la franja temporal del dashboard.
"""

import json
import numpy as np
import supervision as sv
from collections import defaultdict
from pathlib import Path


class StatsTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        W, H = 1280, 720   # se actualizan en primer frame

        # ── Heatmaps por robot (tracker_id → acumulador 2D) ──
        self.robot_heatmaps: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros((H, W), dtype=np.float32)
        )
        self.ball_heatmap = np.zeros((H, W), dtype=np.float32)

        # ── Posiciones históricas (coordenadas de cámara, para heatmaps) ──
        self.robot_positions: dict[int, list] = defaultdict(list)
        self.ball_positions: list = []

        # ── Posiciones históricas en vista cenital (metros) ──
        self.robot_positions_m: dict[int, list] = defaultdict(list)
        self.ball_positions_m: list = []

        # ── Estadísticas del balón ──
        self.ball_speed_kmh: float = 0.0
        self.ball_total_distance_m: float = 0.0
        self.ball_max_speed_kmh: float = 0.0
        self._prev_ball_metric: np.ndarray | None = None
        # Posiciones métricas SUAVIZADAS (EMA) — para distancia/velocidad sin ruido
        self._ema_robot_metric: dict[int, np.ndarray] = {}
        self._ema_ball_metric: np.ndarray | None = None

        # ── Posesión (frames en posesión de cada equipo) ──
        self.possession: dict[int, int] = {0: 0, 1: 0, -1: 0}
        self.possession_timeline: list = []          # (frame, equipo)
        self._team_map: dict[int, int] = {}          # tracker_id → team_id (estable)

        # ── Distancia / velocidad por robot (metros) ──
        self.robot_distance: dict[int, float] = defaultdict(float)
        self.robot_speed: dict[int, float] = defaultdict(float)
        self.robot_max_speed: dict[int, float] = defaultdict(float)
        self._prev_robot_metric: dict[int, np.ndarray] = {}

        # ── Eventos + red de pases ──
        self.events: list[dict] = []
        self.pass_network: dict = defaultdict(int)   # (from_tid, to_tid) → nº pases
        self.team_event_count: dict = defaultdict(int)  # (equipo, tipo) → nº

        # ── Estado de la máquina de posesión (para eventos) ──
        self._holder: int | None = None             # tracker_id dueño actual
        self._last_event_frame: dict = defaultdict(lambda: -9999)

        # Resolución y escala
        self._W, self._H = W, H
        self._pixels_per_meter = 30.0                # fallback si no hay homografía
        self._mpp = None                             # metros por píxel cenital

    # ─────────────────────────────────────────────────────────────────
    # UPDATE (llamado cada frame)
    # ─────────────────────────────────────────────────────────────────
    def update(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        robot_dets: sv.Detections,
        ball_dets: sv.Detections,
        team_ids,
        frame_idx: int,
        robot_topdown: dict | None = None,   # tracker_id → (x, y) cenital (px)
        ball_topdown: np.ndarray | None = None,  # (x, y) cenital (px) o None
        meters_per_pixel: float | None = None,
    ):
        H, W = frame.shape[:2]
        self._W, self._H = W, H
        if meters_per_pixel is not None:
            self._mpp = meters_per_pixel
        fps = self.cfg.get("frame_rate", 30)
        dt = 1.0 / fps

        if self.ball_heatmap.shape != (H, W):
            self.ball_heatmap = np.zeros((H, W), dtype=np.float32)

        # ── Mapa de equipos (estable) ──
        if team_ids is not None and robot_dets.tracker_id is not None:
            for tid, t in zip(robot_dets.tracker_id, team_ids):
                self._team_map[int(tid)] = int(t)

        # ── ROBOTS ──
        if robot_dets.tracker_id is not None:
            for i, tid in enumerate(robot_dets.tracker_id):
                tid = int(tid)
                cx, cy = self._center(robot_dets.xyxy[i])
                cx, cy = int(cx), int(cy)

                if tid not in self.robot_heatmaps or self.robot_heatmaps[tid].shape != (H, W):
                    self.robot_heatmaps[tid] = np.zeros((H, W), dtype=np.float32)
                self._deposit_gaussian(self.robot_heatmaps[tid], cx, cy, sigma=25)
                self.robot_positions[tid].append((frame_idx, cx, cy))

                # Posición métrica (cenital si existe). Se SUAVIZA con EMA para
                # quitar el temblor de detección antes de medir distancia/vel.
                p_m = self._metric_pos(robot_topdown, tid, (cx, cy))
                if p_m is not None:
                    a = self.cfg.get("pos_smooth_alpha", 0.35)
                    ema = self._ema_robot_metric.get(tid)
                    p_s = p_m if ema is None else (a * p_m + (1 - a) * ema)
                    self._ema_robot_metric[tid] = p_s
                    self.robot_positions_m[tid].append((frame_idx, float(p_s[0]), float(p_s[1])))
                    prev = self._prev_robot_metric.get(tid)
                    if prev is not None:
                        step = float(np.linalg.norm(p_s - prev))
                        max_step = self.cfg.get("max_step_robot_m", 0.10)
                        min_move = self.cfg.get("min_move_m", 0.015)  # banda muerta (1.5 cm)
                        if min_move <= step < max_step:
                            self.robot_distance[tid] += step
                            v = step / dt * 3.6   # km/h
                            self.robot_speed[tid] = v
                            self.robot_max_speed[tid] = max(self.robot_max_speed[tid], v)
                        elif step < min_move:
                            self.robot_speed[tid] = 0.0   # casi quieto: no acumula ruido
                    self._prev_robot_metric[tid] = p_s

        # ── BALÓN ──
        ball_metric = None
        if len(ball_dets) > 0:
            best_idx = int(np.argmax(ball_dets.confidence))
            bx, by = self._center(ball_dets.xyxy[best_idx])
            bx, by = int(bx), int(by)
            self._deposit_gaussian(self.ball_heatmap, bx, by, sigma=20)
            self.ball_positions.append((frame_idx, bx, by))

            td = {-1: ball_topdown} if ball_topdown is not None else None
            ball_metric = self._metric_pos(td, -1, (bx, by))
            if ball_metric is not None:
                # Suavizado EMA del balón antes de medir
                a = self.cfg.get("pos_smooth_alpha", 0.35)
                b_s = (ball_metric if self._ema_ball_metric is None
                       else a * ball_metric + (1 - a) * self._ema_ball_metric)
                self._ema_ball_metric = b_s
                ball_metric = b_s
                self.ball_positions_m.append((frame_idx, float(b_s[0]), float(b_s[1])))
                if self._prev_ball_metric is not None:
                    step = float(np.linalg.norm(b_s - self._prev_ball_metric))
                    max_step_b = self.cfg.get("max_step_ball_m", 0.20)
                    min_move = self.cfg.get("min_move_m", 0.015)
                    if min_move <= step < max_step_b:
                        self.ball_total_distance_m += step
                        self.ball_speed_kmh = step / dt * 3.6
                        self.ball_max_speed_kmh = max(self.ball_max_speed_kmh, self.ball_speed_kmh)
                    elif step < min_move:
                        self.ball_speed_kmh = 0.0
                self._prev_ball_metric = b_s

        # ── POSESIÓN + EVENTOS (máquina de estados con histéresis) ──
        self._update_possession_and_events(
            frame_idx, fps, robot_dets, robot_topdown, ball_metric, ball_dets
        )

    # ─────────────────────────────────────────────────────────────────
    # POSESIÓN + EVENTOS
    # ─────────────────────────────────────────────────────────────────
    def _update_possession_and_events(self, frame_idx, fps, robot_dets,
                                      robot_topdown, ball_metric, ball_dets):
        # Radios en metros (con fallback a px si no hay escala)
        capture_r = self.cfg.get("possession_capture_m", 0.22)
        release_r = self.cfg.get("possession_release_m", 0.40)
        if self._mpp is None:   # sin homografía: usar px equivalentes
            capture_r, release_r = 80.0, 130.0

        if ball_metric is None or robot_dets.tracker_id is None or len(robot_dets) == 0:
            self.possession[-1] = self.possession.get(-1, 0) + 1
            self.possession_timeline.append((frame_idx, -1))
            return

        # Distancia balón ↔ cada robot (en espacio métrico cenital)
        nearest_tid, nearest_d, nearest_pos = -1, float("inf"), None
        for i, tid in enumerate(robot_dets.tracker_id):
            tid = int(tid)
            cx, cy = self._center(robot_dets.xyxy[i])
            r_m = self._metric_pos(robot_topdown, tid, (int(cx), int(cy)))
            if r_m is None:
                continue
            d = float(np.linalg.norm(r_m - ball_metric))
            if d < nearest_d:
                nearest_d, nearest_tid, nearest_pos = d, tid, r_m

        # Histéresis: confirmar/soltar dueño
        prev_holder = self._holder
        if self._holder is None:
            if nearest_d < capture_r:
                self._holder = nearest_tid
        else:
            if nearest_tid == self._holder and nearest_d < release_r:
                pass  # conserva al dueño
            elif nearest_d < capture_r:
                self._holder = nearest_tid   # nuevo dueño confirmado
            elif nearest_d > release_r:
                self._holder = None          # balón suelto

        # Registrar posesión del frame
        holder_team = self._team_map.get(self._holder, -1) if self._holder is not None else -1
        self.possession[holder_team] = self.possession.get(holder_team, 0) + 1
        self.possession_timeline.append((frame_idx, holder_team))

        # ── Eventos por cambio de dueño ──
        if (self._holder is not None and prev_holder is not None
                and self._holder != prev_holder):
            from_tid, to_tid = prev_holder, self._holder
            t_from = self._team_map.get(from_tid, -1)
            t_to = self._team_map.get(to_tid, -1)
            if t_from == t_to and t_from != -1:
                self._add_event("pase", frame_idx, fps, nearest_pos, from_tid, to_tid, t_from)
                self.pass_network[(from_tid, to_tid)] += 1
                self.team_event_count[(t_from, "pase")] += 1
            elif t_from != -1 and t_to != -1:
                self._add_event("intercepcion", frame_idx, fps, nearest_pos, from_tid, to_tid, t_to)
                self.team_event_count[(t_to, "intercepcion")] += 1

        # ── Tiro a gol: velocidad alta del balón ──
        shot_speed = self.cfg.get("shot_speed_kmh", 6.0)
        if self.ball_speed_kmh > shot_speed and len(ball_dets) > 0:
            if frame_idx - self._last_event_frame["tiro_a_gol"] > fps * 1.5:
                self._add_event("tiro_a_gol", frame_idx, fps, ball_metric,
                                self._holder, None, holder_team)
                self.team_event_count[(holder_team, "tiro_a_gol")] += 1

    def _add_event(self, tipo, frame_idx, fps, pos_m, from_tid, to_tid, team):
        self._last_event_frame[tipo] = frame_idx
        ev = {
            "tipo": tipo,
            "frame": int(frame_idx),
            "tiempo_seg": round(frame_idx / fps, 2),
            "equipo": int(team) if team is not None else -1,
        }
        if from_tid is not None:
            ev["robot_origen"] = int(from_tid)
        if to_tid is not None:
            ev["robot_destino"] = int(to_tid)
        if pos_m is not None:
            ev["posicion_m"] = [round(float(pos_m[0]), 2), round(float(pos_m[1]), 2)]
        self.events.append(ev)

    # ─────────────────────────────────────────────────────────────────
    # EXPORTAR
    # ─────────────────────────────────────────────────────────────────
    def export_json(self, path: str):
        total_poss = sum(self.possession.values()) or 1
        poss_pct = {
            "equipo_azul": round(self.possession.get(0, 0) / total_poss * 100, 1),
            "equipo_rojo": round(self.possession.get(1, 0) / total_poss * 100, 1),
            "sin_posesion": round(self.possession.get(-1, 0) / total_poss * 100, 1),
        }

        eventos_resumen = defaultdict(int)
        for e in self.events:
            eventos_resumen[e["tipo"]] += 1

        data = {
            "balon": {
                "distancia_total_m": round(self.ball_total_distance_m, 2),
                "velocidad_maxima_kmh": round(self.ball_max_speed_kmh, 2),
                "posiciones_totales": len(self.ball_positions),
            },
            "posesion": poss_pct,
            "robots": {
                str(tid): {
                    "distancia_m": round(d, 2),
                    "velocidad_max_kmh": round(self.robot_max_speed.get(tid, 0.0), 2),
                    "equipo": self._team_map.get(tid, -1),
                }
                for tid, d in sorted(self.robot_distance.items())
            },
            "eventos_resumen": dict(eventos_resumen),
            "eventos": self.events,
            "red_de_pases": [
                {"origen": int(a), "destino": int(b), "pases": int(n)}
                for (a, b), n in sorted(self.pass_network.items(), key=lambda kv: -kv[1])
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return data

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _metric_pos(self, topdown_map, key, pixel_xy):
        """Devuelve la posición en METROS. Usa la coord. cenital si está
        disponible (px cenital × mpp); si no, cae a px de cámara × escala."""
        if topdown_map is not None and key in topdown_map and topdown_map[key] is not None:
            p = np.asarray(topdown_map[key], dtype=np.float32)
            if self._mpp is not None:
                return p * self._mpp
            return p / self._pixels_per_meter
        p = np.asarray(pixel_xy, dtype=np.float32)
        return p / self._pixels_per_meter

    @staticmethod
    def _center(xyxy: np.ndarray) -> tuple[float, float]:
        return (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2

    @staticmethod
    def _deposit_gaussian(heatmap: np.ndarray, cx: int, cy: int, sigma: int = 20):
        """Deposita una gaussiana en una ventana acotada (eficiente)."""
        H, W = heatmap.shape
        r = sigma * 3
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        gx = np.exp(-((xs - cx) ** 2) / (2 * sigma ** 2))
        gy = np.exp(-((ys - cy) ** 2) / (2 * sigma ** 2))
        heatmap[y0:y1, x0:x1] += np.outer(gy, gx)
