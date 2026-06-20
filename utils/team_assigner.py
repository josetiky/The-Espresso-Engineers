"""
TeamAssigner — Asigna robots a equipos mediante k-means en el color
de la región del "jersey" (marcador de color del robot).

v2 — Añade VOTO TEMPORAL por tracker_id.
────────────────────────────────────────────────────────────────────────
Problema en v1: el color de un robot puede leerse mal en algunos frames
(oclusión, brillo, sombra) → el equipo "parpadeaba" y robots vistos antes
de inicializar k-means quedaban sin equipo (-1).

Solución: cada vez que vemos un robot, votamos su equipo según el color de
ESE frame, pero el equipo REPORTADO es el que acumula más votos a lo largo
del video (moda temporal). Así cada tracker_id obtiene una etiqueta de
equipo estable y nunca queda en -1 una vez que se inicializó k-means.
"""

import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import supervision as sv


class TeamAssigner:
    def __init__(self, n_teams: int = 2, sample_size: int = 200):
        self.n_teams = n_teams
        self.sample_size = sample_size
        self.kmeans: KMeans | None = None
        self.team_colors: dict[int, np.ndarray] = {}
        # Voto temporal: tracker_id → [votos_equipo0, votos_equipo1]
        self._votes: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_teams, dtype=np.int64)
        )

    # ─────────────────────────────────────────
    # INICIALIZACIÓN (primer frame con robots)
    # ─────────────────────────────────────────
    def initialize(self, frame: np.ndarray, robot_dets: sv.Detections):
        samples = []
        for bbox in robot_dets.xyxy:
            patch = self._get_jersey_patch(frame, bbox)
            if patch is not None:
                samples.append(patch)
        if len(samples) < self.n_teams:
            return
        all_pixels = np.vstack(samples)
        self.kmeans = KMeans(n_clusters=self.n_teams, n_init=10, random_state=42)
        self.kmeans.fit(all_pixels)
        for t in range(self.n_teams):
            mask = self.kmeans.labels_ == t
            self.team_colors[t] = all_pixels[mask].mean(axis=0)

    # ─────────────────────────────────────────
    # ASIGNACIÓN FRAME A FRAME (con voto temporal)
    # ─────────────────────────────────────────
    def assign(self, frame: np.ndarray, robot_dets: sv.Detections) -> np.ndarray:
        """Devuelve un array de team_id estable (moda temporal) para cada
        robot en robot_dets. Requiere robot_dets.tracker_id."""
        if self.kmeans is None:
            return np.zeros(len(robot_dets), dtype=int)

        tracker_ids = robot_dets.tracker_id
        team_ids = []
        for i, bbox in enumerate(robot_dets.xyxy):
            # Voto instantáneo según color del frame
            patch = self._get_jersey_patch(frame, bbox)
            if patch is None or len(patch) == 0:
                inst = 0
            else:
                color = patch.mean(axis=0).reshape(1, -1)
                inst = int(self.kmeans.predict(color)[0])

            if tracker_ids is not None:
                tid = int(tracker_ids[i])
                self._votes[tid][inst] += 1
                team_ids.append(int(np.argmax(self._votes[tid])))  # moda temporal
            else:
                team_ids.append(inst)
        return np.array(team_ids, dtype=int)

    def get_team_map(self) -> dict[int, int]:
        """tracker_id → equipo estable (según los votos acumulados)."""
        return {tid: int(np.argmax(v)) for tid, v in self._votes.items() if v.sum() > 0}

    # ─────────────────────────────────────────
    # HELPER: extraer parche del "jersey"
    # ─────────────────────────────────────────
    def _get_jersey_patch(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        crop_y2 = y1 + max(h // 3, 5)
        patch = frame[y1:crop_y2, x1:x2]
        if patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        if len(pixels) > self.sample_size:
            idx = np.random.choice(len(pixels), self.sample_size, replace=False)
            pixels = pixels[idx]
        return pixels
