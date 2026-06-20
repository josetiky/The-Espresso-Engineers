"""
FormationTeamAssigner — Asigna equipo por la FORMACIÓN INICIAL (kickoff).
────────────────────────────────────────────────────────────────────────
Idea (del usuario): al inicio del partido los robots se forman por equipos —
un equipo de un lado de la cancha y el otro del lado opuesto. Ese arreglo
espacial es una señal de equipo MUCHO más confiable que el color del robot
(que en estos robots negros casi no distingue equipos).

Cómo funciona:
  1. Durante los primeros `formation_frames` frames (la "ventana de saque"),
     se acumula la posición CENITAL (vista de pájaro) de cada robot, indexada
     por su ID estable del roster.
  2. Al cerrar la ventana, se separa a los robots en 2 grupos por el lado del
     campo: se proyecta su posición media sobre el eje de mayor separación y
     se corta por el HUECO más grande (sin parámetros, robusto a conteos
     desiguales). Un lado = equipo 0, el otro = equipo 1.
  3. Esa etiqueta queda BLOQUEADA por ID: aunque luego los robots se crucen o
     se mezclen, cada ID conserva su equipo del saque.

Limitación (que el propio usuario notó): si el video NO empieza desde el
saque, la formación inicial no es fiable. Por eso esto se usa en modo "auto":
si la ventana de saque da una separación clara se usa; si no, se cae al color.
"""

import numpy as np
from collections import defaultdict


class FormationTeamAssigner:
    def __init__(self, cfg: dict):
        self.window = cfg.get("formation_frames", 45)     # frames de "saque" a observar
        self.axis_cfg = cfg.get("formation_axis", "auto")  # 'x' | 'y' | 'auto'
        self.min_sep = cfg.get("formation_min_sep_px", 60)  # separación mínima válida
        self.pos_hist: dict[int, list] = defaultdict(list)
        self.locked: dict[int, int] = {}
        self._frame0: int | None = None
        self._axis: int | None = None
        self._thr: float | None = None
        self.ok: bool = False        # True si la formación dio una separación clara

    # ─────────────────────────────────────────────────────────────────
    def update(self, robot_topdown: dict, frame_idx: int):
        """robot_topdown: {id_estable: (x, y)} en coordenadas cenitales (px)."""
        if not robot_topdown:
            return
        if self._frame0 is None:
            self._frame0 = frame_idx

        if frame_idx - self._frame0 < self.window:
            for tid, (x, y) in robot_topdown.items():
                self.pos_hist[int(tid)].append((float(x), float(y)))
        elif not self.locked and self.pos_hist:
            self._lock()

    # ─────────────────────────────────────────────────────────────────
    def _lock(self):
        ids = list(self.pos_hist.keys())
        if len(ids) < 2:
            return
        means = {t: np.mean(self.pos_hist[t], axis=0) for t in ids}
        M = np.array([means[t] for t in ids])         # (N, 2)

        # Eje de separación: el de mayor dispersión entre medias
        if self.axis_cfg == "auto":
            ax = 0 if (M[:, 0].max() - M[:, 0].min()) >= (M[:, 1].max() - M[:, 1].min()) else 1
        else:
            ax = 0 if self.axis_cfg == "x" else 1

        coords = M[:, ax]
        order = np.argsort(coords)
        sorted_c = coords[order]
        gaps = np.diff(sorted_c)
        if len(gaps) == 0:
            return
        split = int(np.argmax(gaps))                  # corte por el hueco más grande
        sep = float(gaps[split])
        thr = (sorted_c[split] + sorted_c[split + 1]) / 2.0

        # Validación: la separación entre los dos bandos debe ser apreciable
        if sep < self.min_sep:
            self.ok = False
            return

        self._axis, self._thr, self.ok = ax, thr, True
        for t in ids:
            self.locked[t] = 0 if means[t][ax] < thr else 1

    # ─────────────────────────────────────────────────────────────────
    def team_of(self, tid: int, current_pos=None) -> int:
        """Equipo del robot: bloqueado si ya se fijó; si no, por su lado actual
        (cuando ya hay umbral). Devuelve -1 si aún no se puede decidir."""
        tid = int(tid)
        if tid in self.locked:
            return self.locked[tid]
        if self._thr is not None and current_pos is not None:
            team = 0 if current_pos[self._axis] < self._thr else 1
            self.locked[tid] = team        # lo fija al primer avistamiento post-saque
            return team
        return -1

    def get_team_map(self) -> dict:
        return dict(self.locked)
