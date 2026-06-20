"""
RosterManager + BallCoaster — Reglas del dominio FutBotMX
────────────────────────────────────────────────────────────────────────
Reglas físicas del partido (las impone el usuario, no el detector):

  · COMO MÁXIMO HAY 4 ROBOTS: exactamente 2 por equipo.
  · A veces el detector "ve" más de 4 (reflejos, público, una sombra): NO es
    un robot nuevo, es un falso positivo → se descarta.
  · Hay OCLUSIONES frecuentes entre robots y de la pelota: cuando un robot se
    tapa unos frames, NO debe aparecer con un ID nuevo; su identidad se
    conserva (el "slot" sobrevive y se re-asocia por cercanía al reaparecer).

`RosterManager` mapea los IDs crudos de ByteTrack a un ROSTER FIJO de 4
ranuras estables (Azul-1, Azul-2, Rojo-1, Rojo-2). Cada frame:
  1. asigna cada detección de robot a la ranura más cercana de su equipo,
  2. crea ranura sólo si el equipo tiene menos de 2,
  3. descarta cualquier detección sobrante (regla "máx. 2 por equipo"),
  4. conserva las ranuras ocluidas durante `coast_frames` para no perder la
     identidad.

`BallCoaster` mantiene la última posición conocida de la pelota durante una
oclusión corta, para que posesión y eventos no se rompan.
"""

import numpy as np
import supervision as sv


def _centers(xyxy: np.ndarray) -> np.ndarray:
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2,
                     (xyxy[:, 1] + xyxy[:, 3]) / 2], axis=1)


class RosterManager:
    def __init__(self, cfg: dict):
        self.max_team   = cfg.get("max_robots_per_team", 2)
        self.gate       = cfg.get("roster_gate_px", 220)
        self.coast      = cfg.get("roster_coast_frames", 30)
        # POOL FIJO de IDs: máximo 4 robots en todo el partido.
        #   Equipo 0 (azul) → ids 1, 2     Equipo 1 (rojo) → ids 3, 4
        # Cuando un slot se libera (oclusión larga), su id se RECICLA, así
        # nunca aparecen ids 5, 6… → como mucho 4 mapas de calor.
        self.team_ids = {0: [1, 2], 1: [3, 4]}
        self.all_ids = [1, 2, 3, 4]
        # slot_id → dict(team, pos[np2], last_seen, conf)
        self.slots: dict[int, dict] = {}
        self.team_history: dict[int, int] = {}   # slot_id → equipo (permanente)

    def get_team_map(self) -> dict:
        """slot_id (1..4) → equipo estable. Persistente (no expira)."""
        return dict(self.team_history)

    def _free_id(self, team: int):
        """Devuelve un id libre del pool, prefiriendo el par del equipo.
        None si los 4 ids ya están ocupados (no puede haber un 5º robot)."""
        active = set(self.slots.keys())
        for i in self.team_ids.get(team, []):      # par designado del equipo
            if i not in active:
                return i
        for i in self.all_ids:                     # cualquier id libre
            if i not in active:
                return i
        return None

    # ─────────────────────────────────────────────────────────────────
    def update(self, robot_dets: sv.Detections, team_ids, frame_idx: int):
        """Devuelve (robot_dets_filtrado, team_ids_alineado). El tracker_id
        del resultado son los IDs estables del roster (1..4)."""
        n = len(robot_dets)
        if n == 0:
            self._expire(frame_idx)
            return robot_dets, (np.array([], dtype=int))

        centers = _centers(robot_dets.xyxy)
        confs = robot_dets.confidence if robot_dets.confidence is not None \
            else np.ones(n)
        teams = np.array(team_ids) if team_ids is not None else np.full(n, -1)

        assigned_sid = [None] * n   # slot asignado a cada detección (o None=descartar)

        # Procesar por equipo (si se conocen). Si todos son -1, un solo grupo.
        if np.all(teams < 0):
            groups = {-1: list(range(n))}
        else:
            groups = {}
            for i in range(n):
                groups.setdefault(int(teams[i]), []).append(i)

        for team, idxs in groups.items():
            # Orden por confianza (los más fiables tienen prioridad de slot)
            idxs = sorted(idxs, key=lambda i: -confs[i])
            cap = self.max_team if team >= 0 else 2 * self.max_team
            slot_ids = [sid for sid, s in self.slots.items()
                        if (s["team"] == team or team < 0)]

            # 1) Emparejar candidatos con ranuras existentes (greedy por dist.)
            pairs = []
            for i in idxs:
                for sid in slot_ids:
                    d = float(np.linalg.norm(centers[i] - self.slots[sid]["pos"]))
                    pairs.append((d, i, sid))
            pairs.sort(key=lambda p: p[0])
            used_i, used_s = set(), set()
            for d, i, sid in pairs:
                if i in used_i or sid in used_s or d > self.gate:
                    continue
                used_i.add(i); used_s.add(sid); assigned_sid[i] = sid
                self.slots[sid].update(team=(int(teams[i]) if teams[i] >= 0
                                             else self.slots[sid]["team"]),
                                       pos=centers[i], last_seen=frame_idx,
                                       conf=float(confs[i]))

            # 2) Candidatos no emparejados: crear ranura si hay cupo, si no, descartar
            active = [sid for sid, s in self.slots.items()
                      if (s["team"] == team or team < 0)]
            for i in idxs:
                if assigned_sid[i] is not None:
                    continue
                if len(active) < cap:
                    sid = self._free_id(int(teams[i]))
                    if sid is None:
                        continue   # pool de 4 lleno → se descarta (no hay 5º robot)
                    self.slots[sid] = dict(team=int(teams[i]), pos=centers[i],
                                           last_seen=frame_idx, conf=float(confs[i]))
                    assigned_sid[i] = sid
                    active.append(sid)
                # else: sobra → regla "máx. 2 por equipo" → se descarta

        self._expire(frame_idx)

        # Registrar equipo permanente de cada slot conocido (para reconciliar)
        for sid, s in self.slots.items():
            if s["team"] >= 0:
                self.team_history[sid] = s["team"]

        # Construir detecciones filtradas con IDs estables
        keep = [i for i in range(n) if assigned_sid[i] is not None]
        if not keep:
            return robot_dets[[]], np.array([], dtype=int)
        out = robot_dets[keep]
        out.tracker_id = np.array([assigned_sid[i] for i in keep], dtype=int)
        out_teams = np.array([self.slots[assigned_sid[i]]["team"] for i in keep],
                             dtype=int)
        return out, out_teams

    def _expire(self, frame_idx: int):
        """Libera ranuras ocluidas demasiado tiempo (más que coast_frames)."""
        dead = [sid for sid, s in self.slots.items()
                if frame_idx - s["last_seen"] > self.coast]
        for sid in dead:
            del self.slots[sid]


class BallCoaster:
    """Sostiene la última posición de la pelota durante oclusiones cortas."""

    def __init__(self, cfg: dict):
        self.coast = cfg.get("ball_coast_frames", 10)
        self._last_xyxy = None
        self._last_frame = -9999

    def update(self, ball_dets: sv.Detections, frame_idx: int) -> sv.Detections:
        if len(ball_dets) > 0:
            best = int(np.argmax(ball_dets.confidence))
            self._last_xyxy = ball_dets.xyxy[best].copy()
            self._last_frame = frame_idx
            return ball_dets
        # Oclusión: reusar la última caja conocida si es reciente
        if (self._last_xyxy is not None
                and frame_idx - self._last_frame <= self.coast):
            return sv.Detections(
                xyxy=self._last_xyxy.reshape(1, 4).astype(np.float32),
                confidence=np.array([0.20], dtype=np.float32),  # posición predicha
                class_id=np.array([self._ball_class], dtype=int),
            )
        return ball_dets  # vacío (oclusión larga)

    _ball_class = 2

    def set_ball_class(self, c: int):
        self._ball_class = c
