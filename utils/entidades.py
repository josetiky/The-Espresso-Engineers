"""
Modelo de dominio orientado a objetos — Copa FutBotMX
────────────────────────────────────────────────────────────────────────
Jerarquía de clases (POO con herencia + composición):

    EntidadCancha            ← superclase: cualquier objeto rastreado
    ├── Robot                ← hereda de EntidadCancha (un robot ES una entidad)
    └── Balon                ← hereda de EntidadCancha (la pelota ES una entidad)

    Equipo                   ← COMPOSICIÓN: un equipo CONTIENE robots
                               (un equipo NO es un robot, por eso no hereda)

Todas las posiciones se manejan en METROS (vista cenital). La velocidad se
deriva con los fps. Este módulo es independiente del pipeline: se puede usar
para estructurar las estadísticas, en pruebas, o en notebooks de análisis.
"""

from __future__ import annotations
import numpy as np


# ════════════════════════════════════════════════════════════════════════
# SUPERCLASE
# ════════════════════════════════════════════════════════════════════════
class EntidadCancha:
    """Objeto rastreado en la cancha: guarda su trayectoria y deriva
    distancia y velocidad. Base de Robot y Balon."""

    # salto físico máximo por frame (m) — filtra ruido de homografía
    SALTO_MAX_M = 0.6

    def __init__(self, id_: int, fps: int = 30):
        self.id = int(id_)
        self.fps = fps
        self.trayectoria: list[tuple[int, float, float]] = []  # (frame, x_m, y_m)
        self.distancia_m: float = 0.0
        self.velocidad_kmh: float = 0.0
        self.velocidad_max_kmh: float = 0.0
        self._prev: np.ndarray | None = None

    # ── Actualización frame a frame ──────────────────────────────────
    def actualizar(self, x_m: float, y_m: float, frame: int):
        """Registra una nueva posición (en metros) y actualiza métricas."""
        p = np.array([x_m, y_m], dtype=float)
        self.trayectoria.append((int(frame), float(x_m), float(y_m)))
        if self._prev is not None:
            paso = float(np.linalg.norm(p - self._prev))
            if paso < self.SALTO_MAX_M:          # filtro anti-salto
                self.distancia_m += paso
                self.velocidad_kmh = paso * self.fps * 3.6
                self.velocidad_max_kmh = max(self.velocidad_max_kmh,
                                             self.velocidad_kmh)
        self._prev = p

    # ── Propiedades de solo lectura ──────────────────────────────────
    @property
    def posicion_actual(self) -> np.ndarray | None:
        return self._prev.copy() if self._prev is not None else None

    @property
    def vista(self) -> int:
        """Número de frames en que se vio la entidad."""
        return len(self.trayectoria)

    def resumen(self) -> dict:
        return {
            "id": self.id,
            "distancia_m": round(self.distancia_m, 2),
            "velocidad_max_kmh": round(self.velocidad_max_kmh, 2),
            "frames_visto": self.vista,
        }

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.id} d={self.distancia_m:.1f}m>"


# ════════════════════════════════════════════════════════════════════════
# CLASES HIJAS (heredan de EntidadCancha)
# ════════════════════════════════════════════════════════════════════════
class Robot(EntidadCancha):
    """Un robot del partido. ES UNA EntidadCancha + pertenece a un equipo."""

    def __init__(self, id_: int, equipo: int = -1, fps: int = 30):
        super().__init__(id_, fps)          # reutiliza la lógica de la base
        self.equipo = equipo                 # 0=azul, 1=rojo, -1=desconocido

    @property
    def nombre_equipo(self) -> str:
        return {0: "Azul", 1: "Rojo"}.get(self.equipo, "?")

    def resumen(self) -> dict:
        base = super().resumen()             # extiende el resumen de la base
        base["equipo"] = self.equipo
        base["nombre_equipo"] = self.nombre_equipo
        return base

    def __repr__(self):
        return f"<Robot #{self.id} {self.nombre_equipo} d={self.distancia_m:.1f}m>"


class Balon(EntidadCancha):
    """La pelota. ES UNA EntidadCancha + recuerda quién la tiene."""

    def __init__(self, id_: int = 0, fps: int = 30):
        super().__init__(id_, fps)
        self.dueno: int | None = None        # tracker_id del robot en posesión

    def resumen(self) -> dict:
        base = super().resumen()
        base["dueno_actual"] = self.dueno
        return base


# ════════════════════════════════════════════════════════════════════════
# COMPOSICIÓN: un Equipo CONTIENE robots (no hereda de Robot)
# ════════════════════════════════════════════════════════════════════════
class Equipo:
    """Agrupa los robots de un mismo color y calcula métricas de equipo."""

    NOMBRES = {0: "Equipo Azul", 1: "Equipo Rojo"}

    def __init__(self, id_equipo: int):
        self.id = id_equipo
        self.nombre = self.NOMBRES.get(id_equipo, f"Equipo {id_equipo}")
        self.robots: dict[int, Robot] = {}   # composición: tiene-un(os) Robot(s)
        self.frames_posesion: int = 0

    def agregar_robot(self, robot: Robot):
        robot.equipo = self.id
        self.robots[robot.id] = robot

    def obtener_robot(self, id_: int, fps: int = 30) -> Robot:
        """Devuelve el robot; lo crea y lo agrega si no existía."""
        if id_ not in self.robots:
            self.agregar_robot(Robot(id_, equipo=self.id, fps=fps))
        return self.robots[id_]

    # ── Métricas agregadas del equipo ────────────────────────────────
    @property
    def distancia_total_m(self) -> float:
        return round(sum(r.distancia_m for r in self.robots.values()), 2)

    @property
    def velocidad_max_kmh(self) -> float:
        return round(max((r.velocidad_max_kmh for r in self.robots.values()),
                         default=0.0), 2)

    @property
    def n_robots(self) -> int:
        return len(self.robots)

    def resumen(self) -> dict:
        return {
            "equipo": self.nombre,
            "n_robots": self.n_robots,
            "distancia_total_m": self.distancia_total_m,
            "velocidad_max_kmh": self.velocidad_max_kmh,
            "robots": [r.resumen() for r in self.robots.values()],
        }

    def __repr__(self):
        return f"<{self.nombre}: {self.n_robots} robots, {self.distancia_total_m} m>"
