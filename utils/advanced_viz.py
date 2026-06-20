"""
Visualizaciones avanzadas — Copa FutBotMX
────────────────────────────────────────────────────────────────────────
Sobre los resultados del pipeline (Requisito 3.5.2), genera:

  1. VORONOI DE CONTROL DE CANCHA  (control espacial por equipo)
     Cada punto del campo se "asigna" al robot más cercano; el área que
     domina cada equipo se calcula como fracción de celdas de Voronoi.

  2. LÍNEA DE TIEMPO DE POSESIÓN   (franja temporal azul/rojo/neutro)

  3. GRAFO DE INTERACCIÓN (RED DE PASES)  robot → robot

  4. MAPA DE TRAYECTORIAS DEL BALÓN en vista cenital (metros)

Todo se calcula a partir de los atributos que acumula StatsTracker, sin
volver a procesar el video.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

TEAM_COLORS = ["#1E78DC", "#DC3232"]      # Azul, Rojo
TEAM_NAMES  = ["Equipo Azul", "Equipo Rojo"]
NEUTRAL     = "#9aa0a6"


# ─────────────────────────────────────────────────────────────────────────
# 1. VORONOI DE CONTROL DE CANCHA
# ─────────────────────────────────────────────────────────────────────────
def voronoi_control(stats, cfg, path, grid=220):
    """Reconstruye un fotograma representativo (posición media de cada robot
    en metros) y pinta el diagrama de Voronoi por equipo: el color de cada
    celda es el del equipo del robot más cercano. Reporta el % de control."""
    fw, fh = cfg.get("field_real_m", (1.82, 2.43))

    # Posición media de cada robot (en metros, vista cenital)
    pts, teams = [], []
    for tid, poss in stats.robot_positions_m.items():
        if not poss:
            continue
        arr = np.array([[p[1], p[2]] for p in poss], dtype=np.float32)
        pts.append(arr.mean(axis=0))
        teams.append(stats._team_map.get(int(tid), -1))
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    teams = np.array(teams)

    # Malla del campo
    xs = np.linspace(0, fw, grid)
    ys = np.linspace(0, fh, grid)
    gx, gy = np.meshgrid(xs, ys)
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)          # (grid², 2)

    # Robot más cercano por celda → su equipo
    d = np.linalg.norm(cells[:, None, :] - pts[None, :, :], axis=2)  # (grid², N)
    nearest = np.argmin(d, axis=1)
    cell_team = teams[nearest].reshape(grid, grid)

    # % de control (solo celdas de equipos válidos)
    valid = cell_team >= 0
    ctrl0 = float(np.mean(cell_team[valid] == 0)) * 100 if valid.any() else 0
    ctrl1 = float(np.mean(cell_team[valid] == 1)) * 100 if valid.any() else 0

    # Render
    rgb = np.zeros((grid, grid, 3))
    rgb[cell_team == 0] = (0.118, 0.471, 0.863)
    rgb[cell_team == 1] = (0.863, 0.196, 0.196)
    rgb[cell_team == -1] = (0.6, 0.63, 0.65)

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(rgb, extent=[0, fw, fh, 0], origin="upper", alpha=0.55)
    for p, t in zip(pts, teams):
        c = TEAM_COLORS[t] if t in (0, 1) else NEUTRAL
        ax.scatter(p[0], p[1], s=180, c=c, edgecolors="white", linewidths=2, zorder=3)
    ax.add_patch(plt.Rectangle((0, 0), fw, fh, fill=False, ec="white", lw=2))
    ax.set_xlim(0, fw); ax.set_ylim(fh, 0)
    ax.set_title(f"Control de cancha (Voronoi)\n"
                 f"Azul {ctrl0:.0f}%  ·  Rojo {ctrl1:.0f}%", fontsize=12, weight="bold")
    ax.set_xlabel("metros"); ax.set_ylabel("metros")
    ax.set_aspect("equal")
    _save(fig, path)
    return {"control_azul_pct": round(ctrl0, 1), "control_rojo_pct": round(ctrl1, 1)}


# ─────────────────────────────────────────────────────────────────────────
# 2. LÍNEA DE TIEMPO DE POSESIÓN
# ─────────────────────────────────────────────────────────────────────────
def possession_timeline(stats, cfg, path):
    tl = stats.possession_timeline
    if not tl:
        return None
    fps = cfg.get("frame_rate", 30)
    frames = np.array([f for f, _ in tl])
    teams  = np.array([t for _, t in tl])
    t_seg = frames / fps

    fig, ax = plt.subplots(figsize=(11, 1.8))
    color_map = {0: TEAM_COLORS[0], 1: TEAM_COLORS[1], -1: NEUTRAL}
    # Pinta franjas contiguas del mismo equipo
    start = 0
    for i in range(1, len(teams) + 1):
        if i == len(teams) or teams[i] != teams[start]:
            ax.axvspan(t_seg[start], t_seg[min(i, len(teams) - 1)],
                       color=color_map.get(int(teams[start]), NEUTRAL))
            start = i
    ax.set_xlim(t_seg[0], t_seg[-1]); ax.set_yticks([])
    ax.set_xlabel("tiempo (s)")
    ax.set_title("Línea de tiempo de posesión  (azul · rojo · neutro)",
                 fontsize=12, weight="bold")
    _save(fig, path)
    return True


# ─────────────────────────────────────────────────────────────────────────
# 3. GRAFO DE INTERACCIÓN (RED DE PASES)
# ─────────────────────────────────────────────────────────────────────────
def passing_network(stats, cfg, path):
    if not stats.pass_network:
        return None
    # Posición media de cada robot (cenital, metros) como nodo
    node_pos, node_team = {}, {}
    for tid, poss in stats.robot_positions_m.items():
        if poss:
            arr = np.array([[p[1], p[2]] for p in poss])
            node_pos[int(tid)] = arr.mean(axis=0)
            node_team[int(tid)] = stats._team_map.get(int(tid), -1)

    fw, fh = cfg.get("field_real_m", (1.82, 2.43))
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.add_patch(plt.Rectangle((0, 0), fw, fh, fill=False, ec="#444", lw=2))

    maxn = max(stats.pass_network.values())
    for (a, b), n in stats.pass_network.items():
        if a in node_pos and b in node_pos:
            pa, pb = node_pos[a], node_pos[b]
            ax.add_patch(FancyArrowPatch(
                (pa[0], pa[1]), (pb[0], pb[1]),
                arrowstyle="-|>", mutation_scale=12,
                lw=1 + 4 * n / maxn, color="#555", alpha=0.7,
                connectionstyle="arc3,rad=0.12", zorder=2))

    # Nodos = nº de pases originados
    origin_count = {}
    for (a, _), n in stats.pass_network.items():
        origin_count[a] = origin_count.get(a, 0) + n
    for tid, p in node_pos.items():
        t = node_team.get(tid, -1)
        c = TEAM_COLORS[t] if t in (0, 1) else NEUTRAL
        size = 200 + 120 * origin_count.get(tid, 0)
        ax.scatter(p[0], p[1], s=size, c=c, edgecolors="white", linewidths=2, zorder=3)
        ax.text(p[0], p[1], str(tid), ha="center", va="center",
                color="white", fontsize=9, weight="bold", zorder=4)

    ax.set_xlim(-0.1, fw + 0.1); ax.set_ylim(fh + 0.1, -0.1)
    ax.set_aspect("equal"); ax.set_xlabel("metros"); ax.set_ylabel("metros")
    ax.set_title("Red de pases (grafo de interacción)\n"
                 "grosor = nº de pases · tamaño = pases originados",
                 fontsize=12, weight="bold")
    _save(fig, path)
    return True


# ─────────────────────────────────────────────────────────────────────────
# 4. TRAYECTORIA DEL BALÓN (cenital, metros)
# ─────────────────────────────────────────────────────────────────────────
def ball_trajectory(stats, cfg, path):
    if not stats.ball_positions_m:
        return None
    fw, fh = cfg.get("field_real_m", (1.82, 2.43))
    arr = np.array([[p[1], p[2]] for p in stats.ball_positions_m])
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.add_patch(plt.Rectangle((0, 0), fw, fh, fill=False, ec="#444", lw=2))
    ax.plot(arr[:, 0], arr[:, 1], "-", color="#FF9500", lw=1.5, alpha=0.6)
    ax.scatter(arr[:, 0], arr[:, 1], s=8, c=np.arange(len(arr)),
               cmap="autumn", zorder=3)
    ax.scatter(arr[0, 0], arr[0, 1], s=120, c="lime", marker="o",
               edgecolors="k", label="inicio", zorder=4)
    ax.scatter(arr[-1, 0], arr[-1, 1], s=120, c="red", marker="X",
               edgecolors="k", label="fin", zorder=4)
    ax.set_xlim(-0.1, fw + 0.1); ax.set_ylim(fh + 0.1, -0.1)
    ax.set_aspect("equal"); ax.legend(loc="upper right")
    ax.set_xlabel("metros"); ax.set_ylabel("metros")
    ax.set_title("Trayectoria del balón (vista cenital)", fontsize=12, weight="bold")
    _save(fig, path)
    return True


# ─────────────────────────────────────────────────────────────────────────
# ORQUESTADOR
# ─────────────────────────────────────────────────────────────────────────
def generate_all_advanced(stats, cfg, out_dir="output/avanzadas"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    results = {}
    results["voronoi"] = voronoi_control(stats, cfg, out / "voronoi_control.png")
    results["timeline"] = possession_timeline(stats, cfg, out / "timeline_posesion.png")
    results["red_pases"] = passing_network(stats, cfg, out / "red_de_pases.png")
    results["balon"] = ball_trajectory(stats, cfg, out / "trayectoria_balon.png")
    return results


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(str(path), dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
