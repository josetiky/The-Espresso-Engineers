"""
Visualizaciones — Copa FutBotMX
Incluye: heatmaps por robot, estadísticas del balón, barra de posesión,
         minimapa, imagen de resumen final.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sin pantalla
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES VISUALES
# ─────────────────────────────────────────────────────────────────────────────

TEAM_BGR   = [(220, 120, 30), (50, 50, 220)]    # Azul, Rojo en BGR
TEAM_NAMES = ["Equipo Azul", "Equipo Rojo"]
FONT       = cv2.FONT_HERSHEY_SIMPLEX

# Colormaps personalizados para heatmap (transparente → color de equipo)
_BLUE_CMAP = LinearSegmentedColormap.from_list("blue_heat",  ["#00000000", "#1E78DC"])
_RED_CMAP  = LinearSegmentedColormap.from_list("red_heat",   ["#00000000", "#DC3232"])
_BALL_CMAP = LinearSegmentedColormap.from_list("ball_heat",  ["#00000000", "#FFDC00"])
TEAM_CMAPS = [_BLUE_CMAP, _RED_CMAP]


# ─────────────────────────────────────────────────────────────────────────────
# 1. HEATMAPS POR ROBOT Y DEL BALÓN
# ─────────────────────────────────────────────────────────────────────────────

def draw_heatmaps(bg_frame: np.ndarray, stats, cfg: dict):
    """
    Genera y guarda:
      · Un heatmap individual por robot (output/heatmaps/robot_<id>.png)
      · Un heatmap combinado por equipo    (output/heatmaps/equipo_<n>.png)
      · Un heatmap del balón               (output/heatmaps/balon.png)
      · Una imagen comparativa 2-equipos   (output/heatmaps/comparacion_equipos.png)
    """
    out_dir = Path("output/heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W = bg_frame.shape[:2]
    bg_rgb = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)

    # ── Heatmap del balón ──────────────────────────────────────────────────
    _save_single_heatmap(
        bg_rgb,
        stats.ball_heatmap,
        _BALL_CMAP,
        title="Mapa de Calor — Balón",
        subtitle=f"Distancia total: {stats.ball_total_distance_m:.1f} m  |  "
                 f"Velocidad máx.: {stats.ball_max_speed_kmh:.1f} km/h",
        path=out_dir / "balon.png",
    )

    # ── Heatmaps individuales por robot ───────────────────────────────────
    team_accumulators: dict[int, np.ndarray] = {0: np.zeros((H, W), np.float32),
                                                  1: np.zeros((H, W), np.float32)}

    for tid, hmap in stats.robot_heatmaps.items():
        team = stats._team_map.get(int(tid), -1)
        cmap = TEAM_CMAPS[team] if team in (0, 1) else _BALL_CMAP
        tname = TEAM_NAMES[team] if team in (0, 1) else "Sin equipo"
        dist  = stats.robot_distance.get(int(tid), 0.0)

        if hmap.shape != (H, W):
            hmap = cv2.resize(hmap, (W, H))

        _save_single_heatmap(
            bg_rgb,
            hmap,
            cmap,
            title=f"Mapa de Calor — Robot #{tid} ({tname})",
            subtitle=f"Distancia recorrida: {dist:.1f} m",
            path=out_dir / f"robot_{tid}.png",
        )

        if team in (0, 1):
            resized = hmap if hmap.shape == (H, W) else cv2.resize(hmap, (W, H))
            team_accumulators[team] += resized

    # ── Heatmaps por equipo ───────────────────────────────────────────────
    for team_id, acc in team_accumulators.items():
        if acc.max() == 0:
            continue
        total_dist = sum(
            d for tid, d in stats.robot_distance.items()
            if stats._team_map.get(int(tid), -1) == team_id
        )
        _save_single_heatmap(
            bg_rgb,
            acc,
            TEAM_CMAPS[team_id],
            title=f"Mapa de Calor — {TEAM_NAMES[team_id]}",
            subtitle=f"Distancia total del equipo: {total_dist:.1f} m",
            path=out_dir / f"equipo_{team_id}.png",
        )

    # ── Comparación lado a lado ───────────────────────────────────────────
    _save_comparison_heatmap(bg_rgb, team_accumulators, stats, out_dir / "comparacion_equipos.png")

    print(f"[OK] Heatmaps guardados → {out_dir}/")


def _save_single_heatmap(bg_rgb, hmap, cmap, title, subtitle, path):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(bg_rgb, alpha=0.7)

    if hmap.max() > 0:
        norm_map = hmap / hmap.max()
    else:
        norm_map = hmap

    im = ax.imshow(norm_map, cmap=cmap, alpha=0.65, vmin=0, vmax=1)

    ax.set_title(title, fontsize=16, fontweight="bold", color="white", pad=12)
    ax.set_xlabel(subtitle, fontsize=11, color="#CCCCCC")
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Densidad de presencia", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


def _save_comparison_heatmap(bg_rgb, team_accumulators, stats, path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.patch.set_facecolor("#0d1117")

    for ax, (team_id, acc) in zip(axes, team_accumulators.items()):
        ax.imshow(bg_rgb, alpha=0.65)
        if acc.max() > 0:
            ax.imshow(acc / acc.max(), cmap=TEAM_CMAPS[team_id], alpha=0.7, vmin=0, vmax=1)

        total_dist = sum(
            d for tid, d in stats.robot_distance.items()
            if stats._team_map.get(int(tid), -1) == team_id
        )
        total_poss = sum(stats.possession.values()) or 1
        poss_pct = stats.possession.get(team_id, 0) / total_poss * 100

        ax.set_title(
            f"{TEAM_NAMES[team_id]}",
            fontsize=15, fontweight="bold", color="white", pad=8,
        )
        ax.set_xlabel(
            f"Dist. total: {total_dist:.1f} m   |   Posesión: {poss_pct:.1f}%",
            fontsize=11, color="#BBBBBB",
        )
        ax.axis("off")
        ax.set_facecolor("#0d1117")

    fig.suptitle("Comparación de Zona de Influencia por Equipo",
                 fontsize=18, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. OVERLAY DE ESTADÍSTICAS DEL BALÓN (frame a frame)
# ─────────────────────────────────────────────────────────────────────────────

def draw_ball_stats_overlay(frame: np.ndarray, stats, frame_idx: int, fps: float) -> np.ndarray:
    """Panel en esquina inferior izquierda con estadísticas del balón en tiempo real."""
    out = frame.copy()
    H, W = out.shape[:2]

    panel_w, panel_h = 260, 90
    px, py = 10, H - panel_h - 10

    # Fondo semitransparente
    overlay = out.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    cv2.rectangle(out, (px, py), (px + panel_w, py + panel_h), (255, 220, 0), 1)

    # Ícono de balón (círculo)
    cv2.circle(out, (px + 18, py + 20), 10, (255, 220, 0), -1)
    cv2.circle(out, (px + 18, py + 20), 10, (0, 0, 0), 1)

    tiempo = frame_idx / fps
    lines = [
        f"Velocidad: {stats.ball_speed_kmh:5.1f} km/h",
        f"Max vel.:  {stats.ball_max_speed_kmh:5.1f} km/h",
        f"Distancia: {stats.ball_total_distance_m:6.1f} m",
        f"Tiempo:    {int(tiempo//60):02d}:{int(tiempo%60):02d}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (px + 34, py + 18 + i * 18),
                    FONT, 0.40, (230, 230, 230), 1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. BARRA DE POSESIÓN
# ─────────────────────────────────────────────────────────────────────────────

def draw_possession_bar(frame: np.ndarray, possession: dict, W: int, H: int) -> np.ndarray:
    """Barra horizontal en la parte inferior del frame."""
    out = frame.copy()
    total = sum(possession.values()) or 1

    bar_h   = 22
    bar_y   = H - bar_h
    poss_a  = possession.get(0, 0) / total
    poss_b  = possession.get(1, 0) / total
    poss_n  = possession.get(-1, 0) / total

    xa = int(W * poss_a)
    xb = xa + int(W * poss_b)

    # Fondo negro
    cv2.rectangle(out, (0, bar_y), (W, H), (10, 10, 10), -1)

    # Equipo Azul
    if xa > 0:
        cv2.rectangle(out, (0, bar_y), (xa, H), (180, 80, 20), -1)
    # Equipo Rojo
    if xb > xa:
        cv2.rectangle(out, (xa, bar_y), (xb, H), (40, 40, 200), -1)
    # Sin posesión
    if xb < W:
        cv2.rectangle(out, (xb, bar_y), (W, H), (60, 60, 60), -1)

    # Textos
    def pct_text(pct, x_start, x_end, color):
        txt = f"{pct*100:.0f}%"
        tw  = cv2.getTextSize(txt, FONT, 0.45, 1)[0][0]
        mid = (x_start + x_end) // 2 - tw // 2
        if x_end - x_start > 30:
            cv2.putText(out, txt, (mid, bar_y + 15), FONT, 0.45, color, 1, cv2.LINE_AA)

    pct_text(poss_a, 0,  xa, (255, 255, 255))
    pct_text(poss_b, xa, xb, (255, 255, 255))

    # Etiquetas laterales
    cv2.putText(out, "Azul", (4, bar_y + 15), FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    txt = "Rojo"
    tw  = cv2.getTextSize(txt, FONT, 0.38, 1)[0][0]
    cv2.putText(out, txt, (W - tw - 4, bar_y + 15), FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. MINIMAPA (esquina superior derecha)
# ─────────────────────────────────────────────────────────────────────────────

def draw_mini_map(frame: np.ndarray, detections, team_ids, robot_dets, W: int, H: int) -> np.ndarray:
    """Minimapa cenital con posiciones de robots y balón."""
    import supervision as sv

    out  = frame.copy()
    MW, MH = 160, 100   # tamaño del minimapa
    MX, MY = W - MW - 10, 10

    # Fondo del minimapa (cancha verde)
    mmap = np.zeros((MH, MW, 3), dtype=np.uint8)
    mmap[:] = (30, 100, 30)
    cv2.rectangle(mmap, (2, 2), (MW - 3, MH - 3), (255, 255, 255), 1)
    cv2.line(mmap, (MW // 2, 2), (MW // 2, MH - 3), (255, 255, 255), 1)

    # Robots
    robot_id_to_team = {}
    if team_ids is not None and robot_dets.tracker_id is not None:
        for tid, t in zip(robot_dets.tracker_id, team_ids):
            robot_id_to_team[int(tid)] = int(t)

    COLORS = [(220, 120, 30), (50, 50, 220)]

    for i, bbox in enumerate(detections.xyxy):
        cx = int((bbox[0] + bbox[2]) / 2 / W * MW)
        cy = int((bbox[1] + bbox[3]) / 2 / H * MH)
        cx = max(3, min(cx, MW - 4))
        cy = max(3, min(cy, MH - 4))

        cls = detections.class_id[i]
        tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1

        if cls == 32:  # balón
            cv2.circle(mmap, (cx, cy), 4, (0, 255, 255), -1)
        else:
            team = robot_id_to_team.get(tid, -1)
            color = COLORS[team] if team in (0, 1) else (180, 180, 180)
            cv2.circle(mmap, (cx, cy), 5, color, -1)
            cv2.circle(mmap, (cx, cy), 5, (0, 0, 0), 1)

    # Copiar al frame principal
    out[MY:MY + MH, MX:MX + MW] = mmap
    cv2.rectangle(out, (MX, MY), (MX + MW, MY + MH), (200, 200, 200), 1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. IMAGEN DE RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────

def create_summary_visualization(bg_frame: np.ndarray, stats, cfg: dict, video_info):
    """Dashboard estático con estadísticas finales del partido."""
    bg_rgb = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
    H, W   = bg_frame.shape[:2]

    total_poss = sum(stats.possession.values()) or 1
    poss_a = stats.possession.get(0, 0) / total_poss * 100
    poss_b = stats.possession.get(1, 0) / total_poss * 100

    dist_a = sum(d for tid, d in stats.robot_distance.items()
                 if stats._team_map.get(int(tid), -1) == 0)
    dist_b = sum(d for tid, d in stats.robot_distance.items()
                 if stats._team_map.get(int(tid), -1) == 1)

    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    # ── Campo con trail del balón ──────────────────────────────────────────
    ax_field = fig.add_subplot(gs[:, 0])
    ax_field.imshow(bg_rgb, alpha=0.6)
    if stats.ball_positions:
        xs = [p[1] for p in stats.ball_positions]
        ys = [p[2] for p in stats.ball_positions]
        ax_field.plot(xs, ys, color="#FFDC00", linewidth=0.6, alpha=0.7)
        ax_field.scatter(xs[-1], ys[-1], c="#FFDC00", s=60, zorder=5)
    ax_field.set_title("Trayectoria del Balón", color="white", fontsize=13, fontweight="bold")
    ax_field.axis("off")
    ax_field.set_facecolor("#0d1117")

    # ── Gráfico de posesión ────────────────────────────────────────────────
    ax_poss = fig.add_subplot(gs[0, 1])
    wedges, texts, autotexts = ax_poss.pie(
        [poss_a, poss_b, max(0, 100 - poss_a - poss_b)],
        labels=["Azul", "Rojo", "Neutro"],
        colors=["#1E78DC", "#DC3232", "#555555"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "white", "fontsize": 10},
    )
    for at in autotexts:
        at.set_color("white")
    ax_poss.set_title("Posesión del Balón", color="white", fontsize=12, fontweight="bold")

    # ── Distancia recorrida por robot ──────────────────────────────────────
    ax_dist = fig.add_subplot(gs[0, 2])
    robot_ids = sorted(stats.robot_distance.keys())
    distances  = [stats.robot_distance[r] for r in robot_ids]
    bar_colors = [
        "#1E78DC" if stats._team_map.get(int(r), -1) == 0
        else "#DC3232" if stats._team_map.get(int(r), -1) == 1
        else "#888888"
        for r in robot_ids
    ]
    bars = ax_dist.barh([f"Robot #{r}" for r in robot_ids], distances,
                        color=bar_colors, edgecolor="#333333")
    ax_dist.set_xlabel("Distancia (m)", color="white", fontsize=9)
    ax_dist.set_title("Distancia por Robot", color="white", fontsize=12, fontweight="bold")
    ax_dist.tick_params(colors="white")
    ax_dist.set_facecolor("#161b22")
    ax_dist.spines[["top", "right"]].set_visible(False)
    for sp in ax_dist.spines.values():
        sp.set_color("#333333")

    # Añadir valores en las barras
    for bar, val in zip(bars, distances):
        ax_dist.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f} m", va="center", color="white", fontsize=8)

    # ── Estadísticas del balón ─────────────────────────────────────────────
    ax_ball = fig.add_subplot(gs[1, 1:])
    ax_ball.axis("off")
    ax_ball.set_facecolor("#161b22")

    dur_seg = video_info.total_frames / video_info.fps
    stats_text = [
        ("🏆  Estadísticas del Partido",  "", "#FFDC00"),
        ("",                               "", "white"),
        ("Velocidad máxima del balón:",    f"{stats.ball_max_speed_kmh:.1f} km/h",  "#00BFFF"),
        ("Distancia total del balón:",     f"{stats.ball_total_distance_m:.1f} m",  "#00BFFF"),
        ("Posesión — Equipo Azul:",        f"{poss_a:.1f}%",   "#1E78DC"),
        ("Posesión — Equipo Rojo:",        f"{poss_b:.1f}%",   "#DC3232"),
        ("Distancia — Equipo Azul:",       f"{dist_a:.1f} m",  "#1E78DC"),
        ("Distancia — Equipo Rojo:",       f"{dist_b:.1f} m",  "#DC3232"),
        ("Duración del video:",            f"{int(dur_seg//60):02d}:{int(dur_seg%60):02d} min", "white"),
        ("Eventos detectados:",            str(len(stats.events)), "#FF8C00"),
    ]
    for row, (label, value, color) in enumerate(stats_text):
        ax_ball.text(0.02, 0.95 - row * 0.09, label,
                     transform=ax_ball.transAxes, fontsize=11,
                     color=color, fontweight="bold" if row == 0 else "normal")
        if value:
            ax_ball.text(0.58, 0.95 - row * 0.09, value,
                         transform=ax_ball.transAxes, fontsize=11, color="white")

    fig.suptitle("Copa FutBotMX — Resumen del Análisis",
                 fontsize=20, fontweight="bold", color="white", y=1.01)

    out_path = f"{cfg.get('out_root', 'output')}/resumen_partido.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"[OK] Resumen guardado → {out_path}")
