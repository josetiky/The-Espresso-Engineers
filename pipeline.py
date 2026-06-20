"""
Copa FutBotMX — Capítulo Visión por Computadora (Categoría Amateur)
Pipeline principal: Detección · ROI dinámico · Tracking · Segmentación ·
                     Homografía (vista cenital) · Estadísticas

Optimizado para Apple Silicon (M5, backend MPS).

CAMBIOS v4:
  · Detección con modelo propio entrenado v6 (clase 1 = robot, clase 2 = balón).
  · ROI dinámico por color HSV (cancha verde) — la cámara es móvil, por lo
    que se recalcula una máscara binaria en CADA frame en vez de usar un
    sv.PolygonZone fijo.
  · Guardado seguro de video y JSON: nunca sobreescribe, añade sufijo _1, _2…
  · Homografía (cv2.getPerspectiveTransform): los heatmaps ya NO se dibujan
    sobre el frame inclinado de la cámara, sino sobre una plantilla 2D limpia
    de la cancha vista desde arriba (vista de pájaro).
  · Integración estándar de utils.TeamAssigner, utils.StatsTracker y las
    funciones de utils.visualizations (mini-mapa, barra de posesión, overlay
    de balón, dashboard de resumen).
"""

import os
import cv2
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO, SAM
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")   # backend sin pantalla (sólo guardamos a disco)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils.team_assigner import TeamAssigner
from utils.stats_tracker import StatsTracker
from utils.roster import RosterManager, BallCoaster
from utils.team_formation import FormationTeamAssigner
from utils.visualizations import (
    draw_ball_stats_overlay,
    draw_possession_bar,
    draw_mini_map,
    create_summary_visualization,
)
# NOTA: 'draw_heatmaps' de utils.visualizations NO se usa aquí a propósito.
# Esa función dibuja sobre el frame original (cámara inclinada). El
# Requisito 4 exige heatmaps sobre una vista cenital homografiada, por lo
# que esa lógica vive en este archivo (ver 'draw_homography_heatmaps').


# ════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════

def _resolve_device() -> str:
    """Fuerza el uso de 'mps' en Apple Silicon (M5). Si no está disponible
    (p. ej. corriendo en otra máquina), cae a 'cpu' con una advertencia."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    print("[WARN] MPS no disponible en este equipo — usando CPU (será más lento).")
    return "cpu"


# Permite que operaciones aún no implementadas nativamente en MPS caigan a
# CPU automáticamente en vez de lanzar una excepción (recomendado por
# PyTorch para Apple Silicon).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


CONFIG = {
    # ── Rutas ──────────────────────────────────────────────────────────
    "video_input":  "assets/partido.mp4",
    "out_root":     "output",                          # carpeta raíz de resultados
    "video_output": "output/partido_analizado.mp4",   # se renombra si ya existe
    "stats_output": "output/estadisticas.json",

    # ── Modelo propio entrenado (Requisito 1) ────────────────────────────
    # Dataset v2 (copafutbot.yolov11) → 2 clases, según dataset/data.yaml:
    #   0 = Robot
    #   1 = pelota
    # Modelo actual entrenado: futbotmx_v2.pt.
    "yolo_model": "futbotmx_v2.pt",
    "sam_model":   "sam2.1_s.pt",
    "class_robot": 0,
    "class_ball":  1,

    # ── Detección ──────────────────────────────────────────────────────
    # 0.40 = valores de reconocimiento más altos (menos falsos positivos).
    # Si quieres recuperar robots que parpadean, baja a 0.30 — pero eso mete
    # más ruido. Aquí se deja en el valor original más estable.
    # 0.25: el diagnóstico mostró que con 0.40 se perdían robots (2.4/4) y con
    # 0.25 sube a ~3.0; el roster descarta los falsos de más, así que es seguro.
    "confidence":    0.25,
    "iou_threshold": 0.45,
    # 768 ≈ resolución de entrenamiento (640): el diagnóstico mostró que SUBIR
    # a 960/1280 empeoraba el balón (el modelo no vio esos tamaños). No subir.
    "infer_imgsz":   768,

    # ── Segmentación SAM 3 ─────────────────────────────────────────────
    "sam_enabled": True,
    "sam_max_boxes_per_frame": 12,

    # ── Tracking ByteTrack (Supervision) ──────────────────────────────
    # Valores originales (más estables). Si quieres más "memoria" del tracker
    # para reducir parpadeo, sube track_buffer a 90 y baja track_threshold.
    "track_threshold":  0.35,
    "track_buffer":     60,
    "match_threshold":  0.8,
    "frame_rate":        30,

    # ── Requisito 2 (CRÍTICO) — ROI dinámico por color ────────────────
    # Rango HSV de la cancha verde. La cámara es móvil ⇒ esta máscara se
    # recalcula en cada frame; NO se usa un polígono fijo (sv.PolygonZone).
    "hsv_lower": np.array([35,  40,  40], dtype=np.uint8),
    "hsv_upper": np.array([85, 255, 255], dtype=np.uint8),
    "field_mask_morph_kernel": 9,   # limpieza morfológica de ruido en la máscara

    # ── Requisito 4 — Homografía / vista cenital ──────────────────────
    # Lienzo con la MISMA proporción que la cancha real (182×243 cm, reglas
    # 7.1) a 2 px/cm → 364×486 px (VERTICAL). Antes era 700×440 (horizontal),
    # lo que estiraba la cancha y daba escalas distintas por eje → distancias
    # y velocidades infladas. Ahora la escala es idéntica en ambos ejes.
    "field_template_size": (364, 486),   # (ancho, alto) px = 182×243 cm a 2 px/cm
    "min_field_area_ratio": 0.05,        # área mínima del contorno verde válida
    "homography_recalc_interval": 1,     # recalcular M cada N frames (cámara móvil)
    # Suavizado temporal de las 4 esquinas (EMA). La cámara se mueve y la
    # detección de esquinas "tiembla" frame a frame → posiciones cenitales
    # que saltan → velocidades/distancias infladas. alpha bajo = más estable.
    "homography_smooth_alpha": 0.25,
    "topdown_sigma_robot": 14,
    "topdown_sigma_ball":  10,

    # ── Video de PURA HOMOGRAFÍA (vista cenital animada de la cancha) ──
    "tactical_enabled": True,
    "tactical_video_output": "output/partido_homografia.mp4",
    "tactical_scale": 2,                 # factor de zoom del lienzo cenital
    "tactical_trail": 40,                # frames de estela en la vista cenital

    # ── Escala métrica real (Reglas Copa FutBotMX 2026, sección 7.1) ──
    # Cancha oficial: ANCHO 182 cm × LARGO 243 cm. Pelota = bola de golf de
    # 42 mm. Se usa para convertir la vista cenital a METROS reales. Como el
    # lienzo ya tiene la proporción de la cancha, la escala es consistente.
    "field_real_m": (1.82, 2.43),        # (ancho, alto) reales en metros
    # Topes físicos de velocidad (la bola de golf NO va a 115 km/h):
    "max_step_robot_m": 0.10,            # robot ≤ ~3 m/s (10.8 km/h) por frame
    "max_step_ball_m":  0.20,            # balón ≤ ~6 m/s (21.6 km/h) por frame
    # Anti-ruido: suavizado de posición (EMA) + banda muerta. Sin esto, cada
    # micro-temblor de detección se suma a la distancia e infla la velocidad.
    "pos_smooth_alpha": 0.35,            # 0=muy suave, 1=sin suavizar
    "min_move_m":       0.015,           # ignora movimientos < 1.5 cm (ruido)

    # ── Balón (Requisito 1) — más sensible que los robots ─────────────
    # El balón es pequeño y se detecta peor: se usa un umbral propio más
    # bajo y un respaldo por color HSV (naranja) cuando YOLO lo pierde.
    "ball_confidence": 0.15,
    "ball_hsv_fallback": True,
    "ball_hsv_lower": np.array([5,  120, 120], dtype=np.uint8),
    "ball_hsv_upper": np.array([22, 255, 255], dtype=np.uint8),
    "ball_min_area": 30,                 # px² mínimos del blob naranja

    # ── REGLAS DEL DOMINIO — roster fijo + oclusiones ─────────────────
    # Máximo 4 robots (2 por equipo). El detector a veces ve de más (reflejos,
    # público, sombras): NO es un robot nuevo → se descarta. Las identidades
    # sobreviven a las oclusiones (el slot se conserva 'roster_coast_frames').
    "max_robots_per_team": 2,            # 2 azules + 2 rojos = 4 máx.
    "roster_gate_px":      220,          # radio máx. para re-asociar un robot
    "roster_coast_frames": 30,           # frames que un robot ocluido conserva su ID
    "ball_coast_frames":   12,           # frames que la pelota ocluida "navega" por inercia

    # ── Asignación de EQUIPO ──────────────────────────────────────────
    # "auto"      → usa la formación inicial (kickoff); si no es clara, color.
    # "formation" → solo formación inicial.   "color" → solo k-means de color.
    "team_mode":            "auto",
    "formation_frames":     45,          # frames de "saque" a observar (~1.5 s)
    "formation_axis":       "auto",      # eje de separación: 'x' | 'y' | 'auto'
    "formation_min_sep_px": 60,          # separación mínima entre bandos para ser válida

    # ── Posesión + eventos (Requisito 3.5.1) — radios en METROS ───────
    "possession_capture_m": 0.30,        # entra en posesión si dist < esto
    "possession_release_m": 0.50,        # suelta el balón si dist > esto
    "shot_speed_kmh":       6.0,         # umbral de velocidad para "tiro a gol"

    # ── Backend de segmentación: "sam2" | "sam3" | "sam3_hf" ──────────
    # sam2  → SAM("sam2.1_s.pt")  (rápido, base, por defecto)
    # sam3  → SAM("sam3.pt")      (Ultralytics, requiere el peso de Meta)
    # sam3_hf → SAM 3 de Meta por TEXTO/vocabulario abierto (transformers).
    "seg_backend": "sam3_hf",

    # ── SAM 3 (texto) dentro del pipeline ─────────────────────────────
    # SAM 3 por texto es LENTO (~seg/frame). Para no tardar horas, las
    # MÁSCARAS de SAM 3 se generan cada `sam3_every_n` frames (y se reusan en
    # los intermedios). El tracking, la homografía y TODAS las estadísticas
    # siguen corriendo en CADA frame con YOLO — SAM 3 es la capa de
    # segmentación visual (lo que pide el reto).
    "sam3_prompts":   ["soccer robot", "orange ball"],
    "sam3_every_n":   15,      # genera máscaras SAM 3 cada N frames
    "sam3_threshold": 0.5,     # umbral de confianza de las máscaras

    # ── Visualización sobre el video ──────────────────────────────────
    "trail_length": 45,
    # Si False: el video "partido_analizado" muestra SOLO "Robot" y "Pelota"
    # (sin equipos). La lógica de equipos sigue en las estadísticas/heatmaps.
    "video_show_teams": False,

    # Hardware forzado a MPS (Apple Silicon M5)
    "device": _resolve_device(),
}

print(f"[INFO] Dispositivo de inferencia: {CONFIG['device'].upper()}")


# ════════════════════════════════════════════════════════════════════════
# REQUISITO 3 — GUARDADO SEGURO (nunca sobreescribe)
# ════════════════════════════════════════════════════════════════════════

def safe_output_path(path: str) -> Path:
    """
    Devuelve un Path que no colisione con archivos existentes.
    Si 'output/partido_analizado.mp4' ya existe, devuelve
    'output/partido_analizado_1.mp4', luego _2, _3, etc.
    """
    p = Path(path)
    if not p.exists():
        return p

    stem, suffix, parent = p.stem, p.suffix, p.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ════════════════════════════════════════════════════════════════════════
# REQUISITO 2 (CRÍTICO) — FILTRO DE ROI DINÁMICO POR COLOR
# ════════════════════════════════════════════════════════════════════════

def get_field_mask(frame: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Genera una máscara binaria (0/255) de la cancha verde en el frame
    ACTUAL. Como la cámara es móvil, esto se recalcula en cada frame — no
    se asume ninguna posición fija de la cancha dentro de la imagen.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg["hsv_lower"], cfg["hsv_upper"])

    # Limpieza morfológica: cierra huecos pequeños (líneas blancas de la
    # cancha, reflejos) y elimina ruido aislado (ropa verde del público,
    # carteles, etc.) que pudiera colarse en el rango de color.
    k = cfg["field_mask_morph_kernel"]
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    return mask


def filter_detections_by_field_mask(detections: sv.Detections, mask: np.ndarray) -> sv.Detections:
    """
    Elimina de 'detections' cualquier bounding box cuyo centro (cx, cy)
    caiga FUERA de la máscara verde de la cancha. Se aplica ANTES del
    tracker para no contaminar las trayectorias con falsos positivos del
    público, jueces o mesas alrededor de la cancha.
    """
    if len(detections) == 0:
        return detections

    H, W = mask.shape
    cx = ((detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2).astype(int)
    cy = ((detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2).astype(int)
    cx = np.clip(cx, 0, W - 1)
    cy = np.clip(cy, 0, H - 1)

    inside_field = mask[cy, cx] > 0
    return detections[inside_field]


def filter_robots_by_field_mask(detections: sv.Detections, mask: np.ndarray,
                                ball_class: int) -> sv.Detections:
    """Descarta detecciones FUERA de la cancha (público, mesas) pero NUNCA al
    balón. CLAVE: el robot es negro, así que el centro de su caja NO cae sobre
    verde — por eso se usa la región de cancha RELLENA (el contorno verde más
    grande relleno), de modo que un robot encima del pasto cuente como dentro.
    Si la cancha no se detecta bien, NO filtra (evita quedarse sin robots)."""
    if len(detections) == 0:
        return detections
    H, W = mask.shape

    # Rellenar el contorno verde más grande → polígono sólido de la cancha
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return detections   # sin cancha detectada → no filtrar
    field = np.zeros_like(mask)
    big = max(contours, key=cv2.contourArea)
    cv2.drawContours(field, [big], -1, 255, thickness=cv2.FILLED)

    cx = ((detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2).astype(int)
    cy = ((detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2).astype(int)
    cx = np.clip(cx, 0, W - 1)
    cy = np.clip(cy, 0, H - 1)
    inside_field = field[cy, cx] > 0
    is_robot = detections.class_id != ball_class
    keep = inside_field | (detections.class_id == ball_class)  # balón siempre

    # Respaldo: si el filtro dejaría SIN robots pero sí hay robots detectados,
    # no filtrar (la detección de cancha falló este frame).
    if is_robot.any() and not (keep & is_robot).any():
        return detections
    return detections[keep]


def _combine_dets(a: sv.Detections, b: sv.Detections) -> sv.Detections:
    """Concatena dos sv.Detections (robots + balón) conservando tracker_id.
    Evita sv.Detections.merge (que exige idénticas claves en .data) y garantiza
    que SIEMPRE haya tracker_id (la pelota 'navegada' por inercia no lo trae)."""
    def norm(d):
        if len(d) and d.tracker_id is None:
            d.tracker_id = np.full(len(d), -1, dtype=int)
        return d
    a, b = norm(a), norm(b)
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return sv.Detections(
        xyxy=np.vstack([a.xyxy, b.xyxy]),
        confidence=np.concatenate([a.confidence, b.confidence]),
        class_id=np.concatenate([a.class_id, b.class_id]),
        tracker_id=np.concatenate([a.tracker_id, b.tracker_id]).astype(int),
    )


def _concat_masks(dets_list: list) -> sv.Detections:
    """Combina las máscaras de varios prompts de SAM 3 en un solo
    sv.Detections (solo máscaras, sin class_id — se dibuja por índice)."""
    dets_list = [d for d in dets_list if d is not None and len(d) > 0 and d.mask is not None]
    if not dets_list:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.vstack([d.xyxy for d in dets_list]).astype(np.float32),
        mask=np.concatenate([d.mask for d in dets_list], axis=0),
        confidence=np.concatenate([d.confidence for d in dets_list]).astype(np.float32),
    )


def detect_ball_hsv(frame: np.ndarray, field_mask: np.ndarray, cfg: dict) -> sv.Detections:
    """Respaldo de detección del balón por color (naranja) cuando YOLO lo
    pierde. Busca el blob naranja más grande DENTRO de la cancha verde.
    Devuelve sv.Detections (0 o 1 caja) con class_id = class_ball."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg["ball_hsv_lower"], cfg["ball_hsv_upper"])
    # Dilatar un poco la cancha para no perder el balón sobre la línea
    field_dil = cv2.dilate(field_mask, np.ones((15, 15), np.uint8))
    mask = cv2.bitwise_and(mask, field_dil)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= cfg["ball_min_area"]]
    if not contours:
        return sv.Detections.empty()

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    xyxy = np.array([[x, y, x + w, y + h]], dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.30], dtype=np.float32),   # respaldo: confianza media
        class_id=np.array([cfg["class_ball"]], dtype=int),
    )


# ════════════════════════════════════════════════════════════════════════
# REQUISITO 4 — HOMOGRAFÍA Y VISTA CENITAL (TOP-DOWN)
# ════════════════════════════════════════════════════════════════════════

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como [sup-izq, sup-der, inf-der, inf-izq]."""
    s    = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_field_corners(mask: np.ndarray, cfg: dict) -> np.ndarray | None:
    """
    Detecta las 4 esquinas aproximadas de la cancha a partir de la máscara
    verde (la misma que produce get_field_mask). Toma el contorno más
    grande y le ajusta un rectángulo rotado mínimo (cv2.minAreaRect).

    Devuelve None si no hay un contorno suficientemente grande ese frame
    (cancha ocluida, fuera de cuadro, etc.) — en ese caso el llamador debe
    conservar la última homografía válida.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    H, W = mask.shape
    if cv2.contourArea(largest) < cfg["min_field_area_ratio"] * (H * W):
        return None

    rect = cv2.minAreaRect(largest)
    box  = cv2.boxPoints(rect)
    return _order_corners(box)


def build_field_homography(src_corners: np.ndarray, dst_size: tuple[int, int]) -> np.ndarray:
    """
    Calcula la matriz de homografía (cv2.getPerspectiveTransform) que
    mapea las 4 esquinas detectadas en el frame de la cámara hacia las 4
    esquinas de un lienzo 2D rectangular limpio (vista de pájaro).
    """
    dst_w, dst_h = dst_size
    src = src_corners.astype(np.float32).copy()

    # Alinear orientación: el eje LARGO de la cancha debe caer en el eje
    # largo del lienzo (si no, la cancha sale estirada y la escala se rompe).
    top_edge  = float(np.linalg.norm(src[1] - src[0]))   # |TL→TR|
    left_edge = float(np.linalg.norm(src[3] - src[0]))   # |TL→BL|
    src_landscape = top_edge > left_edge
    dst_landscape = dst_w > dst_h
    if src_landscape != dst_landscape:
        src = np.roll(src, -1, axis=0)   # rota 90° el orden de esquinas

    dst_corners = np.array([
        [0,         0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0,         dst_h - 1],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst_corners)


def warp_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Transforma un array (N, 2) de puntos en coordenadas de cámara hacia
    coordenadas del lienzo cenital, aplicando la homografía M."""
    if points is None or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts    = points.reshape(-1, 1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(pts, M)
    return warped.reshape(-1, 2)


def create_field_template(dst_size: tuple[int, int]) -> np.ndarray:
    """
    Genera una imagen 2D limpia de una cancha vista desde arriba (BGR),
    usada como fondo para los heatmaps homografiados — en lugar del frame
    original con la cámara inclinada (Requisito 4).
    """
    # Cancha VERTICAL según reglas (182×243 cm). A 2 px/cm el lienzo es
    # 364×486. Las medidas se dibujan a escala (1 cm = w/182 px).
    w, h = dst_size
    template = np.full((h, w, 3), (60, 140, 50), dtype=np.uint8)   # verde césped
    cm = w / 182.0                                  # px por cm (eje ancho)
    white, t = (255, 255, 255), 2

    # Perímetro
    m = int(6 * cm * 0)  # sin margen: la cancha llena el lienzo
    cv2.rectangle(template, (1, 1), (w - 2, h - 2), white, t)
    # Línea central (horizontal, a la mitad del largo)
    cv2.line(template, (0, h // 2), (w, h // 2), white, t)
    # Círculo central: 60 cm de diámetro → radio 30 cm
    cv2.circle(template, (w // 2, h // 2), int(30 * cm), white, t)

    # Áreas de penalización: 25 cm profundidad × 80 cm ancho, en cada portería
    pa_w, pa_d = int(80 * cm), int(25 * cm)
    cv2.rectangle(template, (w // 2 - pa_w // 2, 1),
                  (w // 2 + pa_w // 2, 1 + pa_d), white, t)            # arriba
    cv2.rectangle(template, (w // 2 - pa_w // 2, h - 2 - pa_d),
                  (w // 2 + pa_w // 2, h - 2), white, t)               # abajo

    # Porterías: 60 cm de ancho. Amarilla arriba, azul abajo (regla 7.4.3)
    g_w = int(60 * cm)
    cv2.rectangle(template, (w // 2 - g_w // 2, 0),
                  (w // 2 + g_w // 2, int(4 * cm)), (0, 210, 230), -1)  # amarilla
    cv2.rectangle(template, (w // 2 - g_w // 2, h - int(4 * cm)),
                  (w // 2 + g_w // 2, h - 1), (210, 130, 0), -1)        # azul
    return template


# Colores de equipo en BGR para la vista cenital (azul / rojo / gris)
_TACTIC_BGR = {0: (230, 140, 30), 1: (50, 50, 230), -1: (170, 170, 170)}
_BALL_BGR   = (0, 200, 255)   # naranja-amarillo


def render_tactical_frame(field_template, robot_topdown, ball_topdown,
                          tid_team, trails, cfg):
    """Dibuja UN frame de la vista cenital (pura homografía): cancha limpia +
    estelas + robots como discos coloreados por equipo con su ID + balón.
    Devuelve una imagen BGR del tamaño del lienzo (× tactical_scale)."""
    canvas = field_template.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Estelas (trayectorias recientes) ──
    for key, pts in trails.items():
        if len(pts) < 2:
            continue
        if key == "ball":
            col = _BALL_BGR
        else:
            col = _TACTIC_BGR.get(tid_team.get(key, -1), _TACTIC_BGR[-1])
        for i in range(1, len(pts)):
            cv2.line(canvas, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                     (int(pts[i][0]), int(pts[i][1])), col, 2, cv2.LINE_AA)

    # ── Robots ──
    for tid, (x, y) in robot_topdown.items():
        col = _TACTIC_BGR.get(tid_team.get(tid, -1), _TACTIC_BGR[-1])
        c = (int(x), int(y))
        cv2.circle(canvas, c, 13, col, -1)
        cv2.circle(canvas, c, 13, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(tid), (c[0] - 7, c[1] + 5),
                    font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Balón ──
    if ball_topdown is not None:
        b = (int(ball_topdown[0]), int(ball_topdown[1]))
        cv2.circle(canvas, b, 8, _BALL_BGR, -1)
        cv2.circle(canvas, b, 8, (0, 0, 0), 2, cv2.LINE_AA)

    # ── Marca de agua / título ──
    cv2.putText(canvas, "Vista cenital (homografia) - Copa FutBotMX",
                (10, canvas.shape[0] - 12), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    scale = cfg.get("tactical_scale", 1)
    if scale != 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_LINEAR)
    return canvas


class TopDownHeatmaps:
    """
    Acumula posiciones YA TRANSFORMADAS por homografía (coordenadas del
    lienzo cenital limpio) en mapas de calor 2D — uno por robot y uno para
    el balón. Es el equivalente "vista de pájaro" de los heatmaps en
    espacio-frame que mantiene StatsTracker internamente.
    """

    def __init__(self, dst_size: tuple[int, int], cfg: dict):
        self.w, self.h = dst_size
        self.cfg = cfg
        self.ball: np.ndarray = np.zeros((self.h, self.w), dtype=np.float32)
        self.robots: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros((self.h, self.w), dtype=np.float32)
        )

    def _deposit(self, heatmap: np.ndarray, x: float, y: float, sigma: int):
        """Deposita una gaussiana local (ventana acotada, no todo el
        lienzo) — más eficiente que recalcular el grid completo cada vez."""
        cx, cy = int(round(x)), int(round(y))
        if not (0 <= cx < self.w and 0 <= cy < self.h):
            return   # punto fuera del lienzo cenital (homografía imprecisa)

        r = sigma * 3
        x0, x1 = max(0, cx - r), min(self.w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(self.h, cy + r + 1)
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        gauss  = np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * sigma ** 2))
        heatmap[y0:y1, x0:x1] += gauss

    def add_robot(self, tracker_id: int, x: float, y: float):
        self._deposit(self.robots[tracker_id], x, y, self.cfg["topdown_sigma_robot"])

    def add_ball(self, x: float, y: float):
        self._deposit(self.ball, x, y, self.cfg["topdown_sigma_ball"])


# ── Paletas para los heatmaps cenitales ────────────────────────────────
_TEAM_NAMES = ["Equipo Azul", "Equipo Rojo"]
_BLUE_CMAP  = LinearSegmentedColormap.from_list("blue_heat", ["#00000000", "#1E78DC"])
_RED_CMAP   = LinearSegmentedColormap.from_list("red_heat",  ["#00000000", "#DC3232"])
_BALL_CMAP  = LinearSegmentedColormap.from_list("ball_heat", ["#00000000", "#FFDC00"])
_TEAM_CMAPS = [_BLUE_CMAP, _RED_CMAP]


def _save_topdown_heatmap(bg_rgb, hmap, cmap, title, subtitle, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(bg_rgb)
    if hmap.max() > 0:
        ax.imshow(hmap / hmap.max(), cmap=cmap, alpha=0.75, vmin=0, vmax=1)
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=10, color="#CCCCCC")
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


def _save_topdown_comparison(bg_rgb, team_acc, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    for ax, (team_id, acc) in zip(axes, team_acc.items()):
        ax.imshow(bg_rgb)
        if acc.max() > 0:
            ax.imshow(acc / acc.max(), cmap=_TEAM_CMAPS[team_id], alpha=0.75, vmin=0, vmax=1)
        ax.set_title(_TEAM_NAMES[team_id], fontsize=13, fontweight="bold", color="white")
        ax.axis("off")
    fig.suptitle("Comparación de Zona de Influencia — Vista Cenital",
                 fontsize=16, fontweight="bold", color="white", y=1.03)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)


def draw_homography_heatmaps(topdown: TopDownHeatmaps, field_template: np.ndarray,
                             stats: StatsTracker, out_dir="output/heatmaps"):
    """
    Genera y guarda los heatmaps finales SOBRE LA PLANTILLA 2D CENITAL
    (Requisito 4) en vez de sobre el frame original con la cámara
    inclinada. Usa stats._team_map y stats.robot_distance — ya calculados
    de forma estándar por StatsTracker — únicamente para colorear/etiquetar
    cada robot según su equipo.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Limpiar heatmaps de robots de corridas anteriores (cuando había IDs
    # 5,6,7…). Con el roster fijo, sólo deben quedar robot_1..4.png.
    for old in out_dir.glob("robot_*.png"):
        try:
            old.unlink()
        except OSError:
            pass

    bg_rgb = cv2.cvtColor(field_template, cv2.COLOR_BGR2RGB)

    # ── Balón ──────────────────────────────────────────────────────────
    _save_topdown_heatmap(
        bg_rgb, topdown.ball, _BALL_CMAP,
        "Mapa de Calor — Balón (vista cenital)",
        f"Distancia total: {stats.ball_total_distance_m:.1f} m",
        out_dir / "balon.png",
    )

    # ── Robots individuales + acumulado por equipo ───────────────────────
    team_acc = {0: np.zeros((topdown.h, topdown.w), np.float32),
                1: np.zeros((topdown.h, topdown.w), np.float32)}

    for tid, hmap in topdown.robots.items():
        team  = stats._team_map.get(int(tid), -1)
        cmap  = _TEAM_CMAPS[team] if team in (0, 1) else _BALL_CMAP
        tname = _TEAM_NAMES[team] if team in (0, 1) else "Sin equipo"
        dist  = stats.robot_distance.get(int(tid), 0.0)

        _save_topdown_heatmap(
            bg_rgb, hmap, cmap,
            f"Mapa de Calor — Robot #{tid} ({tname})",
            f"Distancia recorrida: {dist:.1f} m",
            out_dir / f"robot_{tid}.png",
        )
        if team in (0, 1):
            team_acc[team] += hmap

    for team_id, acc in team_acc.items():
        if acc.max() == 0:
            continue
        total_dist = sum(
            d for t, d in stats.robot_distance.items()
            if stats._team_map.get(int(t), -1) == team_id
        )
        _save_topdown_heatmap(
            bg_rgb, acc, _TEAM_CMAPS[team_id],
            f"Mapa de Calor — {_TEAM_NAMES[team_id]}",
            f"Distancia total del equipo: {total_dist:.1f} m",
            out_dir / f"equipo_{team_id}.png",
        )

    # ── Comparación lado a lado ───────────────────────────────────────────
    _save_topdown_comparison(bg_rgb, team_acc, out_dir / "comparacion_equipos.png")

    print(f"[OK] Heatmaps (vista cenital, homografía) guardados → {out_dir}/")


# ════════════════════════════════════════════════════════════════════════
# CARGA DE MODELOS
# ════════════════════════════════════════════════════════════════════════

def load_models(cfg: dict):
    """Carga el modelo YOLO propio (v6: 0=robot, 1=balón) y SAM 3."""
    try:
        yolo = YOLO(cfg["yolo_model"])
    except Exception as e:
        raise RuntimeError(
            f"[ERROR] No se pudo cargar el modelo YOLO desde '{cfg['yolo_model']}'. "
            f"Verifica la ruta del modelo entrenado v6. Detalle: {e}"
        )
    yolo.to(cfg["device"])
    print(f"[INFO] Modelo YOLO cargado: {cfg['yolo_model']} (clases: robot=1, balón=2)")

    sam = None
    if cfg["sam_enabled"]:
        backend = cfg.get("seg_backend", "sam2")
        if backend == "sam3_hf":
            # SAM 3 de Meta por TEXTO (transformers). Devuelve un Sam3Backend
            # con interfaz .segment(frame, texto) — se maneja aparte en el loop.
            try:
                from utils.sam3_backend import Sam3Backend
                sam = Sam3Backend(device=cfg["device"])
                print("[INFO] Segmentador SAM 3 (texto/vocabulario abierto) cargado.")
            except Exception as e:
                print(f"[WARN] SAM 3 (texto) no disponible ({e}). Respaldo a SAM 2.1.")
                try:
                    sam = SAM(cfg["sam_model"]); sam.to(cfg["device"])
                except Exception:
                    sam = None
        else:
            weight = {"sam2": cfg["sam_model"], "sam3": "sam3.pt"}.get(backend, cfg["sam_model"])
            try:
                sam = SAM(weight)
                sam.to(cfg["device"])
                print(f"[INFO] Segmentador cargado: {weight} (backend='{backend}').")
            except Exception as e:
                print(f"[WARN] SAM '{weight}' no disponible ({e}).")
                if weight != cfg["sam_model"]:
                    try:
                        sam = SAM(cfg["sam_model"]); sam.to(cfg["device"])
                        print(f"[INFO] Respaldo a {cfg['sam_model']}.")
                    except Exception as e2:
                        print(f"[WARN] Sin segmentación ({e2}). Continúa solo con YOLO.")

    return yolo, sam


# ════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: dict = CONFIG):
    # ── Modelos ────────────────────────────────────────────────────────
    yolo, sam = load_models(cfg)

    # ── Tracker ByteTrack (Supervision) ───────────────────────────────
    tracker = sv.ByteTrack(
        track_activation_threshold=cfg["track_threshold"],
        lost_track_buffer=cfg["track_buffer"],
        minimum_matching_threshold=cfg["match_threshold"],
        frame_rate=cfg["frame_rate"],
    )

    # ── Anotadores Supervision ─────────────────────────────────────────
    label_annotator = sv.LabelAnnotator(text_scale=0.45, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=cfg["trail_length"],
        position=sv.Position.BOTTOM_CENTER,
    )
    mask_annotator = sv.MaskAnnotator(opacity=0.35)
    # Anotador para las máscaras de SAM 3 (sin class_id → color por índice)
    sam3_mask_annotator = sv.MaskAnnotator(opacity=0.45,
                                           color_lookup=sv.ColorLookup.INDEX)
    last_sam3_masks = [None]   # última máscara SAM 3 (se reusa entre frames)

    # ── Módulos externos (Requisito 5 — integración estándar) ─────────
    team_assigner      = TeamAssigner()
    teams_initialized   = False
    stats               = StatsTracker(cfg)

    # Reglas del dominio: roster fijo (máx. 4 robots) + inercia de pelota
    roster        = RosterManager(cfg)
    ball_coaster  = BallCoaster(cfg)
    ball_coaster.set_ball_class(cfg["class_ball"])

    # Equipo por formación inicial (kickoff) — más fiable que el color
    formation     = FormationTeamAssigner(cfg)

    # ── Homografía / vista cenital (Requisito 4) ──────────────────────
    field_template    = create_field_template(cfg["field_template_size"])
    topdown_heatmaps  = TopDownHeatmaps(cfg["field_template_size"], cfg)
    homography_M: np.ndarray | None = None   # se actualiza cada N frames
    smoothed_corners = [None]                 # EMA de las 4 esquinas (contenedor)

    # Escala métrica: metros por píxel del lienzo cenital (invariante a la
    # rotación, usando la diagonal del campo real vs. la del lienzo).
    _fw, _fh = cfg["field_template_size"]
    _rw, _rh = cfg["field_real_m"]
    METERS_PER_PIXEL = float(np.hypot(_rw, _rh) / np.hypot(_fw, _fh))
    print(f"[INFO] Escala cenital: {METERS_PER_PIXEL*100:.3f} cm/px "
          f"(campo {_rw}×{_rh} m sobre lienzo {_fw}×{_fh} px)")

    # ── Info de video ──────────────────────────────────────────────────
    video_info   = sv.VideoInfo.from_video_path(cfg["video_input"])
    W, H         = video_info.width, video_info.height
    total_frames = video_info.total_frames
    print(f"[INFO] Video: {W}×{H} px | {video_info.fps} fps | {total_frames} frames")

    # ── Video de PURA HOMOGRAFÍA (vista cenital animada) ──────────────
    from collections import deque
    tactical_trails: dict = defaultdict(lambda: deque(maxlen=cfg.get("tactical_trail", 40)))
    tactical_writer = None
    if cfg.get("tactical_enabled"):
        scale = cfg.get("tactical_scale", 1)
        tw, th = _fw * scale, _fh * scale
        tac_path = safe_output_path(cfg["tactical_video_output"])
        Path(tac_path).parent.mkdir(parents=True, exist_ok=True)
        tactical_writer = cv2.VideoWriter(
            str(tac_path), cv2.VideoWriter_fourcc(*"mp4v"),
            video_info.fps, (tw, th))
        print(f"[INFO] Video cenital → {tac_path}  ({tw}×{th})")

    frame_idx = 0

    # ─────────────────────────────────────────────────────────────────
    # CALLBACK POR FRAME
    # ─────────────────────────────────────────────────────────────────
    def process_frame(frame: np.ndarray, _: int) -> np.ndarray:
        nonlocal teams_initialized, frame_idx, homography_M
        frame_idx += 1

        # ── 1. Detección YOLO (Requisito 1 — modelo propio) ───────────
        # Se invoca con el umbral MÁS BAJO (el del balón) para no perder al
        # balón; luego se filtra cada clase con su propio umbral.
        results = yolo(
            frame,
            conf=min(cfg["confidence"], cfg["ball_confidence"]),
            iou=cfg["iou_threshold"],
            imgsz=cfg.get("infer_imgsz", 640),   # mayor resolución = mejor en objetos chicos
            device=cfg["device"],
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        is_robot = (detections.class_id == cfg["class_robot"]) & \
                   (detections.confidence >= cfg["confidence"])
        is_ball  = (detections.class_id == cfg["class_ball"]) & \
                   (detections.confidence >= cfg["ball_confidence"])
        detections = detections[is_robot | is_ball]

        # ── 2. Filtro ROI dinámico por color (Requisito 2 — CRÍTICO) ──
        # Cámara móvil ⇒ máscara verde recalculada en CADA frame (sin
        # polígono fijo). El balón NUNCA se descarta por el filtro.
        field_mask = get_field_mask(frame, cfg)
        detections = filter_robots_by_field_mask(detections, field_mask, cfg["class_ball"])

        # ── 2b. Respaldo del balón por color si YOLO no lo encontró ────
        ball_present = bool(np.any(detections.class_id == cfg["class_ball"])) \
            if len(detections) else False
        if cfg["ball_hsv_fallback"] and not ball_present:
            ball_hsv = detect_ball_hsv(frame, field_mask, cfg)
            if len(ball_hsv) > 0:
                if len(detections):
                    # Concatenar a mano (sv.Detections.merge falla si las
                    # detecciones de YOLO traen 'class_name' en .data y la del
                    # balón HSV no). Aquí sólo se necesitan xyxy/conf/class_id;
                    # máscara y tracker_id se asignan después.
                    detections = sv.Detections(
                        xyxy=np.vstack([detections.xyxy, ball_hsv.xyxy]),
                        confidence=np.concatenate(
                            [detections.confidence, ball_hsv.confidence]),
                        class_id=np.concatenate(
                            [detections.class_id, ball_hsv.class_id]),
                    )
                else:
                    detections = ball_hsv

        if len(detections) == 0:
            return frame

        # ── 3. Tracking ByteTrack ────────────────────────────────────
        detections = tracker.update_with_detections(detections)

        robot_dets = detections[detections.class_id == cfg["class_robot"]]
        ball_dets  = detections[detections.class_id == cfg["class_ball"]]

        # ── 3b. Inercia de la pelota ante oclusiones cortas ────────────
        ball_dets = ball_coaster.update(ball_dets, frame_idx)

        # ── 4. Equipos (k-means) + ROSTER FIJO (máx. 4 robots, 2/equipo)─
        if not teams_initialized and len(robot_dets) >= 2:
            team_assigner.initialize(frame, robot_dets)
            teams_initialized = True

        team_ids = None
        if len(robot_dets) > 0:
            team_ids_raw = (team_assigner.assign(frame, robot_dets)
                            if teams_initialized else None)
            # Roster: descarta detecciones espurias (>4) y estabiliza IDs a
            # través de oclusiones. Devuelve robots filtrados + equipos.
            robot_dets, team_ids = roster.update(robot_dets, team_ids_raw, frame_idx)
            if len(team_ids) == 0:
                team_ids = None

        # Reconstruir 'detections' = robots filtrados + balón (para SAM/dibujo)
        detections = _combine_dets(robot_dets, ball_dets)

        # ── 5. Segmentación (capa visual) ──────────────────────────────
        # seg_detections = cajas de YOLO (con tracker_id) para dibujar y para
        # las trayectorias. Las MÁSCARAS son aparte (sam3_masks o de SAM2).
        seg_detections = detections
        sam3_masks_now = None
        if sam is not None and len(detections) > 0:
            if cfg.get("seg_backend") == "sam3_hf":
                # SAM 3 por TEXTO, cada N frames (lento). En los intermedios se
                # reusa la última máscara. Las estadísticas NO dependen de esto.
                if frame_idx % cfg.get("sam3_every_n", 15) == 0:
                    try:
                        partes = []
                        for prompt in cfg.get("sam3_prompts", ["soccer robot", "orange ball"]):
                            d = sam.segment(frame, prompt,
                                            threshold=cfg.get("sam3_threshold", 0.5))
                            if len(d) > 0:
                                partes.append(d)
                        sam3_masks_now = _concat_masks(partes) if partes else None
                    except Exception as e:
                        if frame_idx == cfg.get("sam3_every_n", 15):
                            print(f"[WARN] SAM 3 segmentación falló: {e}")
                        sam3_masks_now = None
                    last_sam3_masks[0] = sam3_masks_now
                else:
                    sam3_masks_now = last_sam3_masks[0]   # reusar última máscara
            else:
                # SAM 2.1 / SAM 3 box-prompt (Ultralytics) — máscara por caja
                boxes = detections.xyxy[: cfg["sam_max_boxes_per_frame"]]
                try:
                    sam_results    = sam(frame, bboxes=boxes, verbose=False)[0]
                    seg_detections = sv.Detections.from_ultralytics(sam_results)
                    if len(seg_detections) == len(detections):
                        seg_detections.tracker_id = detections.tracker_id
                        seg_detections.class_id   = detections.class_id
                    else:
                        seg_detections = detections
                except Exception:
                    seg_detections = detections

        # ── 6. Homografía → coordenadas cenitales (Requisito 4) ────────
        # Se calcula ANTES de las estadísticas para alimentarlas con
        # posiciones en METROS reales (distancia, velocidad, posesión).
        if frame_idx % cfg["homography_recalc_interval"] == 0:
            corners = detect_field_corners(field_mask, cfg)
            if corners is not None:
                # Suavizado temporal (EMA) de las esquinas → evita el "temblor"
                # que inflaba velocidades/distancias y rompía la posesión.
                a = cfg.get("homography_smooth_alpha", 0.25)
                if smoothed_corners[0] is None:
                    smoothed_corners[0] = corners.astype(np.float32)
                else:
                    smoothed_corners[0] = (a * corners +
                                           (1 - a) * smoothed_corners[0]).astype(np.float32)
                homography_M = build_field_homography(
                    smoothed_corners[0], cfg["field_template_size"])
            # Si no se detectan esquinas válidas (oclusión, cancha fuera de
            # cuadro), se conserva la última homografía válida.

        robot_topdown: dict[int, tuple] = {}
        ball_topdown = None
        if homography_M is not None:
            if robot_dets.tracker_id is not None and len(robot_dets) > 0:
                centers = np.stack([
                    (robot_dets.xyxy[:, 0] + robot_dets.xyxy[:, 2]) / 2,
                    (robot_dets.xyxy[:, 1] + robot_dets.xyxy[:, 3]) / 2,
                ], axis=1)
                warped = warp_points(centers, homography_M)
                for tid, (wx, wy) in zip(robot_dets.tracker_id, warped):
                    robot_topdown[int(tid)] = (float(wx), float(wy))
                    topdown_heatmaps.add_robot(int(tid), float(wx), float(wy))

            if len(ball_dets) > 0:
                best_idx = int(np.argmax(ball_dets.confidence))
                bbox     = ball_dets.xyxy[best_idx]
                bx, by   = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                wb = warp_points(np.array([[bx, by]]), homography_M)
                ball_topdown = (float(wb[0, 0]), float(wb[0, 1]))
                topdown_heatmaps.add_ball(ball_topdown[0], ball_topdown[1])

        # ── 6b. EQUIPO POR FORMACIÓN INICIAL (kickoff) ─────────────────
        # Más confiable que el color: en el saque cada equipo está de su
        # lado. Si la formación da una separación clara se usa; si no
        # (video que no empieza en el saque), se cae al color (team_ids).
        if cfg.get("team_mode", "auto") in ("auto", "formation"):
            formation.update(robot_topdown, frame_idx)
            if robot_dets.tracker_id is not None and len(robot_dets) > 0:
                nuevos = []
                for i, tid in enumerate(robot_dets.tracker_id):
                    ft = formation.team_of(int(tid), robot_topdown.get(int(tid)))
                    if ft == -1:                      # formación aún no decide
                        ft = int(team_ids[i]) if team_ids is not None else -1
                    nuevos.append(ft if ft in (0, 1) else 0)
                team_ids = np.array(nuevos, dtype=int)

        # ── 7. Estadísticas en METROS (distancia, velocidad, posesión,
        #        eventos: pases / intercepciones / tiros) ───────────────
        stats.update(
            frame, seg_detections, robot_dets, ball_dets, team_ids, frame_idx,
            robot_topdown=robot_topdown or None,
            ball_topdown=ball_topdown,
            meters_per_pixel=METERS_PER_PIXEL,
        )

        # ── 7b. Video de PURA HOMOGRAFÍA (vista cenital animada) ────────
        if tactical_writer is not None:
            tid_team = {}
            if team_ids is not None and robot_dets.tracker_id is not None:
                tid_team = {int(t): int(tm)
                            for t, tm in zip(robot_dets.tracker_id, team_ids)}
            # Equipos estables del roster (para robots sin team_ids este frame)
            for t, tm in roster.get_team_map().items():
                tid_team.setdefault(int(t), int(tm))
            for tid, (x, y) in robot_topdown.items():
                tactical_trails[tid].append((x, y))
            if ball_topdown is not None:
                tactical_trails["ball"].append(ball_topdown)
            tac_frame = render_tactical_frame(
                field_template, robot_topdown, ball_topdown,
                tid_team, tactical_trails, cfg)
            tactical_writer.write(tac_frame)

        # ── 8. Render del frame anotado ─────────────────────────────────
        annotated = frame.copy()

        # Salvaguarda: el anotador de trazas exige tracker_id en TODAS las
        # detecciones (la pelota navegada por inercia podría no traerlo).
        if len(seg_detections) and seg_detections.tracker_id is None:
            seg_detections.tracker_id = np.full(len(seg_detections), -1, dtype=int)

        # Máscaras: SAM 3 (texto) si está activo; si no, las de SAM 2.1/box.
        if cfg.get("seg_backend") == "sam3_hf":
            if sam3_masks_now is not None and len(sam3_masks_now) > 0:
                annotated = sam3_mask_annotator.annotate(annotated, sam3_masks_now)
        elif sam is not None and seg_detections.mask is not None:
            annotated = mask_annotator.annotate(annotated, seg_detections)

        annotated = trace_annotator.annotate(annotated, seg_detections)

        colors  = _build_color_palette(seg_detections, team_ids, robot_dets, cfg)
        box_ann = sv.BoxAnnotator(thickness=2, color=colors)
        annotated = box_ann.annotate(annotated, seg_detections)

        labels = _build_labels(seg_detections, team_ids, robot_dets, cfg, stats)
        annotated = label_annotator.annotate(annotated, seg_detections, labels=labels)

        # ── Requisito 5: funciones de utils, llamadas de forma estándar ──
        # Las superposiciones por equipo (posesión, minimapa) solo se dibujan
        # si video_show_teams=True. En modo simple, el video muestra únicamente
        # robots y pelota + estadísticas del balón.
        if cfg.get("video_show_teams", False):
            annotated = draw_possession_bar(annotated, stats.possession, W, H)
            annotated = draw_mini_map(annotated, seg_detections, team_ids, robot_dets, W, H)
        annotated = draw_ball_stats_overlay(annotated, stats, frame_idx, video_info.fps)

        if frame_idx % 50 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  [PROGRESO] {frame_idx}/{total_frames} frames ({pct:.1f}%)")

        return annotated

    # ── Guardado seguro del video (Requisito 3) ───────────────────────
    Path(cfg["video_output"]).parent.mkdir(parents=True, exist_ok=True)
    output_path = safe_output_path(cfg["video_output"])
    if output_path != Path(cfg["video_output"]):
        print(f"[INFO] Archivo ya existe → usando nombre alternativo: {output_path}")

    try:
        sv.process_video(
            source_path=cfg["video_input"],
            target_path=str(output_path),
            callback=process_frame,
        )
        print(f"[OK] Video guardado → {output_path}")
    except Exception as e:
        print(f"[ERROR] Falló el guardado del video: {e}")
        raise
    finally:
        if tactical_writer is not None:
            tactical_writer.release()
            print(f"[OK] Video cenital (homografía) guardado → "
                  f"{cfg['tactical_video_output']}")

    # ── Reconciliar equipos: ningún robot debe quedar en -1 ───────────
    # Prioridad: formación inicial (más fiable) → roster/color como respaldo.
    for tid, t in formation.get_team_map().items():
        stats._team_map[int(tid)] = int(t)
    for tid, t in roster.get_team_map().items():
        stats._team_map.setdefault(int(tid), int(t))

    # ── Post-proceso: heatmaps cenitales + dashboard de resumen ───────
    print("[INFO] Generando visualizaciones finales…")

    out_root = cfg.get("out_root", "output")

    # Heatmaps en vista cenital (Requisito 4) — sobre la plantilla 2D limpia
    draw_homography_heatmaps(topdown_heatmaps, field_template, stats,
                             out_dir=f"{out_root}/heatmaps")

    # Dashboard de resumen (Requisito 5 — función estándar de utils)
    cap = cv2.VideoCapture(cfg["video_input"])
    ret, bg_frame = cap.read()
    cap.release()
    if ret:
        create_summary_visualization(bg_frame, stats, cfg, video_info)

    # ── Exportar JSON de estadísticas (también con guardado seguro) ───
    stats_data = None
    try:
        stats_path = safe_output_path(cfg["stats_output"])
        stats_data = stats.export_json(str(stats_path))
        print(f"[OK] Estadísticas exportadas → {stats_path}")
    except Exception as e:
        print(f"[ERROR] Falló la exportación de estadísticas: {e}")

    # ── Visualizaciones avanzadas (Voronoi, timeline, grafo de pases) ──
    try:
        from utils.advanced_viz import generate_all_advanced
        generate_all_advanced(stats, cfg, out_dir=f"{out_root}/avanzadas")
        print(f"[OK] Visualizaciones avanzadas → {out_root}/avanzadas/")
    except Exception as e:
        print(f"[WARN] Visualizaciones avanzadas omitidas: {e}")

    # ── Dashboard HTML interactivo (narrativa del partido) ────────────
    try:
        from dashboard import build_dashboard
        if stats_data is not None:
            build_dashboard(stats_data, f"{out_root}/dashboard.html")
            print(f"[OK] Dashboard interactivo → {out_root}/dashboard.html")
    except Exception as e:
        print(f"[WARN] Dashboard omitido: {e}")

    # ── Interfaz de usuario (visor de resultados con pestañas) ────────
    try:
        from generar_interfaz import build_interface
        ui = build_interface(out_root)
        print(f"[OK] Interfaz de usuario → {ui}  (ábrela en el navegador)")
    except Exception as e:
        print(f"[WARN] Interfaz omitida: {e}")

    print("\n[DONE] Pipeline completado.")


# ════════════════════════════════════════════════════════════════════════
# HELPERS DE ANOTACIÓN
# ════════════════════════════════════════════════════════════════════════

def _build_color_palette(detections, team_ids, robot_dets, cfg):
    """Paleta dinámica: azul/rojo por equipo, amarillo para el balón."""
    TEAM_COLORS = [
        sv.Color(r=30,  g=120, b=220),   # Equipo A — azul
        sv.Color(r=220, g=50,  b=50),    # Equipo B — rojo
    ]
    BALL_COLOR  = sv.Color(r=255, g=220, b=0)
    ROBOT_COLOR = sv.Color(r=30, g=200, b=120)   # un solo color para todos los robots
    DEFAULT     = sv.Color(r=180, g=180, b=180)

    show_teams = cfg.get("video_show_teams", False)
    robot_id_to_team = {}
    if team_ids is not None and robot_dets.tracker_id is not None:
        for tid, t in zip(robot_dets.tracker_id, team_ids):
            robot_id_to_team[int(tid)] = int(t)

    colors = []
    for i, cls in enumerate(detections.class_id):
        if cls == cfg["class_ball"]:
            colors.append(BALL_COLOR)
        elif not show_teams:
            colors.append(ROBOT_COLOR)            # modo simple: solo "Robot"
        elif detections.tracker_id is not None:
            tid  = int(detections.tracker_id[i])
            team = robot_id_to_team.get(tid, -1)
            colors.append(TEAM_COLORS[team] if team in (0, 1) else DEFAULT)
        else:
            colors.append(DEFAULT)

    return sv.ColorPalette(colors=colors)


def _build_labels(detections, team_ids, robot_dets, cfg, stats):
    """Etiquetas de texto: equipo + distancia para robots, velocidad para el balón."""
    robot_id_to_team = {}
    if team_ids is not None and robot_dets.tracker_id is not None:
        for tid, t in zip(robot_dets.tracker_id, team_ids):
            robot_id_to_team[int(tid)] = int(t)

    TEAM_NAMES  = ["Azul", "Rojo"]
    labels      = []
    tracker_ids = (
        detections.tracker_id
        if detections.tracker_id is not None
        else [None] * len(detections)
    )
    show_teams = cfg.get("video_show_teams", False)
    for cls, tid in zip(detections.class_id, tracker_ids):
        if cls == cfg["class_ball"]:
            labels.append("Pelota")
        else:
            tid_int = int(tid) if tid is not None else -1
            if show_teams:
                team  = robot_id_to_team.get(tid_int, -1)
                tname = TEAM_NAMES[team] if team in (0, 1) else "?"
                dist  = stats.robot_distance.get(tid_int, 0.0)
                labels.append(f"{tname} #{tid_int} {dist:.0f}m")
            else:
                labels.append(f"Robot #{tid_int}")   # solo Robot + ID

    return labels


def run_all_videos(base_cfg: dict = CONFIG, videos_dir: str = "assets"):
    """Procesa TODOS los videos de la carpeta `videos_dir`. Cada video genera
    su propia subcarpeta de resultados en output/<nombre_del_video>/ para que
    no se sobrescriban entre sí."""
    exts = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
    videos = sorted(p for p in Path(videos_dir).iterdir()
                    if p.suffix.lower() in exts)
    if not videos:
        print(f"[ERROR] No hay videos en '{videos_dir}/' (extensiones: {exts}).")
        return
    print(f"[INFO] {len(videos)} video(s) encontrados en '{videos_dir}/':")
    for v in videos:
        print(f"   · {v.name}")

    for i, video in enumerate(videos, 1):
        stem = video.stem
        out_root = f"output/{stem}"
        print("\n" + "═" * 70)
        print(f"[{i}/{len(videos)}] Procesando: {video.name}  →  {out_root}/")
        print("═" * 70)
        cfg = dict(base_cfg)   # copia independiente (estado fresco por video)
        cfg["video_input"]           = str(video)
        cfg["out_root"]              = out_root
        cfg["video_output"]          = f"{out_root}/partido_analizado.mp4"
        cfg["stats_output"]          = f"{out_root}/estadisticas.json"
        cfg["tactical_video_output"] = f"{out_root}/partido_homografia.mp4"
        try:
            run_pipeline(cfg)
        except Exception as e:
            print(f"[ERROR] Falló el procesamiento de {video.name}: {e}")
    print("\n[DONE] Todos los videos procesados. Resultados en output/<nombre>/")


if __name__ == "__main__":
    import sys
    if "--all" in sys.argv:
        # Procesa TODOS los videos de assets/  →  python pipeline.py --all
        run_all_videos(CONFIG, videos_dir="assets")
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Un video específico  →  python pipeline.py assets/otro_partido.mp4
        cfg = dict(CONFIG)
        stem = Path(sys.argv[1]).stem
        cfg["video_input"]           = sys.argv[1]
        cfg["out_root"]              = f"output/{stem}"
        cfg["video_output"]          = f"output/{stem}/partido_analizado.mp4"
        cfg["stats_output"]          = f"output/{stem}/estadisticas.json"
        cfg["tactical_video_output"] = f"output/{stem}/partido_homografia.mp4"
        run_pipeline(cfg)
    else:
        # Comportamiento original: el video de CONFIG (assets/partido.mp4)
        run_pipeline()
