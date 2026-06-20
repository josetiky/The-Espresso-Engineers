"""
dashboard.py — Genera un dashboard HTML interactivo (Requisito 3.5.2)
────────────────────────────────────────────────────────────────────────
Convierte output/estadisticas.json en una página autocontenida que narra
la historia del partido: posesión, eventos, distancias por robot, velocidad
y red de pases. Usa Chart.js desde CDN (no requiere instalación).

Uso:
    python dashboard.py                 # lee output/estadisticas.json
    python dashboard.py ruta.json salida.html
"""

import json
import sys
from pathlib import Path


def build_dashboard(data: dict, out_path: str = "output/dashboard.html"):
    """Construye el HTML a partir del dict de estadísticas y lo guarda."""
    poss = data.get("posesion", {})
    balon = data.get("balon", {})
    robots = data.get("robots", {})
    eventos = data.get("eventos", [])
    ev_resumen = data.get("eventos_resumen", {})
    red = data.get("red_de_pases", [])

    # Series para gráficas
    robot_ids = list(robots.keys())
    robot_dist = [robots[r].get("distancia_m", 0) for r in robot_ids]
    robot_team = [robots[r].get("equipo", -1) for r in robot_ids]
    robot_vmax = [robots[r].get("velocidad_max_kmh", 0) for r in robot_ids]
    bar_colors = ["#1E78DC" if t == 0 else "#DC3232" if t == 1 else "#9aa0a6"
                  for t in robot_team]

    n_azul = sum(1 for t in robot_team if t == 0)
    n_rojo = sum(1 for t in robot_team if t == 1)

    # Filas de eventos
    ev_rows = ""
    badge = {"pase": "#1E78DC", "intercepcion": "#e8a72e",
             "tiro_a_gol": "#DC3232"}
    for e in eventos[:200]:
        col = badge.get(e.get("tipo"), "#666")
        det = ""
        if "robot_origen" in e:
            det = f"#{e.get('robot_origen')}→#{e.get('robot_destino', '–')}"
        ev_rows += (
            f"<tr><td>{e.get('tiempo_seg','')}s</td>"
            f"<td><span class='badge' style='background:{col}'>{e.get('tipo','')}</span></td>"
            f"<td>{'Azul' if e.get('equipo')==0 else 'Rojo' if e.get('equipo')==1 else '–'}</td>"
            f"<td>{det}</td></tr>"
        )
    if not ev_rows:
        ev_rows = "<tr><td colspan='4' style='text-align:center;color:#888'>Sin eventos registrados</td></tr>"

    red_rows = ""
    for r in red[:20]:
        red_rows += (f"<tr><td>#{r['origen']}</td><td>#{r['destino']}</td>"
                     f"<td>{r['pases']}</td></tr>")
    if not red_rows:
        red_rows = "<tr><td colspan='3' style='text-align:center;color:#888'>Sin pases registrados</td></tr>"

    html = _TEMPLATE.format(
        poss_azul=poss.get("equipo_azul", 0),
        poss_rojo=poss.get("equipo_rojo", 0),
        poss_neutro=poss.get("sin_posesion", 0),
        ball_dist=balon.get("distancia_total_m", 0),
        ball_vmax=balon.get("velocidad_maxima_kmh", 0),
        n_eventos=len(eventos),
        n_pases=ev_resumen.get("pase", 0),
        n_inter=ev_resumen.get("intercepcion", 0),
        n_tiros=ev_resumen.get("tiro_a_gol", 0),
        n_azul=n_azul, n_rojo=n_rojo,
        robot_ids=json.dumps([f"#{r}" for r in robot_ids]),
        robot_dist=json.dumps(robot_dist),
        robot_vmax=json.dumps(robot_vmax),
        bar_colors=json.dumps(bar_colors),
        ev_rows=ev_rows,
        red_rows=red_rows,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Copa FutBotMX — Análisis del Partido</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{ --azul:#1E78DC; --rojo:#DC3232; --bg:#0f1116; --card:#191c24; --tx:#e8eaed; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--tx); }}
  header {{ padding:24px 32px; background:linear-gradient(110deg,#11131a,#1d2330);
           border-bottom:2px solid #2a2f3a; }}
  header h1 {{ margin:0; font-size:24px; }}
  header p {{ margin:4px 0 0; color:#9aa0a6; font-size:14px; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:24px 32px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; }}
  .kpi {{ background:var(--card); border:1px solid #262b36; border-radius:14px; padding:18px; }}
  .kpi .v {{ font-size:30px; font-weight:700; }}
  .kpi .l {{ color:#9aa0a6; font-size:13px; margin-top:4px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:22px; }}
  .card {{ background:var(--card); border:1px solid #262b36; border-radius:14px; padding:20px; }}
  .card h2 {{ margin:0 0 14px; font-size:16px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ text-align:left; padding:7px 8px; border-bottom:1px solid #262b36; }}
  th {{ color:#9aa0a6; font-weight:600; }}
  .badge {{ color:white; padding:2px 9px; border-radius:20px; font-size:11px; text-transform:capitalize; }}
  .scroll {{ max-height:330px; overflow:auto; }}
  @media (max-width:820px){{ .grid{{grid-template-columns:1fr;}} }}
</style>
</head>
<body>
<header>
  <h1>⚽🤖 Copa FutBotMX — Análisis del Partido</h1>
  <p>Visión por Computadora · Detección + Tracking + Homografía + SAM · Métricas en metros reales</p>
</header>
<div class="wrap">

  <div class="kpis">
    <div class="kpi"><div class="v" style="color:var(--azul)">{poss_azul}%</div><div class="l">Posesión Azul</div></div>
    <div class="kpi"><div class="v" style="color:var(--rojo)">{poss_rojo}%</div><div class="l">Posesión Rojo</div></div>
    <div class="kpi"><div class="v">{poss_neutro}%</div><div class="l">Sin posesión</div></div>
    <div class="kpi"><div class="v">{ball_dist} m</div><div class="l">Distancia del balón</div></div>
    <div class="kpi"><div class="v">{ball_vmax}</div><div class="l">Vel. máx balón (km/h)</div></div>
    <div class="kpi"><div class="v">{n_eventos}</div><div class="l">Eventos detectados</div></div>
  </div>

  <div class="kpis" style="margin-top:14px">
    <div class="kpi"><div class="v">{n_pases}</div><div class="l">Pases</div></div>
    <div class="kpi"><div class="v">{n_inter}</div><div class="l">Intercepciones</div></div>
    <div class="kpi"><div class="v">{n_tiros}</div><div class="l">Tiros a gol</div></div>
    <div class="kpi"><div class="v" style="color:var(--azul)">{n_azul}</div><div class="l">Robots Azul</div></div>
    <div class="kpi"><div class="v" style="color:var(--rojo)">{n_rojo}</div><div class="l">Robots Rojo</div></div>
  </div>

  <div class="grid">
    <div class="card"><h2>Posesión por equipo</h2><canvas id="possChart"></canvas></div>
    <div class="card"><h2>Distancia recorrida por robot (m)</h2><canvas id="distChart"></canvas></div>
  </div>

  <div class="grid">
    <div class="card"><h2>Velocidad máxima por robot (km/h)</h2><canvas id="velChart"></canvas></div>
    <div class="card"><h2>Red de pases (robot → robot)</h2>
      <div class="scroll"><table><thead><tr><th>Origen</th><th>Destino</th><th>Pases</th></tr></thead>
      <tbody>{red_rows}</tbody></table></div>
    </div>
  </div>

  <div class="card" style="margin-top:20px"><h2>Cronología de eventos</h2>
    <div class="scroll"><table><thead><tr><th>Tiempo</th><th>Tipo</th><th>Equipo</th><th>Robots</th></tr></thead>
    <tbody>{ev_rows}</tbody></table></div>
  </div>

  <p style="color:#666;font-size:12px;margin-top:24px">
    Generado automáticamente por <code>dashboard.py</code> · Copa FutBotMX 2026 · Categoría Amateur
  </p>
</div>

<script>
Chart.defaults.color = '#9aa0a6';
Chart.defaults.borderColor = '#262b36';

new Chart(document.getElementById('possChart'), {{
  type:'doughnut',
  data:{{ labels:['Azul','Rojo','Sin posesión'],
    datasets:[{{ data:[{poss_azul},{poss_rojo},{poss_neutro}],
      backgroundColor:['#1E78DC','#DC3232','#9aa0a6'] }}] }},
  options:{{ plugins:{{ legend:{{ position:'bottom' }} }} }}
}});

new Chart(document.getElementById('distChart'), {{
  type:'bar',
  data:{{ labels:{robot_ids},
    datasets:[{{ label:'metros', data:{robot_dist}, backgroundColor:{bar_colors} }}] }},
  options:{{ plugins:{{ legend:{{ display:false }} }},
    scales:{{ y:{{ beginAtZero:true }} }} }}
}});

new Chart(document.getElementById('velChart'), {{
  type:'bar',
  data:{{ labels:{robot_ids},
    datasets:[{{ label:'km/h', data:{robot_vmax}, backgroundColor:{bar_colors} }}] }},
  options:{{ plugins:{{ legend:{{ display:false }} }},
    scales:{{ y:{{ beginAtZero:true }} }} }}
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "output/estadisticas.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "output/dashboard.html"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    out = build_dashboard(data, dst)
    print(f"[OK] Dashboard generado → {out}")
