"""
generar_interfaz.py — Interfaz de usuario (visor de resultados) en HTML.
────────────────────────────────────────────────────────────────────────
Genera un único archivo `interfaz.html` DENTRO de la carpeta de resultados
que junta TODO el análisis en una página con pestañas:

   · Resumen      → KPIs (posesión, balón, eventos, nº robots)
   · Videos       → partido analizado + vista cenital (homografía)
   · Posesión     → dona de posesión + barras por robot (Chart.js)
   · Eventos      → cronología (pases / intercepciones / tiros) + red de pases
   · Mapas calor  → galería de heatmaps
   · Avanzadas    → Voronoi, timeline, red de pases, trayectoria del balón

Se abre en el navegador SIN servidor ni instalación. Enlaza los archivos por
ruta relativa, así que debe guardarse en la MISMA carpeta que los resultados.

Uso:
    python generar_interfaz.py                 # carpeta 'output'
    python generar_interfaz.py output/partido  # un partido concreto
"""

import json
import sys
from pathlib import Path


def _img_gallery(folder: Path, base: Path, titulo_map=None):
    if not folder.exists():
        return ""
    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        return ""
    cards = ""
    for p in pngs:
        rel = p.relative_to(base).as_posix()
        nombre = p.stem.replace("_", " ").title()
        cards += (f'<figure class="shot"><img src="{rel}" loading="lazy" '
                  f'onclick="zoom(this)"><figcaption>{nombre}</figcaption></figure>')
    return cards


def build_interface(out_dir: str = "output") -> str:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    # ── Datos ──────────────────────────────────────────────────────────
    stats_file = base / "estadisticas.json"
    data = {}
    if stats_file.exists():
        data = json.loads(stats_file.read_text(encoding="utf-8"))
    poss = data.get("posesion", {})
    balon = data.get("balon", {})
    robots = data.get("robots", {})
    eventos = data.get("eventos", [])
    ev_res = data.get("eventos_resumen", {})
    red = data.get("red_de_pases", [])

    rids = list(robots.keys())
    rdist = [robots[r].get("distancia_m", 0) for r in rids]
    rvel = [robots[r].get("velocidad_max_kmh", 0) for r in rids]

    # ── Media existente (rutas relativas) ──────────────────────────────
    def rel_if(name):
        p = base / name
        return p.name if p.exists() else None
    vid_analizado = rel_if("partido_analizado.mp4")
    vid_cenital = rel_if("partido_homografia.mp4")
    resumen_png = rel_if("resumen_partido.png")

    def video_block(src, titulo):
        if not src:
            return f'<p class="vacio">({titulo}: aún no generado)</p>'
        return (f'<div class="vid"><h3>{titulo}</h3>'
                f'<video src="{src}" controls preload="metadata"></video></div>')

    heat = _img_gallery(base / "heatmaps", base)
    avanz = _img_gallery(base / "avanzadas", base)

    # ── Filas de eventos / red de pases ────────────────────────────────
    badge = {"pase": "#1E78DC", "intercepcion": "#e8a72e", "tiro_a_gol": "#DC3232"}
    ev_rows = ""
    for e in eventos[:300]:
        col = badge.get(e.get("tipo"), "#666")
        det = f"#{e.get('robot_origen','–')}→#{e.get('robot_destino','–')}" \
            if "robot_origen" in e else ""
        eq = {0: "Azul", 1: "Rojo"}.get(e.get("equipo"), "–")
        ev_rows += (f"<tr><td>{e.get('tiempo_seg','')}s</td>"
                    f"<td><span class='badge' style='background:{col}'>{e.get('tipo','')}</span></td>"
                    f"<td>{eq}</td><td>{det}</td></tr>")
    if not ev_rows:
        ev_rows = "<tr><td colspan=4 class='vacio'>Sin eventos</td></tr>"
    red_rows = "".join(
        f"<tr><td>#{r['origen']}</td><td>#{r['destino']}</td><td>{r['pases']}</td></tr>"
        for r in red[:30]) or "<tr><td colspan=3 class='vacio'>Sin pases</td></tr>"

    html = _TEMPLATE.format(
        poss_azul=poss.get("equipo_azul", 0), poss_rojo=poss.get("equipo_rojo", 0),
        poss_neutro=poss.get("sin_posesion", 0),
        ball_dist=balon.get("distancia_total_m", 0),
        ball_vmax=balon.get("velocidad_maxima_kmh", 0),
        n_eventos=len(eventos), n_robots=len(rids),
        n_pases=ev_res.get("pase", 0), n_inter=ev_res.get("intercepcion", 0),
        n_tiros=ev_res.get("tiro_a_gol", 0),
        video_analizado=video_block(vid_analizado, "Partido analizado"),
        video_cenital=video_block(vid_cenital, "Vista cenital (homografía)"),
        heat_gallery=heat or "<p class='vacio'>Sin heatmaps</p>",
        avanz_gallery=avanz or "<p class='vacio'>Sin visualizaciones avanzadas</p>",
        ev_rows=ev_rows, red_rows=red_rows,
        rids=json.dumps([f"#{r}" for r in rids]),
        rdist=json.dumps(rdist), rvel=json.dumps(rvel),
    )
    out = base / "interfaz.html"
    out.write_text(html, encoding="utf-8")
    return str(out)


_TEMPLATE = """<!DOCTYPE html><html lang=es><head><meta charset=UTF-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Copa FutBotMX — Interfaz de Análisis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{{--azul:#1E78DC;--rojo:#DC3232;--bg:#0d1016;--card:#171b24;--bd:#262b36;--tx:#e8eaed;--mut:#9aa0a6}}
*{{box-sizing:border-box}}body{{margin:0;font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--tx)}}
header{{padding:22px 30px;background:linear-gradient(110deg,#10131b,#1d2533);border-bottom:2px solid var(--bd)}}
header h1{{margin:0;font-size:22px}}header p{{margin:4px 0 0;color:var(--mut);font-size:13px}}
nav{{display:flex;gap:6px;padding:10px 24px;background:#11141c;border-bottom:1px solid var(--bd);flex-wrap:wrap;position:sticky;top:0;z-index:10}}
nav button{{background:transparent;color:var(--mut);border:1px solid transparent;padding:8px 16px;border-radius:10px;cursor:pointer;font-size:14px}}
nav button.act{{background:var(--card);color:var(--tx);border-color:var(--bd)}}
main{{max-width:1180px;margin:0 auto;padding:24px 30px}}
.tab{{display:none}}.tab.act{{display:block;animation:f .25s}}@keyframes f{{from{{opacity:0}}to{{opacity:1}}}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:13px}}
.kpi{{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:17px}}
.kpi .v{{font-size:28px;font-weight:700}}.kpi .l{{color:var(--mut);font-size:12px;margin-top:3px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:20px;margin-top:18px}}
.card h2,.card h3{{margin:0 0 13px;font-size:16px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}@media(max-width:820px){{.grid2{{grid-template-columns:1fr}}}}
video{{width:100%;border-radius:10px;background:#000;border:1px solid var(--bd)}}
.vid h3{{font-size:14px;margin:0 0 8px;color:var(--mut)}}
.gal{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}}
.shot{{margin:0;background:#0f1218;border:1px solid var(--bd);border-radius:10px;overflow:hidden;cursor:zoom-in}}
.shot img{{width:100%;display:block}}.shot figcaption{{padding:7px 9px;font-size:12px;color:var(--mut)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--bd)}}
th{{color:var(--mut)}}.badge{{color:#fff;padding:2px 9px;border-radius:20px;font-size:11px;text-transform:capitalize}}
.scroll{{max-height:340px;overflow:auto}}.vacio{{color:#666;text-align:center;padding:20px}}
#lb{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:50;align-items:center;justify-content:center}}
#lb img{{max-width:92vw;max-height:92vh;border-radius:10px}}
</style></head><body>
<header><h1>⚽🤖 Copa FutBotMX — Interfaz de Análisis</h1>
<p>Visión por Computadora · Detección + Tracking + Homografía + SAM · Métricas en metros reales</p></header>
<nav>
<button class=act onclick="tab(0,this)">Resumen</button>
<button onclick="tab(1,this)">Videos</button>
<button onclick="tab(2,this)">Posesión</button>
<button onclick="tab(3,this)">Eventos</button>
<button onclick="tab(4,this)">Mapas de calor</button>
<button onclick="tab(5,this)">Avanzadas</button>
</nav>
<main>

<section class="tab act">
  <div class=kpis>
    <div class=kpi><div class=v style="color:var(--azul)">{poss_azul}%</div><div class=l>Posesión Azul</div></div>
    <div class=kpi><div class=v style="color:var(--rojo)">{poss_rojo}%</div><div class=l>Posesión Rojo</div></div>
    <div class=kpi><div class=v>{poss_neutro}%</div><div class=l>Sin posesión</div></div>
    <div class=kpi><div class=v>{ball_dist} m</div><div class=l>Distancia del balón</div></div>
    <div class=kpi><div class=v>{ball_vmax}</div><div class=l>Vel. máx balón (km/h)</div></div>
    <div class=kpi><div class=v>{n_robots}</div><div class=l>Robots</div></div>
    <div class=kpi><div class=v>{n_pases}</div><div class=l>Pases</div></div>
    <div class=kpi><div class=v>{n_inter}</div><div class=l>Intercepciones</div></div>
    <div class=kpi><div class=v>{n_tiros}</div><div class=l>Tiros a gol</div></div>
    <div class=kpi><div class=v>{n_eventos}</div><div class=l>Eventos totales</div></div>
  </div>
</section>

<section class=tab>
  <div class="card grid2">{video_analizado}{video_cenital}</div>
</section>

<section class=tab>
  <div class="card grid2">
    <div><h3>Posesión por equipo</h3><canvas id=cPoss></canvas></div>
    <div><h3>Distancia por robot (m)</h3><canvas id=cDist></canvas></div>
  </div>
  <div class=card><h3>Velocidad máxima por robot (km/h)</h3><canvas id=cVel></canvas></div>
</section>

<section class=tab>
  <div class="card grid2">
    <div><h3>Cronología de eventos</h3><div class=scroll><table>
      <thead><tr><th>Tiempo</th><th>Tipo</th><th>Equipo</th><th>Robots</th></tr></thead>
      <tbody>{ev_rows}</tbody></table></div></div>
    <div><h3>Red de pases (robot → robot)</h3><div class=scroll><table>
      <thead><tr><th>Origen</th><th>Destino</th><th>Pases</th></tr></thead>
      <tbody>{red_rows}</tbody></table></div></div>
  </div>
</section>

<section class=tab><div class=card><h2>Mapas de calor</h2><div class=gal>{heat_gallery}</div></div></section>
<section class=tab><div class=card><h2>Visualizaciones avanzadas</h2><div class=gal>{avanz_gallery}</div></div></section>

</main>
<div id=lb onclick="this.style.display='none'"><img id=lbi></div>
<script>
const tabs=document.querySelectorAll('.tab'),btns=document.querySelectorAll('nav button');
function tab(i,b){{tabs.forEach(t=>t.classList.remove('act'));btns.forEach(x=>x.classList.remove('act'));
  tabs[i].classList.add('act');b.classList.add('act');window.scrollTo(0,0);}}
function zoom(img){{document.getElementById('lbi').src=img.src;document.getElementById('lb').style.display='flex';}}
Chart.defaults.color='#9aa0a6';Chart.defaults.borderColor='#262b36';
new Chart(cPoss,{{type:'doughnut',data:{{labels:['Azul','Rojo','Sin posesión'],
  datasets:[{{data:[{poss_azul},{poss_rojo},{poss_neutro}],backgroundColor:['#1E78DC','#DC3232','#9aa0a6']}}]}},
  options:{{plugins:{{legend:{{position:'bottom'}}}}}}}});
new Chart(cDist,{{type:'bar',data:{{labels:{rids},datasets:[{{label:'m',data:{rdist},backgroundColor:'#1E78DC'}}]}},
  options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true}}}}}}}});
new Chart(cVel,{{type:'bar',data:{{labels:{rids},datasets:[{{label:'km/h',data:{rvel},backgroundColor:'#30c878'}}]}},
  options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true}}}}}}}});
</script></body></html>"""


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "output"
    out = build_interface(target)
    print(f"[OK] Interfaz generada → {out}")
    print("    Ábrela en tu navegador (doble clic).")
