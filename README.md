# 🤖⚽ Copa FutBotMX — Visión por Computadora

> Análisis automático de partidos de **fútbol robótico** con visión por computadora.
> Pipeline: **YOLO (modelo propio) → ByteTrack → Homografía → SAM 3 → Estadísticas**.
> Detección, seguimiento, segmentación con **SAM 3 (Meta)** mediante *prompts de
> texto*, métricas en **metros reales**, eventos del juego e interfaz de resultados.
> Optimizado para **Apple Silicon (MPS)**.

---

## 📋 Índice

1. [Descripción del enfoque](#descripción-del-enfoque)
2. [Arquitectura de la solución](#arquitectura-de-la-solución)
3. [Innovación sobre SAM 3](#innovación-sobre-sam-3)
4. [Requisitos de hardware y software](#requisitos-de-hardware-y-software)
5. [Instalación](#instalación)
6. [Cómo reproducir los resultados](#cómo-reproducir-los-resultados)
7. [Herramientas de calibración](#herramientas-de-calibración)
8. [Visualizaciones y estadísticas](#visualizaciones-y-estadísticas)
9. [Relevancia deportiva: ¿para qué sirven estas métricas?](#relevancia-deportiva-para-qué-sirven-estas-métricas)
10. [Problemas encontrados y soluciones](#problemas-encontrados-y-soluciones)
11. [Estructura del repositorio](#estructura-del-repositorio)
12. [Documentación técnica](#documentación-técnica)
13. [Reel de Instagram](#reel-de-instagram)
14. [Licencia y créditos](#licencia-y-créditos)

> 📖 Guía de instalación detallada para principiantes: **[`docs/INSTALACION.md`](docs/INSTALACION.md)**
> 📐 Fundamentos matemáticos (ecuaciones): **[`docs/METODOLOGIA.md`](docs/METODOLOGIA.md)**
> ✅ Checklist de entrega: **[`docs/CHECKLIST_ENTREGA.md`](docs/CHECKLIST_ENTREGA.md)**

---

## Descripción del enfoque

### Problema
Analizar de forma automática los videos de los partidos de la Copa FutBotMX para
extraer información táctica: posición y trayectoria de los robots, posesión del
balón, eventos del juego (pases, intercepciones, tiros) y mapas de actividad.

### Solución
Un pipeline modular que combina varios modelos, cada uno en lo que es mejor:

- **YOLO entrenado con datos propios** (`futbotmx_v2.pt`) detecta robots y pelota
  en cada frame (mAP@50 ≈ 0.89).
- **ByteTrack** sigue a cada objeto entre frames, reforzado con un **roster fijo de
  4 robots** (regla del juego: 2 por equipo) que mantiene la identidad a través de
  oclusiones y descarta detecciones espurias.
- **Homografía** proyecta la cancha a una **vista cenital** con la proporción real
  del campo (182 × 243 cm), lo que permite medir en **metros reales**.
- **SAM 3 (Meta)** segmenta los elementos del partido mediante **prompts de texto /
  vocabulario abierto** ("soccer robot", "orange ball").
- Una capa de **analítica** calcula posesión, distancia, velocidad y eventos, y
  genera mapas de calor, diagrama de Voronoi, red de pases, dashboard e interfaz.

### Decisiones de diseño clave
| Decisión | Justificación |
|---|---|
| Modelo YOLO propio (2 clases) | Robots y pelota; los robots no pueden ser naranja/amarillo/azul (regla 3.7) |
| Roster fijo de 4 robots (2/equipo) | Impone la regla del juego; estabiliza IDs ante oclusiones |
| Equipo por **formación inicial** (saque) | Más fiable que el color: en el saque cada equipo está de su lado |
| Pelota por **color (HSV)** como respaldo | La regla 3.7 garantiza que lo naranja es la pelota |
| Homografía con **suavizado de esquinas** | Evita el "temblor" que inflaba velocidades y distancias |
| Métricas en metros + **suavizado + banda muerta** | Velocidades/distancias realistas, sin ruido de detección |
| SAM 3 por **texto** cada N frames | Segmentación de vocabulario abierto sin frenar el análisis |

---

## Arquitectura de la solución

```
VIDEO ─► YOLO (modelo propio) ─► ROI cancha (HSV, relleno) ─► ByteTrack
                                                                  │
        ┌─────────────────────────────────────────────────────────┤
        ▼                          ▼                               ▼
  Roster fijo (4 robots,    Respaldo de pelota          Equipo por FORMACIÓN
  identidad en oclusión)    por color (HSV)             inicial (+ color de respaldo)
        │                          │                               │
        └─────────────┬────────────┴───────────────┬──────────────┘
                      ▼                             ▼
          Homografía → vista cenital        SAM 3 (texto): "soccer robot",
          (METROS reales, campo 182×243)    "orange ball"  → máscaras
                      │
                      ▼
   StatsTracker (en metros, suavizado + banda muerta):
   distancia · velocidad · posesión (histéresis) · eventos
   (pases / intercepciones / tiros) · red de pases
                      │
   ┌────────────┬─────┴───────┬──────────────┬───────────────┐
   ▼            ▼             ▼              ▼               ▼
 Heatmaps    Voronoi      Timeline       Red de         dashboard.html
 cenitales   control      posesión       pases          + interfaz.html
```

El video de salida muestra robots y pelota etiquetados, las máscaras de SAM 3,
las trayectorias y las estadísticas del balón. La **vista cenital** se exporta
como un video aparte (`partido_homografia.mp4`).

---

## Innovación sobre SAM 3

Esta solución usa **SAM 3** (Segment Anything Model 3, Meta) — el modelo que pide
la convocatoria — de dos formas:

1. **En el pipeline (segmentación por texto):** con `"seg_backend": "sam3_hf"`,
   SAM 3 segmenta los elementos del partido mediante **conceptos en lenguaje
   natural** ("soccer robot", "orange ball"), sin cajas ni entrenamiento. Es la
   capacidad de **vocabulario abierto** que distingue a SAM 3 (sección 3.7.3).
   Las máscaras se generan cada `sam3_every_n` frames (SAM 3 por texto es lento),
   mientras el tracking y las estadísticas corren en cada frame.

2. **Demo independiente** (`sam3_demo.py`): segmenta unos frames con varios
   prompts de texto y guarda las imágenes con las máscaras superpuestas.

> SAM 3 (`facebook/sam3`) es un modelo **gated** en Hugging Face: requiere pedir
> acceso y autenticarse (`hf auth login`). Si no está disponible, el pipeline cae
> automáticamente a **SAM 2.1** (`sam2.1_s.pt`) como respaldo, sin detenerse.

Backends de segmentación (`seg_backend` en `CONFIG`):

| Valor | Qué usa | Notas |
|---|---|---|
| `sam3_hf` | SAM 3 por **texto** (transformers) | Vocabulario abierto; más lento; requiere acceso HF |
| `sam3` | SAM 3 box-prompt (Ultralytics) | Requiere el peso `sam3.pt` |
| `sam2` | SAM 2.1 box-prompt (Ultralytics) | Rápido; respaldo por defecto |

---

## Requisitos de hardware y software

### Hardware
| Componente | Recomendado | Mínimo |
|---|---|---|
| CPU/GPU | Apple Silicon (MPS) o GPU NVIDIA (CUDA) | CPU x86 (lento) |
| RAM | 16 GB | 8 GB |
| Almacenamiento | 10 GB libres | 5 GB (SAM 3 pesa ~3.4 GB) |

### Software
| Componente | Versión |
|---|---|
| Python | 3.11+ |
| PyTorch | 2.3+ (con MPS/CUDA) |
| Ultralytics, Supervision, OpenCV, NumPy, scikit-learn, Matplotlib | ver `requirements.txt` |
| transformers | solo para SAM 3 por texto (`pip install transformers`) |

---

## Instalación

> Guía completa y para principiantes en **[`docs/INSTALACION.md`](docs/INSTALACION.md)**.

```bash
# 1. Clonar y entrar
git clone <URL-del-repo>
cd futbotmx

# 2. Entorno virtual
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1

# 3. Dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Opcional) SAM 3 por texto — modelo gated de Meta
pip install transformers
hf auth login                        # pide acceso en huggingface.co/facebook/sam3
```

Coloca un video de partido en `assets/partido.mp4`. Los modelos `futbotmx_v2.pt`
(detección) y `sam2.1_s.pt` (respaldo) vienen en el repo; SAM 3 se descarga solo
la primera vez (si tienes acceso).

---

## Cómo reproducir los resultados

```bash
python pipeline.py                   # analiza assets/partido.mp4
python pipeline.py --all             # analiza TODOS los videos de assets/
python pipeline.py assets/otro.mp4   # un video específico
```

Al terminar, en `output/` (o `output/<nombre_video>/` con `--all`):

| Archivo | Contenido |
|---|---|
| `partido_analizado.mp4` | Video con detección, tracking, máscaras SAM 3 y stats del balón |
| `partido_homografia.mp4` | **Vista cenital** animada de la cancha (vertical, a escala real) |
| `estadisticas.json` | Posesión, eventos, distancias, velocidades, red de pases |
| `dashboard.html` | Dashboard interactivo (Chart.js) |
| **`interfaz.html`** | **Interfaz de resultados** con pestañas (videos + gráficas + galerías) |
| `heatmaps/` | Mapas de calor por robot (1–4), por equipo y del balón |
| `avanzadas/` | Voronoi de control, timeline de posesión, red de pases, trayectoria |

### Cambiar el backend de segmentación
En `CONFIG` (dentro de `pipeline.py`):
```python
"seg_backend": "sam3_hf",   # SAM 3 por texto | "sam3" | "sam2"
"sam3_prompts": ["soccer robot", "orange ball"],
"sam3_every_n": 15,         # súbelo (30/60) para ir más rápido
```

### Video demo lado a lado (requisito 3.5.3)
```bash
python make_demo_video.py            # original | analizado → output/demo_lado_a_lado.mp4
```

---

## Herramientas de calibración

El detector y el balón pueden afinarse sin adivinar:

```bash
python diagnostico_deteccion.py      # prueba resolución × confianza, recomienda la mejor
python calibrar_balon.py             # encuentra el rango de color (HSV) del balón naranja
python sam3_demo.py                  # demo de SAM 3 con prompts de texto
python entrenar_modelo.py            # reentrena el modelo (aumentación + métricas)
```

> `diagnostico_deteccion.py` y `calibrar_balon.py` te imprimen los valores exactos
> para pegar en `CONFIG`. Detalles en [`docs/CHECKLIST_ENTREGA.md`](docs/CHECKLIST_ENTREGA.md).

---

## Visualizaciones y estadísticas

**Estadísticas (`estadisticas.json`):**
- **Balón:** distancia total (m), velocidad máxima (km/h), nº de posiciones.
- **Por robot:** distancia (m), velocidad máxima (km/h), equipo (Azul/Rojo).
- **Posesión:** % por equipo y sin posesión (con histéresis para evitar parpadeo).
- **Eventos:** pases, intercepciones, tiros a gol (con tiempo, equipo y posición).
- **Red de pases:** grafo dirigido robot → robot.

**Visualizaciones:**
- Mapas de calor cenitales por robot, por equipo y del balón.
- **Diagrama de Voronoi** de control de cancha por equipo.
- **Línea de tiempo** de posesión.
- **Grafo de interacción** (red de pases).
- Trayectoria del balón en metros.
- **Dashboard** y **interfaz** HTML interactivas.

Todas las métricas están en **metros reales** gracias a la homografía calibrada con
las dimensiones oficiales de la cancha (182 × 243 cm, reglas 7.1).

---

## Relevancia deportiva: ¿para qué sirven estas métricas?

El objetivo no es solo "ver" el partido, sino **entenderlo** y dar herramientas
para mejorar — igual que en el fútbol profesional, donde el análisis de datos guía
las decisiones. Así puede usar cada métrica un **director técnico (DT)** o un
**equipo** en el *post-game* y para comprender el juego:

| Métrica / visualización | Qué revela | Cómo la usa el DT o el equipo |
|---|---|---|
| **Posesión por equipo** | Quién controló más el balón | Evaluar si la estrategia fue ofensiva o reactiva; ajustar el plan de juego |
| **Mapas de calor por robot** | Dónde pasó más tiempo cada robot | Detectar si un robot defiende/ataca de más, huecos sin cubrir, mala distribución de roles |
| **Diagrama de Voronoi (control de cancha)** | Qué zonas domina cada equipo | Ver si el equipo se amontona o cubre bien el campo; corregir el posicionamiento |
| **Red de pases** | Qué robots colaboran y cuál distribuye | Medir el trabajo en equipo (clave en la rúbrica de robótica); reforzar la coordinación entre robots |
| **Eventos (pases, intercepciones, tiros)** | Momentos decisivos del partido | Revisar jugadas concretas: ¿por qué se perdió el balón?, ¿desde dónde se tiró? |
| **Velocidad y distancia por robot** | Rendimiento físico/mecánico | Detectar un robot lento o que recorre de más (desgaste, mala ruta, problema de motor) |
| **Trayectoria del balón** | Cómo fluyó el juego | Identificar zonas de estancamiento y oportunidades de progresión |
| **Línea de tiempo de posesión** | Cómo cambió el control en el tiempo | Ubicar los tramos en que el rival dominó y correlacionarlos con los goles |

En la práctica, este análisis permite responder preguntas que a simple vista se
escapan: *¿nuestro equipo realmente colabora o juega cada robot por su cuenta?*,
*¿defendemos amontonados dejando huecos?*, *¿qué robot es el motor del equipo y
cuál hay que mejorar?*. Para un torneo de **robótica autónoma**, estas métricas
ayudan tanto al **juego** (estrategia, coordinación) como a la **ingeniería**
(ajustar control de motores, rutas y reparto de roles entre partidos), cerrando el
ciclo de mejora entre cada encuentro.

---

## Problemas encontrados y soluciones

Documentamos los principales retos del desarrollo y cómo los resolvimos (la
convocatoria valora la documentación de errores y hallazgos):

| Problema | Causa | Solución aplicada |
|---|---|---|
| **El balón salía con 0 posiciones** | El filtro de cancha descartaba el balón cuando caía sobre una línea o un robot | El filtro ya **nunca descarta el balón**; además se añadió respaldo de detección por **color (HSV)** |
| **El detector "no veía" robots** | El filtro usaba el **centro** de la caja, que en un robot cae sobre el robot (negro), no sobre el pasto | Se usa la **región de cancha rellena**; un robot sobre el pasto cuenta como "dentro" |
| **Velocidades imposibles (balón a 115 km/h)** | La homografía se recalculaba cada frame y "temblaba"; además el lienzo no tenía la proporción real de la cancha | **Suavizado (EMA) de las esquinas** + lienzo con la **proporción real** (escala idéntica por eje) |
| **Distancias infladas** | Cada micro-temblor de detección se sumaba frame a frame | **Suavizado de posición** + **banda muerta**: movimientos < 1.5 cm se ignoran |
| **Posesión casi siempre "sin posesión", sin pases** | Consecuencia de las velocidades/posiciones ruidosas | Al estabilizar la homografía y ampliar el radio de posesión, se registran posesión y pases |
| **Equipos marcados como "?" / inestables** | El color de los robots casi no distingue equipos; el k-means parpadeaba | **Asignación por formación inicial** (en el saque cada equipo está de su lado) + voto temporal de color como respaldo |
| **IDs de robots que se multiplicaban (5, 6, 7…)** | Tras una oclusión larga el tracker creaba IDs nuevos | **Roster fijo de 4 IDs reciclables** (2 por equipo) que sobreviven a la oclusión → como mucho 4 mapas de calor |
| **Detección con brecha de dominio** | El video real difiere de las fotos de entrenamiento (ángulo, luz, escala) | Subir/ajustar resolución de inferencia, bajar confianza, y para el balón usar **color**; identificamos que la cura de fondo es **más datos en-dominio** |
| **SAM 3 daba error 401 (gated)** | El modelo `facebook/sam3` requiere acceso aprobado y autenticación | Pedir acceso + `hf auth login`; el pipeline cae a **SAM 2.1** automáticamente si no hay acceso |
| **SAM 3 por texto es lento** | El modelo es pesado (~3.4 GB, segundos por frame) | Generar máscaras **cada N frames** (`sam3_every_n`); el tracking y las estadísticas siguen en cada frame |

> Hallazgo principal: la mayor mejora del detector no vino de más épocas de
> entrenamiento, sino de **calibrar resolución/confianza** y de **detectar el
> balón por color** (aprovechando que la regla 3.7 prohíbe el naranja en los
> robots). El siguiente paso para subir la precisión sería **ampliar el dataset
> con frames del propio torneo**.

---

## Estructura del repositorio

```
futbotmx/
├── pipeline.py              ← Pipeline principal (entrada)
├── dashboard.py             ← Dashboard HTML interactivo
├── generar_interfaz.py      ← Interfaz de resultados (visor con pestañas)
├── make_demo_video.py       ← Video demo lado a lado (3.5.3)
├── diagnostico_deteccion.py ← Calibrar resolución/confianza del detector
├── calibrar_balon.py        ← Calibrar la detección del balón por color
├── sam3_demo.py             ← Demo de SAM 3 con prompts de texto
├── entrenar_modelo.py       ← Reentrenar el modelo YOLO (aumentación + métricas)
├── roboflow_infer.py        ← Inferencia remota opcional (Roboflow, key por env)
├── requirements.txt
├── README.md
├── LICENSE                  ← MIT + atribución de dependencias
├── futbotmx_v2.pt           ← Modelo YOLO entrenado (0=Robot, 1=pelota)
├── sam2.1_s.pt              ← Segmentador de respaldo
├── docs/
│   ├── INSTALACION.md       ← Guía de instalación desde cero
│   ├── METODOLOGIA.md       ← Fundamentos matemáticos (ecuaciones)
│   └── CHECKLIST_ENTREGA.md ← Qué falta para entregar
├── utils/
│   ├── stats_tracker.py     ← Estadísticas, posesión y eventos (métricas en metros)
│   ├── team_assigner.py     ← Equipos por color + voto temporal
│   ├── team_formation.py    ← Equipos por formación inicial (saque)
│   ├── roster.py            ← Roster fijo de 4 + inercia de la pelota
│   ├── visualizations.py    ← Heatmaps, minimapa, resumen
│   ├── advanced_viz.py      ← Voronoi, timeline, red de pases
│   ├── entidades.py         ← Modelo POO (EntidadCancha → Robot/Balon, Equipo)
│   └── sam3_backend.py      ← SAM 3 por texto (transformers)
├── tests/
│   └── test_sintetico.py    ← Prueba de extremo a extremo sin video
├── dataset/                 ← Dataset etiquetado (Roboflow, CC BY 4.0)
├── assets/                  ← Videos de entrada
└── output/                  ← Resultados generados
```

---

## Documentación técnica

- **[`docs/METODOLOGIA.md`](docs/METODOLOGIA.md):** todas las ecuaciones — homografía
  (DLT), escala métrica, cinemática (distancia/velocidad), posesión con histéresis,
  detección de eventos, mapas de calor (KDE), Voronoi, red de pases, k-means.
- **[`docs/INSTALACION.md`](docs/INSTALACION.md):** instalación paso a paso para
  alguien que nunca usó el programa (Mac y Windows).
- **[`docs/CHECKLIST_ENTREGA.md`](docs/CHECKLIST_ENTREGA.md):** mapeo de cada
  requisito de la convocatoria (3.5) a lo entregado.

---

## Librerías de código abierto utilizadas

| Librería | Uso | Licencia |
|---|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO (detección) y SAM | AGPL-3.0 |
| [SAM 3 (Meta)](https://huggingface.co/facebook/sam3) | Segmentación por texto/vocabulario abierto | SAM License |
| [transformers (Hugging Face)](https://github.com/huggingface/transformers) | Cargar SAM 3 | Apache-2.0 |
| [Supervision](https://github.com/roboflow/supervision) | ByteTrack, anotadores, process_video | MIT |
| [PyTorch](https://pytorch.org/) | Backend de inferencia (MPS/CUDA/CPU) | BSD-3 |
| [OpenCV](https://opencv.org/) | Imagen, video, homografía | Apache-2.0 |
| [NumPy](https://numpy.org/) · [scikit-learn](https://scikit-learn.org/) | Cálculo · k-means | BSD-3 |
| [Matplotlib](https://matplotlib.org/) · [Chart.js](https://www.chartjs.org/) | Gráficas · dashboard | PSF · MIT |

Dataset propio etiquetado en **Roboflow** (licencia CC BY 4.0). Todos los créditos
a sus autores; las licencias se respetan según la sección 3.6 de la convocatoria.

---

## Reel de Instagram

📸 Reel con lo más destacado del proyecto:

> https://www.instagram.com/reel/DZy7i7DozM0NG7exCNtJvANVCCuxD40U8VHsyM0/?igsh=MWloZTE5NHhqanJ3eg==

---

## Licencia y créditos

Este proyecto se distribuye bajo licencia **MIT** (ver [`LICENSE`](LICENSE)).

**Equipo:**
- José Ángel Garcia Ruiz
- Luis Carlos Ayuso Guarneros
- Diego Alejandro Vallecillo Parra
- Eddie Adnan Alvarez

**Reto:** Copa FutBotMX — Capítulo Visión por Computadora
**Organización:** Secihti · Meta · Centro

El modelo **SAM 3** se utiliza bajo la [SAM License](https://huggingface.co/facebook/sam3).
**YOLO/SAM (Ultralytics)** bajo [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

---

*Copa FutBotMX 2026 · Visión por Computadora*
