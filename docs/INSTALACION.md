# Guía de instalación desde cero — Copa FutBotMX (Visión por Computadora)

Esta guía es para alguien que **nunca ha usado el programa**. Explica TODO lo
que hay que descargar e instalar, paso a paso. No necesitas saber de visión por
computadora; solo seguir los pasos en orden.

---

## 1. ¿Qué necesito? (resumen rápido)

| Qué | Para qué | ¿Obligatorio? |
|---|---|---|
| **Python 3.11+** | Lenguaje en el que corre el programa | ✅ Sí |
| **Git** (o descargar el ZIP) | Bajar el proyecto | ✅ Sí |
| **El repositorio del proyecto** | El código y el modelo entrenado | ✅ Sí |
| **Las librerías de Python** (`requirements.txt`) | Detección, tracking, gráficas | ✅ Sí |
| **Un video de partido** (`.mp4`) | Lo que se va a analizar | ✅ Sí |
| **Modelos** (`futbotmx_v2.pt`, `sam2.1_s.pt`) | Detectar y segmentar | ✅ Sí (vienen en el repo o se descargan solos) |
| **transformers + acceso a SAM 3** | Segmentación por texto con SAM 3 | ❌ Opcional (recomendado) |
| **inference-sdk** (Roboflow) | Solo si usarás la API en la nube | ❌ Opcional |

**Requisitos de la computadora:**
- **Recomendado:** Mac con chip Apple Silicon (M1/M2/M3/M4/M5) — usa aceleración *MPS*.
  También sirve una PC con GPU NVIDIA (*CUDA*), o cualquier computadora con CPU (lento).
- **RAM:** 8 GB mínimo, 16 GB recomendado.
- **Disco:** ~10 GB libres (el modelo SAM 3 pesa ~3.4 GB).
- **Sistema:** macOS 13+, Windows 10/11 o Linux.

---

## 2. Instalar Python

### Mac
1. Descarga Python 3.11 o superior desde https://www.python.org/downloads/
2. Abre el instalador y sigue los pasos por defecto.
3. Verifica en la app **Terminal**:
   ```bash
   python3 --version
   ```

### Windows
1. Descarga Python desde https://www.python.org/downloads/
2. **IMPORTANTE:** marca la casilla **“Add Python to PATH”** en la primera pantalla.
3. Verifica en **PowerShell**:
   ```bash
   python --version
   ```

> En Windows usa `python`; en Mac/Linux usa `python3`. En esta guía verás `python`.
> Cuidado al escribir: es `python`, no `phyton`.

---

## 3. Instalar Git (para descargar el proyecto)

- **Mac:** suele venir; verifica con `git --version`. Si no: https://git-scm.com/download/mac
- **Windows:** https://git-scm.com/download/win

> **Sin Git:** en GitHub, botón verde **“Code” → “Download ZIP”** y descomprímelo.

---

## 4. Descargar el proyecto

```bash
cd ~/Desktop
git clone <URL-del-repositorio-en-GitHub>
cd futbotmx
```

---

## 5. Crear y activar el entorno virtual

```bash
python3 -m venv .venv
```

Actívalo:
- **Mac/Linux:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`

Verás `(.venv)` al inicio de la línea cuando esté activo.

---

## 6. Instalar las librerías

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Instala: PyTorch, Ultralytics (YOLO + SAM), Supervision (tracking), OpenCV,
NumPy, scikit-learn, Matplotlib, etc. Puede tardar varios minutos.

> **Mac con Apple Silicon:** verifica la aceleración:
> ```bash
> python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
> ```
> Debe imprimir `MPS: True`.

---

## 7. Modelos de detección y segmentación

- **`futbotmx_v2.pt`** — modelo de detección entrenado (clases: `0 = Robot`,
  `1 = pelota`). **Viene en el repositorio.**
- **`sam2.1_s.pt`** — segmentador de respaldo. Viene en el repo o se descarga solo.

No tienes que hacer nada: si están, se usan; si falta SAM, se baja automáticamente.

---

## 8. (Opcional, recomendado) SAM 3 por texto

El reto pide **SAM 3**. Este proyecto lo usa con **prompts de texto / vocabulario
abierto** ("soccer robot", "orange ball"). SAM 3 (`facebook/sam3`) es un modelo
**gated** (cerrado) en Hugging Face, así que hay que pedir acceso y autenticarse.

1. **Instala transformers:**
   ```bash
   pip install transformers
   ```
2. **Crea una cuenta** en https://huggingface.co e inicia sesión.
3. **Pide acceso al modelo:** entra a https://huggingface.co/facebook/sam3 y acepta
   los términos / "Request access". Espera a que diga *"You have been granted access"*.
4. **Crea un token** tipo **Read**: https://huggingface.co/settings/tokens → "New token".
5. **Autentícate en la terminal** (con el entorno activo):
   ```bash
   hf auth login
   ```
   Pega el token (empieza con `hf_...`; no se ve al pegarlo) y Enter. A
   "Add token as git credential?" puedes responder `n`.

> ⚠️ El comando viejo `huggingface-cli login` está **deprecado**; usa `hf auth login`.
>
> La primera vez, SAM 3 descarga ~3.4 GB (tarda). Si **no** tienes acceso o no
> usas SAM 3, el programa funciona igual con SAM 2.1 (no tienes que hacer nada).

---

## 9. Poner un video para analizar

Coloca un video del repositorio oficial en `assets/partido.mp4`:

```
futbotmx/
└── assets/
    └── partido.mp4
```

> Formatos: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`.

---

## 10. Correr el programa

```bash
python pipeline.py
```

Al terminar, en `output/` tendrás: `partido_analizado.mp4`, `partido_homografia.mp4`
(vista cenital), `estadisticas.json`, `dashboard.html`, **`interfaz.html`** (ábrela
en el navegador), y las carpetas `heatmaps/` y `avanzadas/`.

### Otros comandos útiles

```bash
python pipeline.py --all          # analiza TODOS los videos de assets/
python pipeline.py assets/otro.mp4   # un video específico

python diagnostico_deteccion.py   # encuentra la mejor resolución/confianza del detector
python calibrar_balon.py          # ajusta la detección del balón (por color)
python sam3_demo.py               # demo de SAM 3 con prompts de texto
python make_demo_video.py         # video demo "original | analizado"
python generar_interfaz.py output # regenera la interfaz HTML
```

### Elegir el segmentador (en `CONFIG`, dentro de `pipeline.py`)

```python
"seg_backend": "sam3_hf",   # SAM 3 por texto | "sam3" (Ultralytics) | "sam2" (rápido)
"sam3_every_n": 15,         # SAM 3 es lento: genera máscaras cada N frames (súbelo para acelerar)
```

### Reentrenar el modelo (opcional, avanzado)

```bash
python entrenar_modelo.py                          # 80 épocas
python entrenar_modelo.py --modelo yolo11m.pt --imgsz 960 --epochs 100
```

---

## 11. Problemas comunes

| Problema | Solución |
|---|---|
| `command not found: python3` | Reinstala Python con “Add to PATH” (Windows) o reinicia la Terminal. |
| `command not found: phyton` | Escríbelo bien: es `python` / `python3`. |
| `No module named ...` | ¿Activaste el entorno? (`source .venv/bin/activate`) y corre `pip install -r requirements.txt`. |
| `No module named 'transformers'` | `pip install transformers` (solo para SAM 3 por texto). |
| SAM 3: `gated repo` / `401` | Pide acceso en huggingface.co/facebook/sam3 y haz `hf auth login` con tu token. |
| `huggingface-cli is deprecated` | Usa `hf auth login` en su lugar. |
| `MPS: False` en Mac | Tu Mac no es Apple Silicon o falta actualizar PyTorch; correrá en CPU (lento). |
| Va muy lento con SAM 3 | Sube `sam3_every_n` a `30`/`60`, o usa `"seg_backend": "sam2"`. |
| No detecta el balón | `python calibrar_balon.py` y pega el rango HSV en `CONFIG`. |
| `unrecognized arguments: # ...` | No pegues los comentarios (lo que va después de `#`) en la terminal. |
| No abre los videos en `interfaz.html` | Abre el HTML que está DENTRO de `output/`, junto a los videos. |

---

## 12. Comandos de un vistazo (copia-pega)

```bash
# 1. Entrar a la carpeta del proyecto
cd ~/Desktop/futbotmx

# 2. Crear y activar el entorno (solo la primera vez se crea)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

# 3. Instalar librerías (solo la primera vez)
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Opcional) SAM 3 por texto
pip install transformers
hf auth login                      # requiere acceso a huggingface.co/facebook/sam3

# 5. Poner el video en assets/partido.mp4 y correr
python pipeline.py
```

Las próximas veces: entra a la carpeta, activa el entorno
(`source .venv/bin/activate`) y corre `python pipeline.py`.
