"""
roboflow_infer.py — Inferencia REMOTA opcional con la API de Roboflow.
────────────────────────────────────────────────────────────────────────
⚠️  SOLO para PROBAR IMÁGENES SUELTAS. Para procesar VIDEO usa el modelo
    LOCAL (futbotmx_v2.pt) en pipeline.py: la API en la nube es lenta
    frame-a-frame, depende de internet, tiene límites de uso y rompe la
    reproducibilidad (quien clone el repo necesitaría una API key).

🔐  SEGURIDAD: la API key NUNCA va escrita en el código ni se sube al repo.
    Se lee de la variable de entorno ROBOFLOW_API_KEY. Antes de usar esto:
      1. Regenera tu key en Roboflow (la anterior quedó expuesta).
      2. Expórtala en tu terminal (no la pegues en archivos versionados):
             export ROBOFLOW_API_KEY="tu_nueva_key"

Uso:
    export ROBOFLOW_API_KEY="tu_nueva_key"
    python roboflow_infer.py assets/un_frame.jpg
"""

import os
import sys


MODEL_ID = "copafutbot-5lyab/4"
API_URL = "https://serverless.roboflow.com"


def infer_image(image_path: str, model_id: str = MODEL_ID):
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "Falta la API key. Ejecuta primero:\n"
            '    export ROBOFLOW_API_KEY="tu_nueva_key"\n'
            "(NO la escribas en el código ni la subas al repositorio).")

    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError:
        raise SystemExit("Instala el SDK:  pip install inference-sdk")

    client = InferenceHTTPClient(api_url=API_URL, api_key=api_key)
    result = client.infer(image_path, model_id=model_id)

    preds = result.get("predictions", [])
    print(f"[OK] {len(preds)} detecciones en {image_path}:")
    for p in preds:
        print(f"   · {p['class']:8s}  conf={p['confidence']:.2f}  "
              f"({p['x']:.0f},{p['y']:.0f})  {p['width']:.0f}×{p['height']:.0f}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python roboflow_infer.py <imagen.jpg>")
        raise SystemExit(1)
    infer_image(sys.argv[1])
