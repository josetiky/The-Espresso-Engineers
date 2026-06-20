"""
Sam3Backend — SAM 3 (Meta) con PROMPTS DE TEXTO / vocabulario abierto.
────────────────────────────────────────────────────────────────────────
SAM 3 segmenta por CONCEPTO en lenguaje natural: le dices "soccer robot" o
"orange ball" y devuelve las máscaras de esos objetos, sin entrenar nada.
Es justo la innovación que destaca la convocatoria (sección 3.7.3: "prompts
de texto" y "segmentación mediante conceptos de vocabulario abierto").

Basado en el notebook 13 del curso (transformers: Sam3Processor / Sam3Model).

REQUISITOS:
  pip install transformers torch
  huggingface-cli login          # el modelo facebook/sam3 está "gated":
                                 # pide acceso en huggingface.co/facebook/sam3

NOTA DE RENDIMIENTO: en CPU SAM 3 tarda ~30-60 s por imagen; en GPU/MPS es
mucho más rápido. Por eso se usa para DEMOSTRAR la segmentación por texto en
unos cuantos frames, no para procesar las miles de imágenes de un video
completo (para el video se usa el SAM box-prompt de Ultralytics, más rápido).
"""

import numpy as np
import cv2
import supervision as sv


class Sam3Backend:
    def __init__(self, model_id: str = "facebook/sam3", device: str | None = None):
        import torch
        from transformers import Sam3Processor, Sam3Model

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self._torch = torch
        print(f"[SAM3] Cargando facebook/sam3 en {device} (puede tardar la 1ª vez)…")
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(device).eval()
        print("[SAM3] Modelo listo.")

    # ─────────────────────────────────────────────────────────────────
    def segment(self, frame_bgr: np.ndarray, text: str,
                threshold: float = 0.5) -> sv.Detections:
        """Segmenta en `frame_bgr` el CONCEPTO descrito por `text`
        (p. ej. 'soccer robot', 'orange ball'). Devuelve sv.Detections con
        máscaras, cajas y confianza."""
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, text=text,
                                return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)

        # Post-proceso → máscaras/cajas/scores (tamaño original de la imagen)
        target = [image.size[::-1]]   # (alto, ancho)
        try:
            results = self.processor.post_process_instance_segmentation(
                outputs, threshold=threshold, target_sizes=target)[0]
        except Exception:
            # Variante de API: post-proceso de detección con grounding
            results = self.processor.post_process_grounded_object_detection(
                outputs, threshold=threshold, target_sizes=target)[0]

        masks = results.get("masks")
        boxes = results.get("boxes")
        scores = results.get("scores")
        if masks is None or len(masks) == 0:
            return sv.Detections.empty()

        masks = masks.cpu().numpy().astype(bool)
        if masks.ndim == 4:           # (N,1,H,W) → (N,H,W)
            masks = masks[:, 0]
        xyxy = (boxes.cpu().numpy() if boxes is not None
                else _boxes_from_masks(masks))
        conf = (scores.cpu().numpy() if scores is not None
                else np.ones(len(masks), dtype=np.float32))
        return sv.Detections(xyxy=xyxy.astype(np.float32),
                             mask=masks, confidence=conf.astype(np.float32))


def _boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    """Caja envolvente (x1,y1,x2,y2) de cada máscara."""
    out = []
    for m in masks:
        ys, xs = np.where(m)
        if len(xs) == 0:
            out.append([0, 0, 0, 0])
        else:
            out.append([xs.min(), ys.min(), xs.max(), ys.max()])
    return np.array(out, dtype=np.float32)
