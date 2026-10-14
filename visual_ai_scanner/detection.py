"""Detection backends: closed-set (YOLO), open-vocab (Grounding DINO),
reference-image (OWLv2 image-guided). All lazy-loaded so app starts quickly."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    score: float
    track_id: int | None = None
    source: str = "yolo"          # yolo | dino | reference
    mask: np.ndarray | None = None

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


# ---------------------------------------------------------------------------
# YOLO (closed-set, real-time)
# ---------------------------------------------------------------------------
class YoloDetector:
    def __init__(self, weights: str = "yolo11n.pt", device: str = "auto",
                 half: bool = True):
        from ultralytics import YOLO  # lazy
        self.device = self._resolve_device(device)
        self.half = bool(half and self.device != "cpu")
        self.model = YOLO(weights)
        try:
            self.model.to(self.device)
        except Exception as e:
            log.warning("Falling back to CPU for YOLO: %s", e)
            self.device, self.half = "cpu", False

    @staticmethod
    def _resolve_device(dev: str) -> str:
        if dev != "auto":
            return dev
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def infer(self, image: np.ndarray, conf: float = 0.35,
              classes: list[int] | None = None) -> list[Detection]:
        results = self.model.predict(
            image, conf=conf, classes=classes,
            device=self.device, half=self.half, verbose=False,
            imgsz=960,
        )
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        names = r.names
        if r.boxes is None:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        sc = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, sc):
            out.append(Detection(int(x1), int(y1), int(x2), int(y2),
                                 names.get(int(c), str(int(c))), float(s), source="yolo"))
        return out


# ---------------------------------------------------------------------------
# Grounding DINO (open vocabulary, text-conditioned)
# ---------------------------------------------------------------------------
class GroundingDinoDetector:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny",
                 device: str = "auto"):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch = torch
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else (
            device if device != "auto" else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def infer(self, image_bgr: np.ndarray, prompts: list[str],
              box_thr: float = 0.30, text_thr: float = 0.25) -> list[Detection]:
        if not prompts:
            return []
        from PIL import Image
        rgb = image_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)
        # Grounding DINO expects lowercase, period-separated prompts.
        text = ". ".join(p.strip().rstrip(".").lower() for p in prompts) + "."
        inputs = self.processor(images=pil, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=box_thr, text_threshold=text_thr,
            target_sizes=[pil.size[::-1]],
        )[0]
        out: list[Detection] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            out.append(Detection(x1, y1, x2, y2, str(label).strip(),
                                 float(score), source="dino"))
        return out


# ---------------------------------------------------------------------------
# Reference-image guided detection (OWLv2 image-guided)
# ---------------------------------------------------------------------------
class ReferenceMatcher:
    """Uses OWLv2's image_guided_detection. Notes:
    - OWLv2 supports query images natively; that is *detection*, not just embedding.
    - Similarity is visual only; a high score does not prove identity.
    """
    def __init__(self, model_id: str = "google/owlv2-base-patch16-ensemble",
                 device: str = "auto"):
        import torch
        from transformers import OwlVi2Processor, Owlv2ForObjectDetection
        self.torch = torch
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else (
            device if device != "auto" else "cpu")
        self.processor = OwlVi2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self._query_cache = None
        self._query_key = None

    def set_reference(self, image_bgr: np.ndarray) -> None:
        from PIL import Image
        pil = Image.fromarray(image_bgr[:, :, ::-1])
        self._query_cache = self.processor(images=pil, return_tensors="pt").to(self.device)
        self._query_key = id(image_bgr)

    def infer(self, image_bgr: np.ndarray, threshold: float = 0.35) -> list[Detection]:
        if self._query_cache is None:
            return []
        from PIL import Image
        pil = Image.fromarray(image_bgr[:, :, ::-1])
        target = self.processor(images=pil, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(
                input_ids=self._query_cache["input_ids"],
                pixel_values=self._query_cache["pixel_values"],
                query_pixel_values=target["pixel_values"],
            )
        target_sizes = self.torch.Tensor([pil.size[::-1]]).to(self.device)
        results = self.processor.post_process_image_guided_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        out: list[Detection] = []
        for box, score in zip(results["boxes"], results["scores"]):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            out.append(Detection(x1, y1, x2, y2, "Reference match",
                                 float(score), source="reference"))
        return out