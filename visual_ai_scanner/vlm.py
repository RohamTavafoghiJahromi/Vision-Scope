"""Local vision-language model for on-demand frame / crop description.
Uses BLIP by default; can be swapped to a Qwen2-VL model for richer output."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class VlmDescriber:
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-large",
                 device: str = "auto"):
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        self.torch = torch
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else (
            device if device != "auto" else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device).eval()

    def describe(self, image_bgr: np.ndarray, prompt: str | None = None) -> str:
        from PIL import Image
        pil = Image.fromarray(image_bgr[:, :, ::-1])
        if prompt:
            inputs = self.processor(pil, prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(pil, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=64)
        return self.processor.decode(ids[0], skip_special_tokens=True).strip()