"""Model availability probing + download helpers. Never claims a model is
present unless it actually exists on disk or is loadable."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
ULTRALYTICS_WEIGHTS = Path.cwd()  # ultralytics downloads into CWD by default


@dataclass
class ModelEntry:
    key: str
    name: str
    purpose: str
    kind: str          # hf | ultralytics
    identifier: str
    approx_size: str
    license: str
    installed: bool


def _hf_installed(model_id: str) -> bool:
    folder = HF_CACHE / ("models--" + model_id.replace("/", "--"))
    return folder.exists() and any(folder.rglob("*.safetensors"))


def _ul_installed(weights: str) -> bool:
    return (Path.cwd() / weights).exists()


def list_models() -> list[ModelEntry]:
    entries = [
        ModelEntry("yolo_n",  "YOLO11-nano",  "Fast closed-set detection",
                   "ultralytics", "yolo11n.pt",  "~5 MB",  "AGPL-3.0",
                   _ul_installed("yolo11n.pt")),
        ModelEntry("yolo_s",  "YOLO11-small", "Balanced detection",
                   "ultralytics", "yolo11s.pt",  "~19 MB", "AGPL-3.0",
                   _ul_installed("yolo11s.pt")),
        ModelEntry("dino",    "Grounding DINO tiny", "Open-vocabulary text search",
                   "hf", "IDEA-Research/grounding-dino-tiny", "~700 MB",
                   "Apache-2.0 (weights vary; verify upstream)",
                   _hf_installed("IDEA-Research/grounding-dino-tiny")),
        ModelEntry("owlv2",   "OWLv2 base patch16", "Reference-image guided detection",
                   "hf", "google/owlv2-base-patch16-ensemble", "~1.1 GB",
                   "Apache-2.0", _hf_installed("google/owlv2-base-patch16-ensemble")),
        ModelEntry("sam2",    "MobileSAM",    "Optional segmentation",
                   "ultralytics", "mobile_sam.pt", "~40 MB", "Apache-2.0",
                   _ul_installed("mobile_sam.pt")),
        ModelEntry("blip",    "BLIP captioning", "Detailed image descriptions",
                   "hf", "Salesforce/blip-image-captioning-large", "~1.8 GB",
                   "BSD-3-Clause", _hf_installed("Salesforce/blip-image-captioning-large")),
    ]
    return entries


def download(model: ModelEntry) -> None:
    """Explicit, user-initiated download. Nothing here runs silently."""
    if model.kind == "ultralytics":
        from ultralytics import YOLO
        YOLO(model.identifier)  # triggers download to CWD
    elif model.kind == "hf":
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model.identifier)
    else:
        raise ValueError(model.kind)