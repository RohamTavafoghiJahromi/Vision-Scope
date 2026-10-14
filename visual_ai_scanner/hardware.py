"""Detect GPU/CUDA availability and produce conservative model-size advice."""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    cuda_available: bool
    device_name: str
    vram_gb: float
    torch_version: str
    cuda_version: str | None
    recommended_detector: str
    notes: str

    def describe(self) -> str:
        lines = [
            f"Device       : {self.device_name}",
            f"CUDA         : {'yes' if self.cuda_available else 'no (CPU fallback)'}",
            f"VRAM         : {self.vram_gb:.1f} GB" if self.cuda_available else "VRAM         : n/a",
            f"torch        : {self.torch_version}",
            f"CUDA runtime : {self.cuda_version or 'n/a'}",
            f"Recommended detector: {self.recommended_detector}",
        ]
        if self.notes:
            lines.append(f"Notes        : {self.notes}")
        return "\n".join(lines)


def detect_hardware() -> HardwareInfo:
    try:
        import torch
    except Exception as e:  # torch not installed yet
        return HardwareInfo(False, "unknown", 0.0, "not installed", None,
                            "yolo11n.pt", f"torch import failed: {e}")

    cuda = torch.cuda.is_available()
    name = "CPU"
    vram = 0.0
    cuda_ver = getattr(torch.version, "cuda", None)
    if cuda:
        try:
            props = torch.cuda.get_device_properties(0)
            name = props.name
            vram = props.total_memory / (1024 ** 3)
        except Exception as e:
            log.warning("Could not read CUDA device properties: %s", e)

    # Conservative advice based on VRAM only. Not a benchmark.
    if not cuda:
        rec, note = "yolo11n.pt", "No CUDA — detector will run on CPU; expect low FPS."
    elif vram < 4:
        rec, note = "yolo11n.pt", "Low VRAM: stick to nano/small detectors."
    elif vram < 8:
        rec, note = "yolo11s.pt", "Medium VRAM: small detectors and small VLMs OK."
    elif vram < 12:
        rec, note = "yolo11m.pt", "Comfortable: medium detector, Grounding DINO tiny OK."
    else:
        rec, note = "yolo11l.pt", "High VRAM: large detectors and larger VLMs feasible."

    return HardwareInfo(cuda, name, vram, torch.__version__, cuda_ver, rec, note)