"""Typed configuration with YAML persistence."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppCfg:
    theme: str = "dark"
    save_dir: str = "./outputs"
    log_level: str = "INFO"
    event_history_max: int = 500


@dataclass
class VideoCfg:
    camera_index: int = 0
    target_width: int = 1280
    target_height: int = 720
    process_every_n_frames: int = 2
    queue_size: int = 4
    prefer_smooth_playback: bool = True


@dataclass
class ModelsCfg:
    device: str = "auto"
    detector_weights: str = "yolo11n.pt"
    open_vocab_model: str = "IDEA-Research/grounding-dino-tiny"
    reference_model: str = "google/owlv2-base-patch16-ensemble"
    sam_model: str = "mobile_sam.pt"
    vlm_model: str = "Salesforce/blip-image-captioning-large"
    half_precision: bool = True


@dataclass
class ThresholdsCfg:
    detection_confidence: float = 0.35
    open_vocab_box_threshold: float = 0.30
    open_vocab_text_threshold: float = 0.25
    reference_match_threshold: float = 0.35


@dataclass
class FeaturesCfg:
    tracking_enabled: bool = True
    segmentation_enabled: bool = False
    motion_trails: bool = False
    trail_length: int = 30


@dataclass
class TargetsCfg:
    text: list[str] = field(default_factory=list)
    reference_image: str | None = None


@dataclass
class RenderingCfg:
    box_thickness: int = 2
    label_font_scale: float = 0.5
    show_confidence: bool = True
    show_track_id: bool = True


@dataclass
class Config:
    app: AppCfg = field(default_factory=AppCfg)
    video: VideoCfg = field(default_factory=VideoCfg)
    models: ModelsCfg = field(default_factory=ModelsCfg)
    thresholds: ThresholdsCfg = field(default_factory=ThresholdsCfg)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    targets: TargetsCfg = field(default_factory=TargetsCfg)
    rendering: RenderingCfg = field(default_factory=RenderingCfg)

    # ---- IO ---------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str) -> "Config":
        p = Path(path)
        if not p.exists():
            cfg = cls()
            cfg.save(p)
            return cfg
        with p.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        def build(dc, sub):
            if not isinstance(sub, dict):
                return dc()
            valid = {f for f in dc.__dataclass_fields__}
            return dc(**{k: v for k, v in sub.items() if k in valid})
        return cls(
            app=build(AppCfg, raw.get("app", {})),
            video=build(VideoCfg, raw.get("video", {})),
            models=build(ModelsCfg, raw.get("models", {})),
            thresholds=build(ThresholdsCfg, raw.get("thresholds", {})),
            features=build(FeaturesCfg, raw.get("features", {})),
            targets=build(TargetsCfg, raw.get("targets", {})),
            rendering=build(RenderingCfg, raw.get("rendering", {})),
        )

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)