"""Screenshots and annotated video export."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class Exporter:
    def __init__(self, out_dir: Path):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self._writer: cv2.VideoWriter | None = None
        self._writer_path: Path | None = None

    # ---- screenshots ------------------------------------------------------
    def screenshot(self, image: np.ndarray, tag: str = "shot") -> Path:
        path = self.out / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(path), image)
        return path

    # ---- annotated video --------------------------------------------------
    def start_video(self, path: str | Path, size: tuple[int, int], fps: float) -> Path:
        p = Path(path)
        if p.suffix.lower() not in (".mp4", ".avi", ".mov"):
            p = p.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(p), fourcc, max(1.0, fps), size)
        self._writer_path = p
        return p

    def write_frame(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def stop_video(self) -> Path | None:
        if self._writer is not None:
            self._writer.release()
        p = self._writer_path
        self._writer = None
        self._writer_path = None
        return p