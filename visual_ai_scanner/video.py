"""Unified camera / file video source + bounded frame queue that drops stale frames."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray        # BGR
    index: int
    timestamp: float         # seconds, source-relative
    is_live: bool


class VideoSource:
    """Common interface for CameraSource and FileSource."""

    def read(self) -> Frame | None: ...
    def release(self) -> None: ...
    @property
    def fps(self) -> float: ...
    @property
    def size(self) -> tuple[int, int]: ...
    @property
    def duration(self) -> float: return 0.0
    def seek(self, seconds: float) -> None: ...


class CameraSource(VideoSource):
    def __init__(self, index: int, width: int, height: int):
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DSHOW is reliable on Windows
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._i = 0
        self._t0 = time.time()

    def read(self) -> Frame | None:
        ok, img = self._cap.read()
        if not ok or img is None:
            return None
        self._i += 1
        return Frame(img, self._i, time.time() - self._t0, True)

    def release(self) -> None:
        self._cap.release()

    @property
    def fps(self) -> float: return self._fps or 30.0

    @property
    def size(self) -> tuple[int, int]: return self._size


class FileSource(VideoSource):
    def __init__(self, path: str | Path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        self._cap = cv2.VideoCapture(str(p))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {p}")
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self._size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._i = 0

    def read(self) -> Frame | None:
        ok, img = self._cap.read()
        if not ok or img is None:
            return None
        self._i += 1
        return Frame(img, self._i, self._i / self._fps, False)

    def release(self) -> None:
        self._cap.release()

    def seek(self, seconds: float) -> None:
        self._cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        self._i = int(seconds * self._fps)

    @property
    def fps(self) -> float: return self._fps or 25.0

    @property
    def size(self) -> tuple[int, int]: return self._size

    @property
    def duration(self) -> float:
        return self._count / self._fps if self._fps else 0.0


class BoundedFrameQueue:
    """Thread-safe queue that drops oldest frames when full (avoids backlog)."""

    def __init__(self, maxlen: int = 4):
        self._q: deque[Frame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False
        self.dropped = 0

    def put(self, frame: Frame) -> None:
        with self._not_empty:
            if len(self._q) == self._q.maxlen:
                self.dropped += 1
            self._q.append(frame)
            self._not_empty.notify()

    def get(self, timeout: float = 0.5) -> Frame | None:
        with self._not_empty:
            if not self._q and not self._closed:
                self._not_empty.wait(timeout)
            if self._q:
                return self._q.popleft()
            return None

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)