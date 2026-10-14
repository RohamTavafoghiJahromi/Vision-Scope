"""Worker thread: pulls frames, runs detection/tracking/segmentation,
emits annotated frames + detection lists via Qt signals. All heavy work
happens off the UI thread."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal

from .config import Config
from .detection import Detection, YoloDetector, GroundingDinoDetector, ReferenceMatcher
from .events import EventHistory
from .rendering import draw_detections
from .segmentation import SamSegmenter
from .tracking import Tracker
from .video import BoundedFrameQueue, FileSource, Frame, VideoSource
log = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    fps: float = 0.0
    latency_ms: float = 0.0
    device: str = "cpu"
    dropped: int = 0
    frames_seen: int = 0


@dataclass
class PipelineState:
    text_targets: list[str] = field(default_factory=list)
    reference_image: np.ndarray | None = None
    running: bool = True
    paused: bool = False
    speed: float = 1.0   # 0.1 = slow-mo, 8.0 = fast-forward, 0 = unlimited

class ProcessingWorker(QObject):
    frame_ready = Signal(object, list, object)   # (annotated_bgr, detections, stats)
    error = Signal(str)
    finished = Signal()

    def __init__(self, cfg: Config, source: VideoSource, queue: BoundedFrameQueue,
                 on_describe: Callable[[np.ndarray, str], str] | None = None):
        super().__init__()
        self.cfg = cfg
        self.source = source
        self.queue = queue
        self.on_describe = on_describe
        self.state = PipelineState()
        self._stop = threading.Event()

        self.stats = PipelineStats()
        self.history = EventHistory(cfg.app.event_history_max)

        # Lazy components — created on first use.
        self._yolo: YoloDetector | None = None
        self._tracker: Tracker | None = None
        self._dino: GroundingDinoDetector | None = None
        self._refmatch: ReferenceMatcher | None = None
        self._sam: SamSegmenter | None = None

    # -- lazy loaders ------------------------------------------------------
    def _ensure_yolo(self) -> YoloDetector:
        if self._yolo is None:
            self._yolo = YoloDetector(self.cfg.models.detector_weights,
                                      self.cfg.models.device,
                                      self.cfg.models.half_precision)
            self.stats.device = self._yolo.device
            if self.cfg.features.tracking_enabled:
                self._tracker = Tracker(self._yolo)
        return self._yolo

    def _ensure_dino(self) -> GroundingDinoDetector:
        if self._dino is None:
            self._dino = GroundingDinoDetector(self.cfg.models.open_vocab_model,
                                               self.cfg.models.device)
        return self._dino

    def _ensure_refmatch(self) -> ReferenceMatcher:
        if self._refmatch is None:
            self._refmatch = ReferenceMatcher(self.cfg.models.reference_model,
                                              self.cfg.models.device)
        return self._refmatch

    def _ensure_sam(self) -> SamSegmenter:
        if self._sam is None:
            self._sam = SamSegmenter(self.cfg.models.sam_model, self.cfg.models.device)
        return self._sam

    # -- control -----------------------------------------------------------
    def request_stop(self) -> None:
        self._stop.set()

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        try:
            n = max(1, self.cfg.video.process_every_n_frames)
            counter = 0
            last_t = time.perf_counter()
            fps_ema = 0.0

            # Prime the queue in a background pump so the UI stays responsive.
            stop_pump = threading.Event()

            def pump():
                import time as _t
                is_file = isinstance(self.source, FileSource)
                base_interval = 1.0 / max(1.0, self.source.fps) if is_file else 0.0
                next_time = _t.perf_counter()
                eof = False
                while not stop_pump.is_set() and not self._stop.is_set():
                    if self.state.paused:
                        _t.sleep(0.02)
                        next_time = _t.perf_counter()
                        continue
                    if eof:
                        if is_file:
                            self.source.seek(0.0)
                            eof = False
                            next_time = _t.perf_counter()
                        else:
                            _t.sleep(0.05)
                            continue
                    f = self.source.read()
                    if f is None:
                        log.info("Source returned None (EOF or read failure).")
                        eof = True
                        _t.sleep(0.02)
                        continue
                    if is_file and self.state.speed > 0:
                        interval = base_interval / self.state.speed
                        now = _t.perf_counter()
                        if next_time > now:
                            _t.sleep(next_time - now)
                        next_time = max(next_time + interval, _t.perf_counter())
                    self.queue.put(f)

            t = threading.Thread(target=pump, daemon=True, name="FramePump")
            t.start()

            while not self._stop.is_set():
                frame: Frame | None = self.queue.get(timeout=0.3)
                if frame is None:
                    if stop_pump.is_set() and len(self.queue) == 0:
                        break
                    continue

                counter += 1
                self.stats.frames_seen += 1
                if counter % n != 0:
                    continue  # skip inference but keep pump moving

                t0 = time.perf_counter()
                detections = self._run_detection(frame.image)
                latency = (time.perf_counter() - t0) * 1000.0
                self.stats.latency_ms = latency

                annotated = draw_detections(
                    frame.image, detections,
                    show_conf=self.cfg.rendering.show_confidence,
                    show_track_id=self.cfg.rendering.show_track_id,
                    thickness=self.cfg.rendering.box_thickness,
                    font_scale=self.cfg.rendering.label_font_scale,
                )

                self.history.update(detections)

                # FPS (EMA)
                now = time.perf_counter()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps_ema = inst if fps_ema == 0 else (0.85 * fps_ema + 0.15 * inst)
                self.stats.fps = fps_ema
                self.stats.dropped = self.queue.dropped

                self.frame_ready.emit(annotated, detections, self.stats)
        except Exception as e:
            log.exception("Pipeline crashed")
            self.error.emit(str(e))
        finally:
            try:
                stop_pump.set()
            except Exception:
                pass
            self.finished.emit()

    # -- inference orchestration ------------------------------------------
    def _run_detection(self, image: np.ndarray) -> list[Detection]:
        dets: list[Detection] = []

        if self.cfg.targets.text:
            dino = self._ensure_dino()
            dets += dino.infer(image, self.cfg.targets.text,
                               box_thr=self.cfg.thresholds.open_vocab_box_threshold,
                               text_thr=self.cfg.thresholds.open_vocab_text_threshold)

        if self.state.reference_image is not None:
            try:
                ref = self._ensure_refmatch()
                ref.set_reference(self.state.reference_image)
                dets += ref.infer(image, self.cfg.thresholds.reference_match_threshold)
            except Exception as e:
                self.error.emit(f"Reference matcher error: {e}")

        # If no explicit targets, fall back to closed-set YOLO.
        if not dets and not self.cfg.targets.text and self.state.reference_image is None:
            yolo = self._ensure_yolo()
            if self.cfg.features.tracking_enabled and self._tracker is not None:
                dets = self._tracker.track(image, self.cfg.thresholds.detection_confidence)
            else:
                dets = yolo.infer(image, self.cfg.thresholds.detection_confidence)

        if self.cfg.features.segmentation_enabled and dets:
            try:
                sam = self._ensure_sam()
                masks = sam.segment(image, [d.xyxy for d in dets])
                for d, m in zip(dets, masks):
                    d.mask = m
            except Exception as e:
                self.error.emit(f"Segmentation error: {e}")

        return dets