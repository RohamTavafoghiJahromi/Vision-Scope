"""Runnable smoke tests. They do NOT require a GPU or model weights except
where noted; skip gracefully when heavy deps are missing."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# --- Config persistence ----------------------------------------------------
def test_config_roundtrip(tmp_path):
    from visual_ai_scanner.config import Config
    p = tmp_path / "cfg.yaml"
    cfg = Config()
    cfg.thresholds.detection_confidence = 0.42
    cfg.targets.text = ["cup", "backpack"]
    cfg.save(p)
    cfg2 = Config.load(p)
    assert cfg2.thresholds.detection_confidence == 0.42
    assert cfg2.targets.text == ["cup", "backpack"]


# --- Detection dataclass / coordinate helpers -----------------------------
def test_detection_geometry():
    from visual_ai_scanner.detection import Detection
    d = Detection(10, 20, 110, 220, "cup", 0.9)
    assert d.xyxy == (10, 20, 110, 220)
    assert d.center == (60, 120)


# --- Rendering doesn't crash on edge coordinates --------------------------
def test_render_edge_labels():
    from visual_ai_scanner.rendering import draw_detections
    from visual_ai_scanner.detection import Detection
    img = np.zeros((200, 300, 3), np.uint8)
    d = Detection(0, 0, 50, 50, "top-left", 0.5)
    d2 = Detection(280, 190, 300, 200, "bottom-right", 0.5)
    out = draw_detections(img, [d, d2])
    assert out.shape == img.shape


# --- Bounded queue drops oldest ------------------------------------------
def test_frame_queue_drops():
    from visual_ai_scanner.video import BoundedFrameQueue, Frame
    q = BoundedFrameQueue(maxlen=2)
    for i in range(5):
        q.put(Frame(np.zeros((4, 4, 3), np.uint8), i, float(i), False))
    assert len(q) == 2
    assert q.dropped == 3


# --- Event history de-duplication ----------------------------------------
def test_event_dedup():
    from visual_ai_scanner.events import EventHistory
    from visual_ai_scanner.detection import Detection
    h = EventHistory(max_events=10)
    det = [Detection(0, 0, 10, 10, "cup", 0.9, track_id=1)]
    h.update(det, now=0.0)
    h.update(det, now=0.1)
    h.update(det, now=0.2)
    kinds = [e.kind for e in h.events]
    assert kinds.count("appear") == 1
    # After a gap, we should see a disappear event.
    h.update([], now=5.0)
    kinds = [e.kind for e in h.events]
    assert "disappear" in kinds


# --- Screenshot export ----------------------------------------------------
def test_screenshot(tmp_path):
    from visual_ai_scanner.export import Exporter
    img = np.zeros((100, 100, 3), np.uint8)
    p = Exporter(tmp_path).screenshot(img, "test")
    assert p.exists() and p.suffix == ".png"


# --- Video file loading (skips if opencv can't write video here) ----------
@pytest.mark.skipif(os.environ.get("CI_SKIP_VIDEO") == "1", reason="video disabled")
def test_video_file_roundtrip(tmp_path):
    import cv2
    from visual_ai_scanner.video import FileSource
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    if not w.isOpened():
        pytest.skip("No video writer backend available.")
    for _ in range(20):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    src = FileSource(path)
    f = src.read()
    assert f is not None and f.image.shape[:2] == (48, 64)
    src.release()


# --- Reference image validation (light) -----------------------------------
def test_reference_matcher_import():
    # Do not instantiate (needs weights); just confirm module is importable.
    import visual_ai_scanner.detection as det
    assert hasattr(det, "ReferenceMatcher")


# --- Hardware report ------------------------------------------------------
def test_hardware_report():
    from visual_ai_scanner.hardware import detect_hardware
    info = detect_hardware()
    assert isinstance(info.cuda_available, bool)
    assert isinstance(info.device_name, str)


# --- Confidence threshold filtering (unit-testable shape) -----------------
def test_confidence_threshold():
    from visual_ai_scanner.detection import Detection
    dets = [Detection(0, 0, 10, 10, "a", 0.2), Detection(0, 0, 10, 10, "b", 0.9)]
    filtered = [d for d in dets if d.score >= 0.5]
    assert len(filtered) == 1 and filtered[0].label == "b"