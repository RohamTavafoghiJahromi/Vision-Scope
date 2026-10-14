"""Main window: wires the source → pipeline → UI, plus menus and shortcuts."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSlider, QSplitter, QStatusBar, QTabWidget, QToolBar, QVBoxLayout, QWidget,
)

from ..config import Config
from ..export import Exporter
from ..hardware import HardwareInfo
from ..pipeline import ProcessingWorker
from ..video import BoundedFrameQueue, CameraSource, FileSource, VideoSource
from .dialogs import ModelsDialog, SettingsDialog
from .panels import EventsPanel, ObjectPanel, SearchPanel, TracksPanel
from .video_widget import VideoWidget

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config, cfg_path: Path, hw: HardwareInfo):
        super().__init__()
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.hw = hw
        self.setWindowTitle("Advanced Real-Time Visual AI Scanner")

        self.exporter = Exporter(Path(cfg.app.save_dir))
        self.source: VideoSource | None = None
        self.queue: BoundedFrameQueue | None = None
        self.thread: QThread | None = None
        self.worker: ProcessingWorker | None = None

        self._last_detections = []
        self._last_frame: np.ndarray | None = None
        self._last_stats = None

        self._build_ui()
        self._apply_theme()
        self._install_shortcuts()
        self.statusBar().showMessage(f"Device: {hw.device_name}  |  "
                                     f"CUDA: {'yes' if hw.cuda_available else 'no'}  |  "
                                     f"Recommended: {hw.recommended_detector}")

    # ---- UI ---------------------------------------------------------------
    def _build_ui(self):
        # Central: splitter with video + side panels
        self.video = VideoWidget()

        self.search_panel = SearchPanel(self.cfg)
        self.search_panel.target_added.connect(self._on_target_changed)
        self.search_panel.target_removed.connect(self._on_target_changed)
        self.search_panel.reference_set.connect(self._on_reference_set)
        self.search_panel.reference_cleared.connect(self._on_reference_cleared)
        self.search_panel.threshold_changed.connect(self._on_ref_threshold)

        self.tracks = TracksPanel()
        self.events = EventsPanel()
        self.obj = ObjectPanel()
        self.obj.describe_requested.connect(self._describe_selected)

        right_tabs = QTabWidget()
        right_tabs.addTab(self.tracks, "Tracks")
        right_tabs.addTab(self.events, "Events")
        right_tabs.addTab(self.obj, "Selected object")

        splitter = QSplitter()
        splitter.addWidget(self.search_panel)
        splitter.addWidget(self.video)
        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([320, 900, 320])
        self.setCentralWidget(splitter)

        # Bottom playback controls
        ctrl = QWidget()
        hb = QHBoxLayout(ctrl)
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.seek = QSlider(Qt.Horizontal); self.seek.setRange(0, 1000)
        self.pos_label = QLabel("00:00 / 00:00")
        # NEW: playback speed selector
        from PySide6.QtWidgets import QComboBox
        self.speed_box = QComboBox()
        self.speed_box.addItems(["0.25x", "0.5x", "1x", "2x", "4x", "8x", "Max"])
        self.speed_box.setCurrentText("1x")
        self.speed_box.currentTextChanged.connect(self._on_speed_changed)

        for w in (self.play_btn, self.pause_btn, self.stop_btn):
            hb.addWidget(w)
        hb.addWidget(self.seek, 1)
        hb.addWidget(self.pos_label)
        hb.addWidget(QLabel("Speed:"))
        hb.addWidget(self.speed_box)
        self.play_btn.clicked.connect(self._play)
        self.pause_btn.clicked.connect(self._pause)
        self.stop_btn.clicked.connect(self._stop)
        self.seek.sliderReleased.connect(self._on_seek)
        self.setStatusBar(QStatusBar())
        self.statusBar().addPermanentWidget(ctrl, 1)

        # Toolbar + menu
        tb = QToolBar("Main"); self.addToolBar(tb)
        a_cam = QAction("Open camera…", self); a_cam.triggered.connect(self._open_camera)
        a_file = QAction("Open video…", self); a_file.triggered.connect(self._open_file)
        a_shot = QAction("Screenshot (annotated)", self); a_shot.triggered.connect(
            lambda: self._screenshot(True))
        a_shot2 = QAction("Screenshot (clean)", self); a_shot2.triggered.connect(
            lambda: self._screenshot(False))
        a_rec = QAction("Record annotated…", self); a_rec.triggered.connect(self._record)
        a_stoprec = QAction("Stop recording", self); a_stoprec.triggered.connect(self._stop_record)
        a_settings = QAction("Settings", self); a_settings.triggered.connect(self._open_settings)
        a_models = QAction("Models", self); a_models.triggered.connect(self._open_models)
        for a in (a_cam, a_file, a_shot, a_shot2, a_rec, a_stoprec, a_settings, a_models):
            tb.addAction(a)

        m = self.menuBar().addMenu("File")
        for a in (a_cam, a_file, a_shot, a_shot2, a_rec, a_stoprec):
            m.addAction(a)
        m.addSeparator()
        m.addAction(QAction("Quit", self, triggered=self.close))
        e = self.menuBar().addMenu("Edit")
        e.addAction(a_settings)
        e.addAction(a_models)
        h = self.menuBar().addMenu("Help")
        h.addAction(QAction("Hardware report", self, triggered=self._show_hw))

        self._recording = False

    def _install_shortcuts(self):
        QAction("Play", self, shortcut=QKeySequence("Space"),
                triggered=self._toggle_pause).setShortcutContext(Qt.ApplicationShortcut)
        self.addAction(self.findChild(QAction))

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#0f1216; color:#e6e9ef;
                                   font-family: 'Segoe UI', 'Inter', sans-serif; }
            QGroupBox { border:1px solid #2a3140; border-radius:6px; margin-top:10px; }
            QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px;
                               color:#a7b0c2; }
            QPushButton { background:#1a2030; border:1px solid #2f3a52; border-radius:5px;
                          padding:6px 10px; color:#dbe4f4; }
            QPushButton:hover { background:#233049; }
            QLineEdit, QTextEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {
                background:#0c1119; border:1px solid #2a3140; border-radius:5px;
                padding:4px 6px; color:#e6e9ef; }
            QTabWidget::pane { border:1px solid #2a3140; }
            QTabBar::tab { background:#131a26; padding:6px 12px; }
            QTabBar::tab:selected { background:#1f2a3d; }
            QToolBar { background:#0d131c; border-bottom:1px solid #1c2333; spacing:6px; }
            QStatusBar { background:#0d131c; color:#93a0b8; }
            QSlider::groove:horizontal { height:6px; background:#1a2030; border-radius:3px; }
            QSlider::handle:horizontal { background:#5b8def; width:12px; margin:-4px 0;
                                         border-radius:6px; }
        """)

    # ---- sources ----------------------------------------------------------
    def _open_camera(self):
        idx, ok = self._ask_int("Camera index", "Enter camera index (0, 1, …):", 0)
        if not ok:
            return
        try:
            self._start_source(CameraSource(idx, self.cfg.video.target_width,
                                            self.cfg.video.target_height))
        except Exception as e:
            QMessageBox.critical(self, "Camera error", str(e))

    def _open_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open video", "",
                                           "Videos (*.mp4 *.avi *.mov *.mkv *.webm)")
        if not p:
            return
        try:
            self._start_source(FileSource(p))
        except Exception as e:
            QMessageBox.critical(self, "Video error", str(e))

    def _start_source(self, src: VideoSource):
        self._stop()
        self.source = src
        self.queue = BoundedFrameQueue(self.cfg.video.queue_size)
        self.worker = ProcessingWorker(self.cfg, src, self.queue)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.error.connect(lambda s: self.statusBar().showMessage(f"Error: {s}", 6000))
        self.worker.finished.connect(self._on_worker_finished)

        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

        # Reflect active state on widgets
        self.worker.state.text_targets = self.search_panel.active_targets()

    # ---- pipeline signals -------------------------------------------------
    def _on_frame(self, annotated, detections, stats):
        self._last_frame = annotated
        self._last_detections = detections
        self._last_stats = stats
        self.video.show_frame(annotated)
        self.tracks.update_tracks(detections, 0.0)

        for d in detections:
            pass  # per-track events handled inside worker's history

        # Events — display the newest few events.
        for ev in self.worker.history.events:
            pass

        # Update position label for file playback
        if self.source and hasattr(self.source, "duration") and self.source.duration > 0:
            cur = getattr(self.source, "_i", 0) / max(1e-6, self.source.fps)
            dur = self.source.duration
            self.pos_label.setText(f"{_fmt(cur)} / {_fmt(dur)}")
            self.seek.blockSignals(True)
            self.seek.setValue(int(1000 * cur / max(1e-6, dur)))
            self.seek.blockSignals(False)

        if stats:
            self.statusBar().showMessage(
                f"Device: {stats.device}  |  FPS: {stats.fps:5.1f}  |  "
                f"latency: {stats.latency_ms:5.1f} ms  |  "
                f"queue: {len(self.queue) if self.queue else 0}  |  "
                f"dropped: {stats.dropped}"
            )
        if self._recording and self.exporter and annotated is not None:
            self.exporter.write_frame(annotated)

    def _on_worker_finished(self):
        log.info("Pipeline finished")

    # ---- controls ---------------------------------------------------------
    def _play(self):
        if self.worker:
            self.worker.state.paused = False

    def _pause(self):
        if self.worker:
            self.worker.state.paused = True

    def _toggle_pause(self):
        if not self.worker:
            return
        self.worker.state.paused = not self.worker.state.paused

    def _stop(self):
        if self.worker:
            self.worker.request_stop()
        if self.queue:
            self.queue.close()
        if self.thread:
            self.thread.quit()
            self.thread.wait(2000)
        if self.source:
            try:
                self.source.release()
            except Exception:
                pass
        self.source = None
        self.worker = None
        self.thread = None

    def _on_seek(self):
        if isinstance(self.source, FileSource) and hasattr(self.source, "duration"):
            t = self.seek.value() / 1000.0 * self.source.duration
            self.source.seek(t)
            
    def _on_speed_changed(self, text: str) -> None:
        if self.worker is None:
            return
        if text == "Max":
            self.worker.state.speed = 0.0     # 0 = no throttling
        else:
            try:
                self.worker.state.speed = float(text.rstrip("x"))
            except ValueError:
                self.worker.state.speed = 1.0

    # ---- search wiring ----------------------------------------------------
    def _on_target_changed(self, *_):
        if self.worker:
            self.worker.state.text_targets = self.search_panel.active_targets()
            self.cfg.targets.text = self.worker.state.text_targets

    def _on_reference_set(self, img_bgr: np.ndarray):
        if self.worker:
            self.worker.state.reference_image = img_bgr

    def _on_reference_cleared(self):
        if self.worker:
            self.worker.state.reference_image = None

    def _on_ref_threshold(self, v: float):
        self.cfg.thresholds.reference_match_threshold = float(v)

    # ---- screenshot / record ---------------------------------------------
    def _screenshot(self, annotated: bool):
        frame = self._last_frame if annotated else self._clean_frame()
        if frame is None:
            QMessageBox.information(self, "No frame", "Nothing to capture yet.")
            return
        p = self.exporter.screenshot(frame, "annotated" if annotated else "clean")
        self.statusBar().showMessage(f"Saved {p}", 5000)

    def _clean_frame(self):
        # We only keep the annotated frame in memory for simplicity; take a
        # fresh read for clean capture.
        if isinstance(self.source, FileSource):
            return None
        return None

    def _record(self):
        if self._last_frame is None:
            QMessageBox.information(self, "No frame", "Start a source first.")
            return
        h, w = self._last_frame.shape[:2]
        fps = self.source.fps if self.source else 25.0
        p, _ = QFileDialog.getSaveFileName(self, "Save annotated video", "annotated.mp4",
                                           "MP4 (*.mp4);;AVI (*.avi)")
        if not p:
            return
        self.exporter.start_video(p, (w, h), fps)
        self._recording = True
        self.statusBar().showMessage(f"Recording to {p}", 5000)

    def _stop_record(self):
        if not self._recording:
            return
        p = self.exporter.stop_video()
        self._recording = False
        if p:
            self.statusBar().showMessage(f"Saved {p}", 5000)

    # ---- dialogs ----------------------------------------------------------
    def _open_settings(self):
        d = SettingsDialog(self.cfg, self)
        if d.exec():
            d.apply_to(self.cfg)
            self.cfg.save(self.cfg_path)
            self.statusBar().showMessage("Settings saved. Restart source to apply.",
                                         6000)

    def _open_models(self):
        ModelsDialog(self).exec()

    def _show_hw(self):
        QMessageBox.information(self, "Hardware report", self.hw.describe())

    def _ask_int(self, title, prompt, default):
        from PySide6.QtWidgets import QInputDialog
        v, ok = QInputDialog.getInt(self, title, prompt, default)
        return v, ok

    # ---- VLM --------------------------------------------------------------
    def _describe_selected(self):
        if self._last_frame is None or not self._last_detections:
            return
        # Use first detection for simplicity; extend to per-track selection.
        d = self._last_detections[0]
        crop = self._last_frame[max(0, d.y1):d.y2, max(0, d.x1):d.x2]
        if crop.size == 0:
            return
        try:
            from ..vlm import VlmDescriber
            if not hasattr(self, "_vlm"):
                self._vlm = VlmDescriber(self.cfg.models.vlm_model, self.cfg.models.device)
            text = self._vlm.describe(crop)
            self.obj.append_description(text)
        except Exception as e:
            QMessageBox.warning(self, "VLM error", str(e))

    # ---- shutdown ---------------------------------------------------------
    def closeEvent(self, e):  # noqa: N802
        try:
            self._stop()
            self.cfg.targets.text = self.search_panel.active_targets()
            self.cfg.save(self.cfg_path)
        finally:
            super().closeEvent(e)


def _fmt(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"