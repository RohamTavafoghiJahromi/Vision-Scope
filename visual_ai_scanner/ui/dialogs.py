"""Settings dialog + model management dialog."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .. import models as models_mod


class SettingsDialog(QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.cfg = cfg
        form = QFormLayout(self)

        self.device = QComboBox(); self.device.addItems(["auto", "cuda", "cpu"])
        self.device.setCurrentText(cfg.models.device)
        form.addRow("Inference device", self.device)

        self.weights = QLineEdit(cfg.models.detector_weights)
        form.addRow("YOLO weights", self.weights)

        self.dino_id = QLineEdit(cfg.models.open_vocab_model)
        form.addRow("Open-vocab model", self.dino_id)

        self.ref_id = QLineEdit(cfg.models.reference_model)
        form.addRow("Reference model", self.ref_id)

        self.sam_id = QLineEdit(cfg.models.sam_model)
        form.addRow("Segmentation model", self.sam_id)

        self.vlm_id = QLineEdit(cfg.models.vlm_model)
        form.addRow("Vision-language model", self.vlm_id)

        self.conf = QDoubleSpinBox(); self.conf.setRange(0.01, 1.0); self.conf.setSingleStep(0.05)
        self.conf.setValue(cfg.thresholds.detection_confidence)
        form.addRow("Detection confidence", self.conf)

        self.ref_thr = QDoubleSpinBox(); self.ref_thr.setRange(0.01, 1.0); self.ref_thr.setSingleStep(0.05)
        self.ref_thr.setValue(cfg.thresholds.reference_match_threshold)
        form.addRow("Reference match threshold", self.ref_thr)

        self.skip = QSpinBox(); self.skip.setRange(1, 20)
        self.skip.setValue(cfg.video.process_every_n_frames)
        form.addRow("Process every N frames", self.skip)

        self.tracking = QCheckBox(); self.tracking.setChecked(cfg.features.tracking_enabled)
        form.addRow("Enable tracking", self.tracking)

        self.seg = QCheckBox(); self.seg.setChecked(cfg.features.segmentation_enabled)
        form.addRow("Enable segmentation", self.seg)

        self.half = QCheckBox(); self.half.setChecked(cfg.models.half_precision)
        form.addRow("Use half precision (FP16)", self.half)

        self.save_dir = QLineEdit(cfg.app.save_dir)
        form.addRow("Save directory", self.save_dir)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        form.addRow(bb)

    def apply_to(self, cfg) -> None:
        cfg.models.device = self.device.currentText()
        cfg.models.detector_weights = self.weights.text().strip()
        cfg.models.open_vocab_model = self.dino_id.text().strip()
        cfg.models.reference_model = self.ref_id.text().strip()
        cfg.models.sam_model = self.sam_id.text().strip()
        cfg.models.vlm_model = self.vlm_id.text().strip()
        cfg.thresholds.detection_confidence = float(self.conf.value())
        cfg.thresholds.reference_match_threshold = float(self.ref_thr.value())
        cfg.video.process_every_n_frames = int(self.skip.value())
        cfg.features.tracking_enabled = self.tracking.isChecked()
        cfg.features.segmentation_enabled = self.seg.isChecked()
        cfg.models.half_precision = self.half.isChecked()
        cfg.app.save_dir = self.save_dir.text().strip()


class _DownloadWorker(QThread):
    progress = Signal(str)
    done = Signal(bool, str)

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        try:
            self.progress.emit(f"Downloading {self.model.name}…")
            models_mod.download(self.model)
            self.done.emit(True, "Installed.")
        except Exception as e:
            self.done.emit(False, str(e))


class ModelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model management")
        self.resize(900, 520)
        v = QVBoxLayout(self)

        note = QLabel("Downloads are explicit and never silent. "
                      "Check the license column before commercial use.")
        note.setWordWrap(True)
        v.addWidget(note)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Purpose", "Size", "License", "Installed", "Status", "Action"])
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table)

        self.pbar = QProgressBar(); self.pbar.setRange(0, 0); self.pbar.hide()
        v.addWidget(self.pbar)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

        self.refresh()

    def refresh(self):
        self.table.setRowCount(0)
        for m in models_mod.list_models():
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(m.name))
            self.table.setItem(r, 1, QTableWidgetItem(m.purpose))
            self.table.setItem(r, 2, QTableWidgetItem(m.approx_size))
            self.table.setItem(r, 3, QTableWidgetItem(m.license))
            self.table.setItem(r, 4, QTableWidgetItem("yes" if m.installed else "no"))
            self.table.setItem(r, 5, QTableWidgetItem("—"))
            btn = QPushButton("Reinstall" if m.installed else "Download")
            btn.clicked.connect(lambda _=False, mm=m: self._download(mm))
            self.table.setCellWidget(r, 6, btn)

    def _download(self, m):
        if m.installed:
            ok = QMessageBox.question(self, "Reinstall?",
                                      f"{m.name} appears installed. Re-download anyway?")
            if ok != QMessageBox.Yes:
                return
        self.pbar.show()
        self.worker = _DownloadWorker(m)
        self.worker.progress.connect(lambda s: self.pbar.setFormat(s))
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _on_done(self, ok: bool, msg: str):
        self.pbar.hide()
        if not ok:
            QMessageBox.warning(self, "Download failed", msg)
        else:
            self.refresh()