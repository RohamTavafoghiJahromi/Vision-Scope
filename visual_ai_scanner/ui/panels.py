"""Left (search) and right (tracks/events/object) dock panels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSpinBox, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)


class SearchPanel(QWidget):
    target_added = Signal(str)
    target_removed = Signal(str)
    reference_set = Signal(object)       # numpy BGR
    reference_cleared = Signal()
    threshold_changed = Signal(float)

    def __init__(self, cfg):
        super().__init__()
        v = QVBoxLayout(self)

        # ---- text targets ----
        g1 = QGroupBox("Text targets (open-vocabulary)")
        l1 = QVBoxLayout(g1)
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("e.g.  cup, red backpack, cardboard box")
        self.input.returnPressed.connect(self._add)
        b = QPushButton("Add")
        b.clicked.connect(self._add)
        row.addWidget(self.input); row.addWidget(b)
        l1.addLayout(row)
        self.list = QListWidget()
        self.list.setToolTip("Uncheck to disable a target without removing it.")
        l1.addWidget(self.list)
        self.list.itemChanged.connect(self._on_item_changed)
        v.addWidget(g1)

        # ---- reference image ----
        g2 = QGroupBox("Reference image (image-guided)")
        l2 = QVBoxLayout(g2)
        self.ref_preview = QLabel("Drop an image here\nor click Load.")
        self.ref_preview.setAlignment(Qt.AlignCenter)
        self.ref_preview.setMinimumHeight(140)
        self.ref_preview.setStyleSheet("border:1px dashed #3a4152; color:#7d8ba3;")
        self.ref_preview.setAcceptDrops(True)
        l2.addWidget(self.ref_preview)
        hb = QHBoxLayout()
        load = QPushButton("Load image…")
        clear = QPushButton("Clear")
        load.clicked.connect(self._pick_image)
        clear.clicked.connect(self._clear_ref)
        hb.addWidget(load); hb.addWidget(clear)
        l2.addLayout(hb)

        form = QFormLayout()
        self.thr = QDoubleSpinBox(); self.thr.setRange(0.05, 0.95)
        self.thr.setSingleStep(0.05); self.thr.setValue(cfg.thresholds.reference_match_threshold)
        self.thr.valueChanged.connect(self.threshold_changed.emit)
        form.addRow("Match threshold", self.thr)
        l2.addLayout(form)
        note = QLabel("Visual similarity does not prove identity. "
                      "Confirm matches visually.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#7d8ba3; font-size:11px;")
        l2.addWidget(note)
        v.addWidget(g2)
        v.addStretch(1)

        self._ref_img: np.ndarray | None = None

    # -- targets -----------------------------------------------------------
    def _add(self):
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self._append_target(text, True)
        self.target_added.emit(text)

    def _append_target(self, text: str, checked: bool):
        it = QListWidgetItem(text)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.list.addItem(it)

    def _on_item_changed(self, _item):
        # Recompose the active target list from checked entries.
        pass

    def active_targets(self) -> list[str]:
        out = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.text())
        return out

    # -- reference image ---------------------------------------------------
    def _pick_image(self):
        from PySide6.QtWidgets import QFileDialog
        p, _ = QFileDialog.getOpenFileName(self, "Reference image", "",
                                           "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if p:
            self._load_ref(Path(p))

    def _load_ref(self, path: Path):
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            self.ref_preview.setText("Invalid image file.")
            return
        self._ref_img = img
        self._show_preview(img)
        self.reference_set.emit(img)

    def _clear_ref(self):
        self._ref_img = None
        self.ref_preview.setText("Drop an image here\nor click Load.")
        self.ref_preview.setPixmap(QPixmap())
        self.reference_cleared.emit()

    def _show_preview(self, bgr: np.ndarray):
        h, w, ch = bgr.shape
        q = QImage(bgr.data, w, h, ch * w, QImage.Format_BGR888).copy()
        self.ref_preview.setPixmap(QPixmap.fromImage(q).scaled(
            self.ref_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # drag & drop
    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                self._load_ref(p)
                break


class TracksPanel(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        self.list = QListWidget()
        v.addWidget(QLabel("Active tracks"))
        v.addWidget(self.list)

    def update_tracks(self, detections, now: float):
        self.list.clear()
        for d in detections:
            if d.track_id is None:
                continue
            self.list.addItem(f"#{d.track_id:>3}  {d.label:<18}  {d.score:.2f}")


class EventsPanel(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        self.text = QTextEdit(); self.text.setReadOnly(True)
        v.addWidget(QLabel("Detection event history"))
        v.addWidget(self.text)

    def append(self, line: str):
        self.text.append(line)


class ObjectPanel(QWidget):
    describe_requested = Signal()

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        self.preview = QLabel("No object selected.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(180)
        self.preview.setStyleSheet("border:1px solid #2a3140; background:#0d1014;")
        v.addWidget(self.preview)
        self.info = QTextEdit(); self.info.setReadOnly(True)
        v.addWidget(self.info)
        self.describe = QPushButton("Describe with vision-language model")
        self.describe.clicked.connect(self.describe_requested.emit)
        v.addWidget(self.describe)
        self._crop: np.ndarray | None = None
        self._label: str = ""

    def set_object(self, crop_bgr: np.ndarray, label: str, score: float, tid):
        self._crop = crop_bgr
        self._label = label
        h, w, ch = crop_bgr.shape
        q = QImage(crop_bgr.data, w, h, ch * w, QImage.Format_BGR888).copy()
        self.preview.setPixmap(QPixmap.fromImage(q).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.info.setPlainText(f"Label : {label}\nScore : {score:.3f}\n"
                               f"Track : {tid if tid is not None else '-'}")

    def append_description(self, text: str):
        self.info.append("\n--- VLM description ---")
        self.info.append(text)