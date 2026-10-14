"""Video display: converts BGR numpy arrays to QImage with correct stride."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#0b0d10; color:#8a93a6;")
        self.setText("No video source.\nOpen a camera or a video file.")
        self._last_pix: QPixmap | None = None

    def show_frame(self, bgr: np.ndarray) -> None:
        h, w, ch = bgr.shape
        if ch != 3:
            bgr = bgr[:, :, :3]
        # QImage needs the array to remain alive while used; copy to be safe.
        img = QImage(bgr.data, w, h, ch * w, QImage.Format_BGR888).copy()
        pix = QPixmap.fromImage(img)
        self._last_pix = pix
        self._render()

    def _render(self) -> None:
        if self._last_pix is None:
            return
        self.setPixmap(self._last_pix.scaled(self.size(), Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation))

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self._render()