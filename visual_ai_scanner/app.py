"""Application bootstrap: logging, hardware report, Qt app, main window."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .config import Config
from .hardware import detect_hardware
from .logging_setup import configure_logging

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def main() -> int:
    cfg = Config.load(CONFIG_PATH)
    configure_logging(cfg.app.log_level, PROJECT_ROOT / "outputs")
    hw = detect_hardware()
    log.info("Hardware report:\n%s", hw.describe())

    app = QApplication(sys.argv)
    app.setApplicationName("Visual AI Scanner")

    # Import here so a Qt import failure doesn't break --help/--version flows.
    from .ui.main_window import MainWindow
    w = MainWindow(cfg, CONFIG_PATH, hw)
    w.resize(1500, 900)
    w.show()
    return app.exec()