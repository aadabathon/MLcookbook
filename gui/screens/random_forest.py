"""Random Forests screen. app.py supplies the engine description and report."""
from __future__ import annotations

from gui.screens.report_screen import ReportScreen


class RandomForestScreen(ReportScreen):
    def __init__(self, engine: dict[str, str], report: dict, parent=None) -> None:
        super().__init__("Random Forests", engine, report, parent)
