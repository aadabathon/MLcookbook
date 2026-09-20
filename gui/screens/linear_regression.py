"""Linear Regression screen: the shared report layout with this recipe's title.

app.py supplies ``engine`` (callable path, signature, summary) and ``report``
(the pre-formatted results of a run on the default dataset). This module never
imports mlcookbook.
"""

from __future__ import annotations

from gui.screens.report_screen import ReportScreen


class LinearRegressionScreen(ReportScreen):
    def __init__(self, engine: dict[str, str], report: dict, parent=None) -> None:
        super().__init__("Linear Regression", engine, report, parent)
