"""Linear Regression screen: the shared report layout with this recipe's title, task, and settings.

app.py supplies ``engine`` (callable path, signature, summary) and ``report``
(pre-formatted results), and maps the ``params`` keys onto the core function's
arguments. This module never imports mlcookbook.
"""

from __future__ import annotations

from gui.screens.report_screen import ReportScreen

PARAMS = [
    {"key": "penalty", "label": "Penalty", "kind": "choice",
     "options": [("None (ordinary least squares)", "none"), ("Ridge (L2)", "ridge"), ("Lasso (L1)", "lasso")],
     "help": "Ridge shrinks every coefficient; lasso can set weak ones exactly to zero."},
    {"key": "alpha", "label": "Penalty strength", "kind": "float", "min": 0.001, "max": 1000.0, "value": 1.0,
     "decimals": 3, "step": 0.1, "help": "Larger alpha = stronger shrinkage. Ignored without a penalty."},
]


class LinearRegressionScreen(ReportScreen):
    def __init__(self, engine: dict[str, str], report: dict, parent=None) -> None:
        super().__init__("Linear Regression", engine, report, parent, task="regression", params=PARAMS)
