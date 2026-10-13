"""Support Vector Machines screen: the shared report layout with this recipe's title, tasks, and settings.

app.py supplies ``engine`` (callable path, signature, summary) and ``report``
(pre-formatted results), and maps the ``params`` keys onto the core function's
arguments. This module never imports mlcookbook.
"""

from __future__ import annotations

from gui.screens.report_screen import ReportScreen

PARAMS = [
    {"key": "kernel", "label": "Kernel", "kind": "choice",
     "options": [("RBF (curved boundary)", "rbf"), ("Linear (flat boundary)", "linear"), ("Polynomial", "poly")],
     "help": "The kernel decides the shape of the boundary."},
    {"key": "C", "label": "C (margin penalty)", "kind": "float", "min": 0.01, "max": 1000.0, "value": 1.0,
     "decimals": 2, "step": 0.5, "help": "Small C: wide, tolerant margin. Large C: chase every training row."},
    {"key": "epsilon", "label": "Epsilon (target SDs)", "kind": "float", "min": 0.0, "max": 2.0, "value": 0.1,
     "decimals": 2, "step": 0.05, "tasks": ("regression",),
     "help": "Errors smaller than this many target standard deviations cost nothing."},
]


class SVMScreen(ReportScreen):
    def __init__(self, engine: dict[str, str], report: dict, parent=None) -> None:
        super().__init__("Support Vector Machines", engine, report, parent,
                         tasks=("classification", "regression"), params=PARAMS)
