"""Decision Trees screen: the shared report layout with this recipe's title, tasks, and settings.

app.py supplies ``engine`` (callable path, signature, summary) and ``report``
(pre-formatted results), and maps the ``params`` keys onto the core function's
arguments. This module never imports mlcookbook.
"""

from __future__ import annotations

from gui.screens.report_screen import ReportScreen

PARAMS = [
    {"key": "max_depth", "label": "Max depth", "kind": "int", "min": 1, "max": 20, "value": 3,
     "help": "How many questions deep the tree may grow."},
    {"key": "min_samples_leaf", "label": "Min rows per leaf", "kind": "int", "min": 1, "max": 500, "value": 5,
     "help": "Larger leaves average more rows: smoother, less overfit predictions."},
]


class DecisionTreeScreen(ReportScreen):
    def __init__(self, engine: dict[str, str], report: dict, parent=None) -> None:
        super().__init__("Decision Trees", engine, report, parent,
                         tasks=("classification", "regression"), params=PARAMS)
