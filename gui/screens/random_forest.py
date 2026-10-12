"""Random Forests screen: the shared report layout with this recipe's title, tasks, and settings.

app.py supplies ``engine`` (callable path, signature, summary) and ``report``
(pre-formatted results), and maps the ``params`` keys onto the core function's
arguments. This module never imports mlcookbook.
"""

from __future__ import annotations

from gui.screens.report_screen import ReportScreen

PARAMS = [
    {"key": "n_estimators", "label": "Trees", "kind": "int", "min": 10, "max": 1000, "value": 100,
     "help": "More trees average away more variance (and take longer)."},
    {"key": "max_depth", "label": "Max depth", "kind": "int", "min": 0, "max": 50, "value": 0,
     "none_at_min": "unlimited", "help": "Forests usually grow deep trees; averaging controls the variance."},
    {"key": "min_samples_leaf", "label": "Min rows per leaf", "kind": "int", "min": 1, "max": 500, "value": 5,
     "help": "Minimum training rows in each leaf of each tree."},
]


class RandomForestScreen(ReportScreen):
    def __init__(self, engine: dict[str, str], report: dict, parent=None) -> None:
        super().__init__("Random Forests", engine, report, parent,
                         tasks=("classification", "regression"), params=PARAMS)
