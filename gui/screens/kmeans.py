"""K-Means screen: the shared report layout with this recipe's title and settings.

app.py supplies ``engine`` (callable path, signature, summary) and ``report``
(pre-formatted results), and maps the ``params`` keys onto the core function's
arguments. This module never imports mlcookbook.
"""

from __future__ import annotations

from gui.screens.report_screen import ReportScreen

PARAMS = [
    {"key": "n_clusters", "label": "Clusters (k)", "kind": "int", "min": 2, "max": 20, "value": 3,
     "help": "How many groups to find. The k scan in the report helps choose it."},
    {"key": "max_k", "label": "Scan k up to", "kind": "int", "min": 2, "max": 20, "value": 10,
     "help": "The report compares inertia and silhouette for k = 2 .. this value."},
]


class KMeansScreen(ReportScreen):
    def __init__(self, engine: dict[str, str], report: dict, parent=None) -> None:
        super().__init__("K-Means", engine, report, parent, task="unsupervised", params=PARAMS,
                         cross_validation=False)
