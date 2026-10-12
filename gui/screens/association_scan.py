"""Association Scan screen: the shared report layout with this recipe's title.

app.py supplies ``engine`` (callable path, signature, summary) and ``report``
(pre-formatted results), and maps the ``params`` keys onto the core function's
arguments. This module never imports mlcookbook.
"""

from __future__ import annotations

from gui.screens.report_screen import ReportScreen


class AssociationScanScreen(ReportScreen):
    def __init__(self, engine: dict[str, str], report: dict, parent=None) -> None:
        super().__init__("Association Scan", engine, report, parent, task="unsupervised",
                         feature_roles=("numeric", "categorical", "boolean"), min_features=2,
                         cross_validation=False, preprocessing=False)
