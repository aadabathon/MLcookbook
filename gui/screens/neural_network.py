"""Neural Networks screen: the shared report layout with this recipe's title, tasks, and settings.

app.py supplies ``engine`` (callable path, signature, summary) and ``report``
(pre-formatted results), and maps the ``params`` keys onto the core function's
arguments. This module never imports mlcookbook.
"""

from __future__ import annotations

from gui.screens.report_screen import ReportScreen

PARAMS = [
    {"key": "hidden_layers", "label": "Hidden layers", "kind": "choice",
     "options": [("64 → 32", "64,32"), ("32", "32"), ("128 → 64", "128,64"), ("64 → 64 → 32", "64,64,32")],
     "help": "Units per hidden layer. Wider and deeper networks fit more, and overfit more."},
    {"key": "activation", "label": "Activation", "kind": "choice",
     "options": [("ReLU", "relu"), ("tanh", "tanh"), ("Sigmoid", "logistic")]},
    {"key": "alpha", "label": "L2 penalty (alpha)", "kind": "float", "min": 0.0, "max": 10.0, "value": 0.0001,
     "decimals": 4, "step": 0.001, "help": "Weight decay: larger values give smoother, less overfit networks."},
    {"key": "max_epochs", "label": "Max epochs", "kind": "int", "min": 10, "max": 5000, "value": 500,
     "help": "Upper limit on passes over the training rows."},
    {"key": "early_stopping", "label": "Early stopping", "kind": "choice",
     "options": [("On (10% validation rows)", True), ("Off", False)],
     "help": "Stop when the held-out validation score stops improving."},
]


class NeuralNetworkScreen(ReportScreen):
    def __init__(self, engine: dict[str, str], report: dict, parent=None) -> None:
        super().__init__("Neural Networks", engine, report, parent,
                         tasks=("classification", "regression"), params=PARAMS)
