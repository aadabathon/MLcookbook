"""Shared-fold ridge versus random forest comparison controls."""

from gui.screens.report_screen import ReportScreen

PARAMS = [
    {"key": "random_state", "label": "Split / model seed", "kind": "int", "min": 0, "max": 2147483647, "value": 0},
    {"key": "test_size", "label": "Test fraction", "kind": "float", "min": 0.1, "max": 0.5, "value": 0.25,
     "decimals": 2, "step": 0.05},
    {"key": "alpha", "label": "Ridge penalty", "kind": "float", "min": 0.001, "max": 1000.0, "value": 1.0},
    {"key": "n_estimators", "label": "Forest trees", "kind": "int", "min": 10, "max": 500, "value": 100},
    {"key": "max_depth", "label": "Forest max depth", "kind": "int", "min": 0, "max": 30, "value": 0,
     "none_at_min": "Unlimited"},
    {"key": "min_samples_leaf", "label": "Forest min leaf", "kind": "int", "min": 1, "max": 100, "value": 5},
]


class ModelComparisonScreen(ReportScreen):
    def __init__(self, engine, report, parent=None):
        super().__init__("Compare Regression Models", engine, report, parent, task="regression", params=PARAMS)

    def _problem(self):
        problem = super()._problem()
        if problem:
            return problem
        if self.cv_folds is None or self.cv_folds < 2:
            return "Model comparison needs cross-validation. Choose at least 3 folds in validation settings."
        return None
