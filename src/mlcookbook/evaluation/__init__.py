"""Evaluation utilities returning structured metric objects.

Implemented: regression metrics.
Planned (see ROADMAP.md): classification metrics, cross-validation,
residual diagnostics.
"""

from mlcookbook.evaluation.metrics import RegressionMetrics, regression_metrics

__all__ = ["RegressionMetrics", "regression_metrics"]
