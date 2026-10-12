"""Evaluation utilities returning structured metric objects.

Implemented: regression and classification metrics, k-fold cross-validation.
"""

from mlcookbook.evaluation.cross_validation import (
    CrossValidationFold,
    CrossValidationResult,
    cross_validate_pipeline,
)
from mlcookbook.evaluation.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    classification_metrics,
    regression_metrics,
)

__all__ = [
    "ClassificationMetrics",
    "CrossValidationFold",
    "CrossValidationResult",
    "RegressionMetrics",
    "classification_metrics",
    "cross_validate_pipeline",
    "regression_metrics",
]
