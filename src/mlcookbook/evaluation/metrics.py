"""Model evaluation metrics as structured results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn import metrics as sk_metrics

from mlcookbook.core.results import Result


@dataclass(frozen=True)
class RegressionMetrics(Result):
    r2: float
    rmse: float
    mae: float
    n_obs: int


def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    """Compute standard regression metrics for one set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return RegressionMetrics(
        r2=float(sk_metrics.r2_score(y_true, y_pred)),
        rmse=float(sk_metrics.root_mean_squared_error(y_true, y_pred)),
        mae=float(sk_metrics.mean_absolute_error(y_true, y_pred)),
        n_obs=len(y_true),
    )


@dataclass(frozen=True)
class ClassificationMetrics(Result):
    accuracy: float
    precision: float  # of the positive class
    recall: float  # of the positive class
    f1: float  # of the positive class
    roc_auc: float  # NaN (-> null in JSON) if y_true holds only one class
    confusion_matrix: list[list[int]]  # rows = actual, cols = predicted, ordered as `labels`
    labels: list[Any]  # [negative_class, positive_class]
    positive_class: Any
    n_obs: int


def classification_metrics(
    y_true, y_pred, y_score, labels: list[Any]
) -> ClassificationMetrics:
    """Compute binary classification metrics for one set of predictions.

    *labels* is ``[negative_class, positive_class]``; *y_score* is the predicted
    probability of the positive class (used for ROC AUC only).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score, dtype=float)
    if len(labels) != 2:
        raise ValueError(f"classification_metrics is binary; got labels {labels!r}")
    if not (y_true.shape == y_pred.shape == y_score.shape):
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, y_score {y_score.shape}"
        )
    positive = labels[1]
    if len(np.unique(y_true)) < 2:  # AUC is undefined with a single class
        roc_auc = float("nan")
    else:
        roc_auc = float(sk_metrics.roc_auc_score(y_true == positive, y_score))
    return ClassificationMetrics(
        accuracy=float(sk_metrics.accuracy_score(y_true, y_pred)),
        precision=float(sk_metrics.precision_score(y_true, y_pred, pos_label=positive, zero_division=0)),
        recall=float(sk_metrics.recall_score(y_true, y_pred, pos_label=positive, zero_division=0)),
        f1=float(sk_metrics.f1_score(y_true, y_pred, pos_label=positive, zero_division=0)),
        roc_auc=roc_auc,
        confusion_matrix=sk_metrics.confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        labels=list(labels),
        positive_class=positive,
        n_obs=len(y_true),
    )
