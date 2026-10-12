"""Model evaluation metrics as structured results."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
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
class ClassMetrics(Result):
    """One class treated as "positive" against all the others."""

    label: Any
    precision: float
    recall: float
    f1: float
    support: int  # rows whose true class is this one


@dataclass(frozen=True)
class ClassificationMetrics(Result):
    accuracy: float
    precision: float  # of the positive class (binary) or macro average (multiclass)
    recall: float  # likewise
    f1: float  # likewise
    roc_auc: float  # binary AUC or macro one-vs-rest AUC; NaN (-> null) if undefined
    confusion_matrix: list[list[int]]  # rows = actual, cols = predicted, ordered as `labels`
    labels: list[Any]  # binary: [negative_class, positive_class]; multiclass: sorted classes
    positive_class: Any  # None for multiclass
    n_obs: int
    balanced_accuracy: float  # mean recall over classes; robust to imbalance
    average: str = "binary"  # "binary" or "macro" (unweighted mean over classes)
    per_class: list[ClassMetrics] = field(default_factory=list)


def classification_metrics(
    y_true, y_pred, y_score, labels: list[Any]
) -> ClassificationMetrics:
    """Compute classification metrics for one set of predictions.

    Binary: *labels* is ``[negative_class, positive_class]`` and *y_score* the
    predicted probability of the positive class. Multiclass (3+ labels):
    *y_score* holds one probability column per label, in *labels* order, and
    precision/recall/F1/ROC AUC are macro averages (every class counts equally).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score, dtype=float)
    if len(labels) < 2:
        raise ValueError(f"classification_metrics needs at least two labels; got {labels!r}")
    binary = len(labels) == 2
    expected_score_shape = y_true.shape if binary else (len(y_true), len(labels))
    if y_true.shape != y_pred.shape or y_score.shape != expected_score_shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, y_score {y_score.shape} "
            f"(expected {expected_score_shape})"
        )
    present = set(np.unique(y_true).tolist())
    if binary:
        positive = labels[1]
        if len(present) < 2:  # AUC is undefined with a single class
            roc_auc = float("nan")
        else:
            roc_auc = float(sk_metrics.roc_auc_score(y_true == positive, y_score))
        kwargs = {"pos_label": positive, "average": "binary"}
    else:
        positive = None
        if present != set(labels):  # one-vs-rest AUC needs every class among the true labels
            roc_auc = float("nan")
        else:
            roc_auc = float(sk_metrics.roc_auc_score(y_true, y_score, multi_class="ovr", average="macro", labels=labels))
        kwargs = {"labels": labels, "average": "macro"}
    precision, recall, f1, support = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    with warnings.catch_warnings():  # sklearn warns when y_pred holds classes absent from y_true
        warnings.simplefilter("ignore", UserWarning)
        balanced = sk_metrics.balanced_accuracy_score(y_true, y_pred)
    return ClassificationMetrics(
        accuracy=float(sk_metrics.accuracy_score(y_true, y_pred)),
        precision=float(sk_metrics.precision_score(y_true, y_pred, zero_division=0, **kwargs)),
        recall=float(sk_metrics.recall_score(y_true, y_pred, zero_division=0, **kwargs)),
        f1=float(sk_metrics.f1_score(y_true, y_pred, zero_division=0, **kwargs)),
        roc_auc=roc_auc,
        confusion_matrix=sk_metrics.confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        labels=list(labels),
        positive_class=positive,
        n_obs=len(y_true),
        balanced_accuracy=float(balanced),
        average="binary" if binary else "macro",
        per_class=[
            ClassMetrics(label, float(p), float(r), float(f), int(s))
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        ],
    )
