"""K-fold cross-validation as a structured result.

A single train/test split gives one number; how much that number would move
with a different split is invisible. Cross-validation refits the *whole*
pipeline (preprocessing included, so nothing leaks across folds) on k-1 folds
and scores the held-out fold, k times. The spread across folds is the point.

Workflows run this on their training rows only, so the test split stays
untouched for the final, independent estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from mlcookbook.core.results import Result

TASKS = ("regression", "binary", "multiclass")


@dataclass(frozen=True)
class CrossValidationFold(Result):
    fold: int  # 1-based
    n_train: int
    n_validation: int
    scores: dict[str, float]


@dataclass(frozen=True)
class CrossValidationResult(Result):
    n_folds: int
    strategy: str  # "k-fold" (shuffled) or "stratified k-fold" (class shares kept per fold)
    n_rows: int
    metrics: list[str]
    folds: list[CrossValidationFold]
    mean: dict[str, float]
    std: dict[str, float]  # sample standard deviation across folds (ddof=1)
    warnings: list[str] = field(default_factory=list)


def _scoring(task: str, positive_class: Any) -> tuple[dict, set[str]]:
    """sklearn scorers keyed by our metric names, plus those sklearn reports negated."""
    if task == "regression":
        return (
            {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
            {"rmse", "mae"},
        )
    if task == "binary":
        return (
            {"accuracy": "accuracy", "f1": make_scorer(f1_score, pos_label=positive_class), "roc_auc": "roc_auc"},
            set(),
        )
    return {"accuracy": "accuracy", "f1_macro": "f1_macro", "roc_auc_ovr": "roc_auc_ovr"}, set()


def cross_validate_pipeline(
    model,
    X,
    y,
    task: str,
    folds: int = 5,
    random_state: int = 0,
    positive_class: Any = None,
) -> CrossValidationResult:
    """Cross-validate an unfitted sklearn estimator/pipeline on (X, y).

    *task* is "regression" (shuffled k-fold; R², RMSE, MAE), "binary"
    (stratified; accuracy, F1 of *positive_class*, ROC AUC) or "multiclass"
    (stratified; accuracy, macro F1, one-vs-rest macro ROC AUC).
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {list(TASKS)}; got {task!r}.")
    if folds < 2:
        raise ValueError(f"Cross-validation needs at least 2 folds; got {folds}.")
    if task == "binary" and positive_class is None:
        raise ValueError("Binary cross-validation needs the positive_class for F1.")
    if task == "regression":
        splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scoring, negated = _scoring(task, positive_class)
    raw = cross_validate(model, X, y, cv=splitter, scoring=scoring, return_indices=True, error_score="raise")

    per_metric = {name: -raw[f"test_{name}"] if name in negated else raw[f"test_{name}"] for name in scoring}
    fold_results = [
        CrossValidationFold(
            fold=i + 1,
            n_train=len(raw["indices"]["train"][i]),
            n_validation=len(raw["indices"]["test"][i]),
            scores={name: float(values[i]) for name, values in per_metric.items()},
        )
        for i in range(folds)
    ]
    return CrossValidationResult(
        n_folds=folds,
        strategy="k-fold" if task == "regression" else "stratified k-fold",
        n_rows=len(y),
        metrics=list(scoring),
        folds=fold_results,
        mean={name: float(np.mean(values)) for name, values in per_metric.items()},
        std={name: float(np.std(values, ddof=1)) for name, values in per_metric.items()},
    )


def training_cross_validation(
    model, X_train, y_train, task: str, folds: int, random_state: int,
    warnings: list[str], positive_class: Any = None,
) -> CrossValidationResult | None:
    """Workflow helper: cross-validate on training rows, shrinking *folds* to what the data allows.

    Returns None (with a warning appended) when cross-validation is disabled
    (``folds`` of 0 or 1) or impossible.
    """
    if folds < 2:
        return None
    if task == "regression":
        available = len(y_train) // 2  # keep at least two rows per validation fold
    else:
        available = int(y_train.value_counts().min())  # every fold needs every class
    if available < 2:
        warnings.append("Skipped cross-validation: too few training rows (or rows of the smallest class).")
        return None
    if available < folds:
        warnings.append(f"Cross-validation used {available} folds instead of {folds}: too few rows for more.")
        folds = available
    return cross_validate_pipeline(model, X_train, y_train, task, folds, random_state, positive_class)
