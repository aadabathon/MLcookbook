"""Validation shared by the regression workflows."""

from __future__ import annotations

import pandas as pd

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.schema import VarType
from mlcookbook.preprocessing.features import resolve_features

OVERFIT_GAP = 0.10  # warn if train R² beats test R² by more than this


def prepare_regression_problem(
    dataset: Dataset, target: str, features: list[str] | None, name: str
) -> tuple[list[str], pd.DataFrame, list[str]]:
    """Validate a regression request; return ``(features, frame, warnings)``.

    *frame* holds the features and target with missing-target rows dropped.
    *name* is the calling workflow, used in error messages.
    """
    schema = dataset.schema()
    if schema.var_type(target) is not VarType.NUMERIC:
        raise ValueError(
            f"{name} requires a numeric target; {target!r} is {schema.var_type(target).value}."
        )
    features, warnings = resolve_features(dataset, target, features)
    frame = dataset.df[features + [target]]
    n_missing_target = int(frame[target].isna().sum())
    if n_missing_target:
        warnings.append(f"Dropped {n_missing_target} row(s) with missing target {target!r}.")
        frame = frame.dropna(subset=[target])
    if frame[target].nunique() <= 2:
        warnings.append(
            f"Target {target!r} has at most two distinct values; a classifier is usually the better fit."
        )
    return features, frame, warnings


def overfit_warning(train_r2: float, test_r2: float, advice: str) -> list[str]:
    if train_r2 - test_r2 > OVERFIT_GAP:
        return [f"Possible overfitting: train R² {train_r2:.3f} vs test {test_r2:.3f}. {advice}"]
    return []
