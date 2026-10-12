"""Validation shared by binary classification workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.schema import VarType

FEATURE_TYPES = (VarType.NUMERIC, VarType.CATEGORICAL, VarType.BOOLEAN)
TARGET_TYPES = (VarType.CATEGORICAL, VarType.BOOLEAN, VarType.NUMERIC)
IMBALANCE_WARN_FRACTION = 0.10  # warn if the minority class is below this share


def prepare_binary_problem(
    dataset: Dataset,
    target: str,
    features: list[str] | None,
    positive_class: Any,
    name: str,
) -> tuple[list[str], pd.DataFrame, list[Any], list[str]]:
    """Validate a binary classification request against the schema.

    Returns ``(features, frame, labels, warnings)``: the resolved feature list,
    the feature+target frame with missing-target rows dropped, the class labels
    ordered ``[negative, positive]``, and warnings to attach to the result.
    *name* is the calling workflow, used in error messages.
    """
    schema = dataset.schema()
    target_type = schema.var_type(target)  # raises KeyError for unknown columns
    if target_type not in TARGET_TYPES:
        raise ValueError(
            f"{name} requires a categorical, boolean, or two-valued numeric "
            f"target; {target!r} is {target_type.value}."
        )

    warnings: list[str] = []
    if features is None:
        features = [c.name for c in schema.columns if c.name != target and c.var_type in FEATURE_TYPES]
        skipped = [c.name for c in schema.columns if c.name != target and c.var_type not in FEATURE_TYPES]
        if skipped:
            warnings.append(
                f"Skipped columns with non-feature types (datetime/text/unsupported): {skipped}."
            )
    else:
        if target in features:
            raise ValueError(f"Target {target!r} cannot also be a feature.")
        for feature in features:
            var_type = schema.var_type(feature)  # raises KeyError for unknown columns
            if var_type not in FEATURE_TYPES:
                raise ValueError(f"Feature {feature!r} has unsupported type {var_type.value}.")
    if not features:
        raise ValueError("No usable feature columns.")

    frame = dataset.df[features + [target]]
    n_missing_target = int(frame[target].isna().sum())
    if n_missing_target:
        warnings.append(f"Dropped {n_missing_target} row(s) with missing target {target!r}.")
        frame = frame.dropna(subset=[target])

    y = frame[target]
    classes = sorted(y.unique().tolist())
    if len(classes) != 2:
        raise ValueError(
            f"{name} is binary; target {target!r} has {len(classes)} class(es): {classes}."
        )
    if positive_class is None:
        positive_class = classes[1]
    elif positive_class not in classes:
        raise ValueError(f"positive_class {positive_class!r} is not one of the target classes {classes}.")
    labels = [c for c in classes if c != positive_class] + [positive_class]

    counts = y.value_counts()
    if counts.min() < 2:
        raise ValueError(
            f"Each class needs at least 2 rows for a stratified split; counts: {counts.to_dict()}."
        )
    minority_share = counts.min() / len(y)
    if minority_share < IMBALANCE_WARN_FRACTION:
        warnings.append(
            f"Class imbalance: minority class is {minority_share:.1%} of rows; "
            "accuracy can be misleading, prefer recall/precision/ROC AUC."
        )
    return list(features), frame, labels, warnings
