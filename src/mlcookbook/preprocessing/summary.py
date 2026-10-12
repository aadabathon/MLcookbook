"""Describe a fitted preprocessor as plain data.

Every supervised result carries a ``PreprocessingSummary`` so a reader can see
exactly what was done to each column and which values were learned from the
training rows (the imputation fill, the scaling centre/spread, the category
levels) without touching sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType
from mlcookbook.preprocessing.builders import DATETIME_PARTS, MISSING_LEVEL
from mlcookbook.preprocessing.config import PreprocessingConfig

_SHOW_LEVELS = 8


@dataclass(frozen=True)
class ColumnPreprocessing(Result):
    column: str
    var_type: VarType
    n_missing_fit: int  # missing values among the rows the preprocessor was fitted on
    steps: list[str]  # in order, with the fitted values
    output_features: list[str]  # model-facing columns this input became


@dataclass(frozen=True)
class PreprocessingSummary(Result):
    config: PreprocessingConfig  # with the workflow's default scaling filled in
    n_rows_fitted: int  # training rows only, for supervised workflows
    n_output_features: int
    columns: list[ColumnPreprocessing] = field(default_factory=list)


def _fmt(value) -> str:
    return f"{value:.4g}" if isinstance(value, float) else repr(value) if isinstance(value, str) else str(value)


def _levels(values) -> str:
    shown = [_fmt(v.item() if hasattr(v, "item") else v) for v in list(values)[:_SHOW_LEVELS]]
    more = len(values) - _SHOW_LEVELS
    return ", ".join(shown) + (f", ... (+{more})" if more > 0 else "")


def _numeric_steps(pipeline, i: int, config: PreprocessingConfig, n_missing: int, dates: bool) -> list[str]:
    """Steps for the *i*-th column of a numeric (or datetime) group pipeline."""
    steps = []
    if dates:
        steps.append("expanded to days since 1970, year, month, and day of week (Monday = 0)")
    imputer = pipeline.named_steps["impute"]
    counted = f" ({n_missing} missing in training)" if n_missing else ""
    if dates:
        steps.append(f"missing dates: each part filled with its training {config.numeric_impute}{counted}")
    else:
        fill = _fmt(float(imputer.statistics_[i]))
        steps.append(f"missing values filled with training {config.numeric_impute} {fill}{counted}")
    scaler = pipeline.named_steps.get("scale")
    if scaler is None:
        steps.append("not scaled (kept in real units)")
    elif dates:
        steps.append(f"{config.scaling} scaling of each part, fitted on training rows")
    elif config.scaling == "standard":
        steps.append(f"standard scaling: (x - {_fmt(float(scaler.mean_[i]))}) / {_fmt(float(scaler.scale_[i]))}")
    elif config.scaling == "robust":
        steps.append(
            f"robust scaling: (x - median {_fmt(float(scaler.center_[i]))}) / IQR {_fmt(float(scaler.scale_[i]))}"
        )
    else:
        steps.append(
            f"min-max scaling from [{_fmt(float(scaler.data_min_[i]))}, {_fmt(float(scaler.data_max_[i]))}] to [0, 1]"
        )
    return steps


def _infrequent(encoder, i: int):
    infrequent = getattr(encoder, "infrequent_categories_", None)
    levels = infrequent[i] if infrequent is not None else None
    return levels if levels is not None and len(levels) else None


def _categorical_steps(pipeline, i: int, config: PreprocessingConfig, n_missing: int) -> list[str]:
    """Steps for the *i*-th column of the categorical group pipeline."""
    imputer = pipeline.named_steps["impute"]
    if config.categorical_impute == "missing_category":
        steps = [f"missing values kept as their own level {MISSING_LEVEL!r}"]
    else:
        fill = imputer.statistics_[i]
        steps = [f"missing values filled with training mode {_fmt(fill.item() if hasattr(fill, 'item') else fill)}"]
    if n_missing:
        steps[0] += f" ({n_missing} missing in training)"
    encoder = pipeline.named_steps.get("onehot") or pipeline.named_steps["ordinal"]
    levels = encoder.categories_[i]
    infrequent = _infrequent(encoder, i)
    if infrequent is not None:
        steps.append(f"{len(infrequent)} rare level(s) pooled as 'infrequent': {_levels(infrequent)}")
    if "onehot" in pipeline.named_steps:
        unseen = "'infrequent'" if infrequent is not None else "all zeros"
        steps.append(f"one-hot encoded: {len(levels)} training level(s) [{_levels(levels)}]; unseen levels -> {unseen}")
    else:
        steps.append(f"ordinal codes in sorted order [{_levels(levels)}]; unseen levels -> -1")
    return steps


def _output_widths(pipeline, n_columns: int, dates: bool) -> list[int]:
    """How many model columns each input column of a group pipeline becomes."""
    if dates:
        return [len(DATETIME_PARTS)] * n_columns
    encoder = pipeline.named_steps.get("onehot")
    if encoder is None:  # numeric or ordinal: one output per input
        return [1] * n_columns
    widths = []
    for i, levels in enumerate(encoder.categories_):
        infrequent = _infrequent(encoder, i)
        widths.append(len(levels) - (len(infrequent) - 1 if infrequent is not None else 0))
    return widths


def describe_preprocessor(
    preprocessor: ColumnTransformer,
    X_fit: pd.DataFrame,
    dataset: Dataset,
    config: PreprocessingConfig,
) -> PreprocessingSummary:
    """Summarize a ColumnTransformer from ``build_preprocessor`` after it was fitted on *X_fit*."""
    schema = dataset.schema()
    by_column: dict[str, dict] = {}
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or not len(columns):
            continue
        outputs = [str(n) for n in transformer.get_feature_names_out()]
        if isinstance(transformer, MissingIndicator):
            flagged = {columns[j] for j in transformer.features_}
            for column, output in zip([c for c in columns if c in flagged], outputs):
                by_column[column]["steps"].append("0/1 missing-value indicator added")
                by_column[column]["outputs"].append(output)
            continue
        dates = "expand" in transformer.named_steps
        categorical = "onehot" in transformer.named_steps or "ordinal" in transformer.named_steps
        offset = 0
        for i, (column, width) in enumerate(zip(columns, _output_widths(transformer, len(columns), dates))):
            n_missing = int(X_fit[column].isna().sum())
            if categorical:
                steps = _categorical_steps(transformer, i, config, n_missing)
            else:
                steps = _numeric_steps(transformer, i, config, n_missing, dates)
            by_column[column] = {"steps": steps, "outputs": outputs[offset:offset + width], "n_missing": n_missing}
            offset += width

    columns = [
        ColumnPreprocessing(
            column=column,
            var_type=schema.var_type(column),
            n_missing_fit=by_column[column]["n_missing"],
            steps=by_column[column]["steps"],
            output_features=by_column[column]["outputs"],
        )
        for column in X_fit.columns
        if column in by_column
    ]
    return PreprocessingSummary(
        config=config,
        n_rows_fitted=len(X_fit),
        n_output_features=sum(len(c.output_features) for c in columns),
        columns=columns,
    )
