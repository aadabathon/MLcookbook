"""Build sklearn preprocessors from a Dataset's schema.

The returned ``ColumnTransformer`` is unfitted: it must be fitted inside a
``Pipeline`` on training data only, which is what keeps preprocessing free of
train/test leakage.

Each input column gets its own transformer entry (named ``"<kind>:<column>"``),
so every fitted value can be traced back to the column it came from; see
``preprocessing.summary.describe_preprocessor``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.schema import VarType
from mlcookbook.preprocessing.config import PreprocessingConfig

DATETIME_PARTS = ("days", "year", "month", "dayofweek")
MISSING_LEVEL = "(missing)"
_EPOCH = pd.Timestamp("1970-01-01")


def _categorical_missing_values(frame):
    """Keep categorical values intact; present all missing sentinels as np.nan."""
    return frame.astype(object).where(frame.notna(), np.nan)


def _categorical_text(frame):
    """Categorical values as text (so a "(missing)" level sorts with them); missing as np.nan."""
    return frame.astype(str).where(frame.notna(), np.nan)


def _datetime_parts(frame):
    """Expand each datetime column into numeric parts. Stateless, so nothing can leak."""
    parts = {}
    for name in frame.columns:
        values = frame[name]
        if values.dt.tz is not None:
            values = values.dt.tz_convert(None)
        parts[f"{name}_days"] = (values - _EPOCH) / pd.Timedelta(days=1)  # trend
        parts[f"{name}_year"] = values.dt.year
        parts[f"{name}_month"] = values.dt.month  # season
        parts[f"{name}_dayofweek"] = values.dt.dayofweek  # weekly cycle, Monday = 0
    return pd.DataFrame(parts, index=frame.index).astype(float)


def _datetime_names(transformer, input_features):
    return np.array([f"{name}_{part}" for name in input_features for part in DATETIME_PARTS], dtype=object)


def resolve_config(config: PreprocessingConfig | None, scale: bool = True) -> PreprocessingConfig:
    """Fill in the workflow's default scaling when the config leaves it unset."""
    config = config or PreprocessingConfig()
    if config.scaling is None:
        config = replace(config, scaling="standard" if scale else "none")
    return config


def _scaler(config: PreprocessingConfig):
    return {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}[config.scaling]()


def _numeric_pipeline(config: PreprocessingConfig, expand_dates: bool = False) -> Pipeline:
    steps = []
    if expand_dates:
        steps.append(("expand", FunctionTransformer(_datetime_parts, feature_names_out=_datetime_names)))
    steps.append(("impute", SimpleImputer(strategy=config.numeric_impute, keep_empty_features=True)))
    if config.scaling != "none":
        steps.append(("scale", _scaler(config)))
    return Pipeline(steps)


def _categorical_pipeline(config: PreprocessingConfig) -> Pipeline:
    if config.categorical_impute == "missing_category":
        steps = [
            ("missing", FunctionTransformer(_categorical_text, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_LEVEL, keep_empty_features=True)),
        ]
    else:
        steps = [
            ("missing", FunctionTransformer(_categorical_missing_values, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ]
    if config.categorical_encoding == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=config.min_category_frequency
        )
        steps.append(("ordinal", encoder))
    else:
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist", sparse_output=False, min_frequency=config.min_category_frequency
        )
        steps.append(("onehot", encoder))
    return Pipeline(steps)


def build_preprocessor(
    dataset: Dataset,
    columns: list[str],
    scale: bool = True,
    config: PreprocessingConfig | None = None,
) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer covering *columns*.

    Defaults (see ``PreprocessingConfig`` for the alternatives):

    * Numeric: median imputation + standard scaling (``scale=False`` keeps real
      units, e.g. for tree thresholds; an explicit ``config.scaling`` wins).
    * Categorical/boolean: mode imputation + one-hot encoding (unknown
      categories at predict time are ignored, not errors).
    * Datetime: expanded to days-since-1970, year, month and day of week, then
      treated like numeric columns. Columns must already hold parsed dates.
    """
    config = resolve_config(config, scale)
    schema = dataset.schema()
    entries = []
    unsupported = []
    for column in columns:
        var_type = schema.var_type(column)
        if var_type is VarType.NUMERIC:
            entries.append((f"numeric:{column}", _numeric_pipeline(config), [column]))
            if config.missing_indicators:
                indicator = MissingIndicator(features="missing-only", error_on_new=False)
                entries.append((f"missing:{column}", indicator, [column]))
        elif var_type in (VarType.CATEGORICAL, VarType.BOOLEAN):
            entries.append((f"categorical:{column}", _categorical_pipeline(config), [column]))
        elif var_type is VarType.DATETIME:
            if not pd.api.types.is_datetime64_any_dtype(dataset.df[column]):
                raise ValueError(
                    f"Datetime column {column!r} is not stored as dates ({dataset.df[column].dtype}). "
                    "Parse it first; load_dataset converts ISO-8601 date columns automatically."
                )
            entries.append((f"datetime:{column}", _numeric_pipeline(config, expand_dates=True), [column]))
        else:
            unsupported.append(column)
    if unsupported:
        raise ValueError(f"Columns not supported by build_preprocessor (text/unsupported): {sorted(unsupported)}")
    if not entries:
        raise ValueError("No usable columns to build a preprocessor from.")
    if any("__" in name for name, _, _ in entries):  # sklearn reserves "__" for parameter paths
        entries = [(f"{i}:{name.split(':', 1)[0]}", t, cols) for i, (name, t, cols) in enumerate(entries)]
    return ColumnTransformer(entries, verbose_feature_names_out=False)
