"""Column type model and inference.

``VarType`` is the statistical role of a column (numeric, categorical, ...),
distinct from its pandas dtype. Analyses decide applicability based on
``VarType``, so getting these right — or overriding them — matters more than
the raw dtype.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from mlcookbook.core.results import Result


class VarType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    UNSUPPORTED = "unsupported"


# Object columns with at most this many distinct values (or at most
# CATEGORICAL_MAX_RATIO of rows) are treated as categorical rather than text.
CATEGORICAL_MAX_UNIQUE = 20
CATEGORICAL_MAX_RATIO = 0.05


@dataclass(frozen=True)
class ColumnSchema(Result):
    name: str
    var_type: VarType
    dtype: str
    overridden: bool = False


@dataclass(frozen=True)
class DatasetSchema(Result):
    columns: list[ColumnSchema] = field(default_factory=list)

    def __getitem__(self, name: str) -> ColumnSchema:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    def var_type(self, name: str) -> VarType:
        return self[name].var_type

    def columns_of_type(self, *types: VarType) -> list[str]:
        wanted = set(types)
        return [c.name for c in self.columns if c.var_type in wanted]


def infer_var_type(series: pd.Series) -> VarType:
    """Infer the statistical role of a single column from its values."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return VarType.CATEGORICAL
    if pd.api.types.is_bool_dtype(series):
        return VarType.BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(series):
        return VarType.DATETIME
    if pd.api.types.is_numeric_dtype(series):
        return VarType.NUMERIC
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        values = series.dropna()
        if values.empty:
            return VarType.UNSUPPORTED
        if not values.map(lambda v: isinstance(v, str)).all():
            return VarType.UNSUPPORTED  # mixed / nested objects
        n_unique = values.nunique()
        if n_unique <= CATEGORICAL_MAX_UNIQUE or n_unique / len(values) <= CATEGORICAL_MAX_RATIO:
            return VarType.CATEGORICAL
        return VarType.TEXT
    return VarType.UNSUPPORTED
