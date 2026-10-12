"""Column-level data-quality checks.

These flag columns that tend to mislead an analysis: no information
(constant / entirely missing), mostly imputed values, row identifiers, and
categoricals with so many levels that one-hot encoding explodes. They are
reported, never acted on: the user decides what to drop.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlcookbook.core.results import Result
from mlcookbook.core.schema import DatasetSchema, VarType

HIGH_MISSING_SHARE = 0.5
HIGH_CARDINALITY = 50
ID_MIN_ROWS = 20


@dataclass(frozen=True)
class QualityIssue(Result):
    column: str
    kind: str  # "all_missing" | "constant" | "high_missing" | "likely_id" | "high_cardinality"
    message: str


def _looks_like_id(series: pd.Series, var_type: VarType) -> bool:
    """Every row has its own distinct integer or label: a key, not a measurement."""
    if len(series) < ID_MIN_ROWS or series.isna().any() or series.nunique() != len(series):
        return False
    if var_type is VarType.NUMERIC:
        return pd.api.types.is_integer_dtype(series)
    return var_type is VarType.CATEGORICAL


def column_issues(frame: pd.DataFrame, schema: DatasetSchema, columns: list[str] | None = None) -> list[QualityIssue]:
    """Quality issues for *columns* (default: all), in column order."""
    n_rows = len(frame)
    issues = []
    for name in columns if columns is not None else list(frame.columns):
        series = frame[name]
        var_type = schema.var_type(name)
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        if n_rows and missing == n_rows:
            issues.append(QualityIssue(name, "all_missing", f"Column {name!r} is entirely missing; it carries no information."))
            continue
        if unique <= 1:
            issues.append(QualityIssue(name, "constant", f"Column {name!r} is constant; it carries no information."))
            continue
        if n_rows and missing / n_rows > HIGH_MISSING_SHARE:
            issues.append(QualityIssue(
                name, "high_missing",
                f"Column {name!r} is {missing / n_rows:.0%} missing; imputed values will dominate it.",
            ))
        if _looks_like_id(series, var_type):
            issues.append(QualityIssue(
                name, "likely_id",
                f"Column {name!r} looks like a row identifier (every value is unique); "
                "identifiers rarely generalize, so leave it out unless it carries meaning.",
            ))
        if var_type is VarType.CATEGORICAL and unique > HIGH_CARDINALITY:
            issues.append(QualityIssue(
                name, "high_cardinality",
                f"Categorical column {name!r} has {unique} levels; one-hot encoding adds a column per level. "
                "Consider pooling rare levels (min_category_frequency).",
            ))
    return issues
