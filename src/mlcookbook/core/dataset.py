"""The ``Dataset`` abstraction: a pandas DataFrame plus a typed schema.

``Dataset`` is the entry point of the library. It infers a ``VarType`` for
every column (user-overridable) and computes lightweight metadata. Analyses
take a ``Dataset`` so they can validate applicability against the schema
instead of guessing from raw dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from mlcookbook.core.results import Result
from mlcookbook.core.schema import ColumnSchema, DatasetSchema, VarType, infer_var_type


@dataclass(frozen=True)
class ColumnSummary(Result):
    name: str
    var_type: VarType
    dtype: str
    missing_count: int
    missing_fraction: float
    unique_count: int
    is_constant: bool


@dataclass(frozen=True)
class DatasetSummary(Result):
    n_rows: int
    n_columns: int
    constant_columns: list[str] = field(default_factory=list)
    columns: list[ColumnSummary] = field(default_factory=list)


class Dataset:
    """Wrap a DataFrame with an inferred, overridable column schema.

    >>> data = Dataset(df, overrides={"zip_code": "categorical"})
    >>> data.schema().var_type("age")
    <VarType.NUMERIC: 'numeric'>
    """

    def __init__(
        self,
        df: pd.DataFrame,
        overrides: Mapping[str, VarType | str] | None = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Dataset expects a pandas DataFrame, got {type(df).__name__}")
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"DataFrame has duplicate column names: {dupes}")
        self._df = df.copy()
        self._overrides: dict[str, VarType] = {}
        for name, var_type in (overrides or {}).items():
            self.set_type(name, var_type)

    @property
    def df(self) -> pd.DataFrame:
        """The underlying DataFrame (a defensive copy is made at init)."""
        return self._df

    def set_type(self, column: str, var_type: VarType | str) -> None:
        """Override the inferred type of *column* (e.g. an integer ID as categorical)."""
        if column not in self._df.columns:
            raise KeyError(f"Unknown column: {column!r}")
        self._overrides[column] = VarType(var_type)

    def schema(self) -> DatasetSchema:
        columns = []
        for name in self._df.columns:
            override = self._overrides.get(name)
            columns.append(
                ColumnSchema(
                    name=name,
                    var_type=override if override is not None else infer_var_type(self._df[name]),
                    dtype=str(self._df[name].dtype),
                    overridden=override is not None,
                )
            )
        return DatasetSchema(columns=columns)

    def describe(self) -> DatasetSummary:
        """Structural metadata: missingness, cardinality, constant columns."""
        schema = self.schema()
        n_rows = len(self._df)
        summaries = []
        for col in schema.columns:
            series = self._df[col.name]
            missing = int(series.isna().sum())
            unique = int(series.nunique(dropna=True))
            summaries.append(
                ColumnSummary(
                    name=col.name,
                    var_type=col.var_type,
                    dtype=col.dtype,
                    missing_count=missing,
                    missing_fraction=missing / n_rows if n_rows else 0.0,
                    unique_count=unique,
                    is_constant=unique <= 1,
                )
            )
        return DatasetSummary(
            n_rows=n_rows,
            n_columns=len(summaries),
            constant_columns=[s.name for s in summaries if s.is_constant],
            columns=summaries,
        )

    def columns_of_type(self, *types: VarType) -> list[str]:
        return self.schema().columns_of_type(*types)

    def __repr__(self) -> str:
        return f"Dataset(n_rows={len(self._df)}, n_columns={self._df.shape[1]})"
