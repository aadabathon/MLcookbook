"""Local tabular file loading. Statistical roles remain Dataset's responsibility."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from mlcookbook.core.dataset import Dataset


def load_dataset(path: str | Path, *, delimiter: str = ",", encoding: str = "utf-8-sig") -> Dataset:
    """Read a flat CSV or Parquet file without renaming columns or coercing roles.

    CSV uses a header row and pandas' standard missing-value recognition.
    Delimiter and encoding are explicit; malformed records are rejected.
    Parquet preserves its stored pandas-compatible types.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        if len(delimiter) != 1:
            raise ValueError("Choose a single-character CSV delimiter.")
        with path.open(encoding=encoding, newline="") as source:
            reader = csv.reader(source, delimiter=delimiter, strict=True)
            header = next(reader, [])
            _validate_columns(header)
            for row in reader:
                if row and len(row) > len(header):
                    raise ValueError(f"CSV record ending on line {reader.line_num} has more fields than the header.")
        frame = pd.read_csv(path, sep=delimiter, encoding=encoding, on_bad_lines="error")
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, engine="pyarrow")
    else:
        raise ValueError("Choose a .csv or .parquet file.")
    _validate_columns(list(frame.columns))
    if frame.empty:
        raise ValueError("The file contains no data rows.")
    for name in frame.columns:
        if not frame[name].map(pd.api.types.is_scalar).all():
            raise ValueError(f"Column {name!r} contains nested values. Flatten it before importing.")
    return Dataset(frame)


def _validate_columns(columns: list) -> None:
    if not columns:
        raise ValueError("The file needs a header with named columns.")
    if any(not isinstance(name, str) or not name.strip() for name in columns):
        raise ValueError("Every column needs a non-empty text name.")
    if len(set(columns)) != len(columns):
        raise ValueError("Column names must be unique; the file contains duplicate headers.")
