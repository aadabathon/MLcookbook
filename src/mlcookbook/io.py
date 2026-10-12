"""Local tabular file loading. Statistical roles remain Dataset's responsibility.

Reading is deliberately conservative: values are only converted when the
conversion is lossless and unambiguous, and every conversion is reported.

* Leading zeros: a CSV column such as ``01234`` would become the number 1234.
  Columns where any value has a leading zero are kept as text instead.
* Dates: text columns whose every value is an ISO-8601 date/time
  (``2024-05-31``, ``2024-05-31T14:00:00Z``) are parsed into datetimes;
  values with UTC offsets are converted to UTC. Ambiguous formats such as
  ``05/06/2024``, or a mix of zoned and unzoned values, are left as text.
* Missing values: pandas' standard markers (empty, ``NA``, ``NaN``, ...) plus
  any extra *missing_markers* you pass (e.g. ``"?"`` or ``"-999"``).
"""
from __future__ import annotations

import csv
import re
import warnings as python_warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result

_LEADING_ZERO = r"^[+-]?0\d"
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?(Z|[+-]\d{2}:?\d{2})?$")


@dataclass(frozen=True)
class ImportReport(Result):
    source: str  # file name
    file_format: str  # "csv" | "parquet"
    n_rows: int
    n_columns: int
    missing_markers: list[str]  # extra markers treated as missing, beyond pandas' defaults
    kept_as_text: list[str]  # numeric-looking columns kept as text because of leading zeros
    parsed_dates: list[str]  # text columns converted to datetimes
    notes: list[str] = field(default_factory=list)  # one human-readable line per conversion


def load_dataset(
    path: str | Path,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    missing_markers: list[str] | tuple[str, ...] = (),
    parse_dates: bool = True,
    keep_leading_zeros: bool = True,
) -> Dataset:
    """Read a flat CSV or Parquet file without renaming columns or coercing roles.

    CSV uses a header row and pandas' standard missing-value recognition.
    Delimiter and encoding are explicit; malformed records are rejected.
    Parquet preserves its stored pandas-compatible types. See the module
    docstring for the conversions; ``load_dataset_with_report`` also returns them.
    """
    return load_dataset_with_report(
        path, delimiter=delimiter, encoding=encoding, missing_markers=missing_markers,
        parse_dates=parse_dates, keep_leading_zeros=keep_leading_zeros,
    )[0]


def load_dataset_with_report(
    path: str | Path,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    missing_markers: list[str] | tuple[str, ...] = (),
    parse_dates: bool = True,
    keep_leading_zeros: bool = True,
) -> tuple[Dataset, ImportReport]:
    """Like ``load_dataset``, plus an ``ImportReport`` of every conversion made."""
    path = Path(path)
    markers = [str(m) for m in missing_markers if str(m) != ""]
    notes: list[str] = []
    kept_as_text: list[str] = []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        if len(delimiter) != 1:
            raise ValueError("Choose a single-character CSV delimiter.")
        _validate_csv_shape(path, delimiter, encoding)
        options = dict(sep=delimiter, encoding=encoding, on_bad_lines="error", na_values=markers or None)
        frame = pd.read_csv(path, **options)
        if keep_leading_zeros:
            raw = pd.read_csv(path, dtype=str, **options)
            for name in frame.columns:
                if pd.api.types.is_numeric_dtype(frame[name]) and raw[name].dropna().str.match(_LEADING_ZERO).any():
                    frame[name] = raw[name]
                    kept_as_text.append(name)
                    notes.append(f"Kept {name!r} as text: values such as {_example(raw[name])!r} have leading zeros.")
        if markers:
            notes.append(f"Treated these values as missing: {markers}.")
        file_format = "csv"
    elif suffix == ".parquet":
        frame = pd.read_parquet(path, engine="pyarrow")
        for name in frame.columns:
            replaced = _replace_markers(frame, name, markers)
            if replaced:
                notes.append(f"Treated {replaced} value(s) in {name!r} as missing (markers {markers}).")
        file_format = "parquet"
    else:
        raise ValueError("Choose a .csv or .parquet file.")
    _validate_columns(list(frame.columns))
    if frame.empty:
        raise ValueError("The file contains no data rows.")
    for name in frame.columns:
        if not frame[name].map(pd.api.types.is_scalar).all():
            raise ValueError(f"Column {name!r} contains nested values. Flatten it before importing.")

    parsed_dates = []
    if parse_dates:
        for name in frame.columns:
            parsed = _iso_dates(frame[name])
            if parsed is not None:
                frame[name] = parsed
                parsed_dates.append(name)
                notes.append(f"Parsed {name!r} as dates (every value is an ISO-8601 date/time).")

    report = ImportReport(
        source=path.name,
        file_format=file_format,
        n_rows=len(frame),
        n_columns=frame.shape[1],
        missing_markers=markers,
        kept_as_text=kept_as_text,
        parsed_dates=parsed_dates,
        notes=notes,
    )
    return Dataset(frame), report


def _validate_csv_shape(path: Path, delimiter: str, encoding: str) -> None:
    with path.open(encoding=encoding, newline="") as source:
        reader = csv.reader(source, delimiter=delimiter, strict=True)
        header = next(reader, [])
        _validate_columns(header)
        for row in reader:
            if row and len(row) > len(header):
                raise ValueError(f"CSV record ending on line {reader.line_num} has more fields than the header.")


def _example(values: pd.Series) -> str:
    return values.dropna()[values.dropna().str.match(_LEADING_ZERO)].iloc[0]


def _replace_markers(frame: pd.DataFrame, name: str, markers: list[str]) -> int:
    """Blank out marker values in a Parquet column: text markers in text columns, numeric ones in numbers."""
    if not markers:
        return 0
    series = frame[name]
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        targets = []
        for marker in markers:
            try:
                targets.append(float(marker))
            except ValueError:
                pass
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        targets = markers
    else:
        return 0
    mask = series.isin(targets) & series.notna()
    count = int(mask.sum())
    if count:
        frame[name] = series.mask(mask)
    return count


def _iso_dates(series: pd.Series) -> pd.Series | None:
    """The column as datetimes if every non-missing value is an ISO-8601 string; otherwise None."""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return None
    values = series.dropna()
    matches = values.map(lambda v: _ISO_DATE.match(v) if isinstance(v, str) else None)
    if values.empty or matches.isna().any():
        return None
    zoned = matches.map(lambda m: m.group(4) is not None)
    if zoned.any() and not zoned.all():
        return None  # some values carry a UTC offset and some don't: ambiguous, leave as text
    try:
        with python_warnings.catch_warnings():
            python_warnings.simplefilter("error")
            parsed = pd.to_datetime(series, format="ISO8601", utc=bool(zoned.all()))
    except (ValueError, TypeError, Warning):
        return None
    return parsed if pd.api.types.is_datetime64_any_dtype(parsed) else None


def _validate_columns(columns: list) -> None:
    if not columns:
        raise ValueError("The file needs a header with named columns.")
    if any(not isinstance(name, str) or not name.strip() for name in columns):
        raise ValueError("Every column needs a non-empty text name.")
    if len(set(columns)) != len(columns):
        raise ValueError("Column names must be unique; the file contains duplicate headers.")
