"""Read-only SQLite intake into ordinary, independent dataset snapshots.

Database values are preserved rather than converted from SQL declarations into
statistical roles. Preview is a bounded sample of a query's first returned rows;
an explicit import executes the query anew and requires its complete result to
fit the declared limits. Neither operation changes a database.
"""

from __future__ import annotations

import math
import sqlite3
import sys
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.io import ImportReport, _iso_dates, _validate_columns

MAX_QUERY_LENGTH = 65_536
MAX_COLUMNS = 1_000
MAX_CELL_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SQLiteCatalogResult(Result):
    source: str
    objects: list[dict[str, str]]
    warnings: list[str] = field(default_factory=list)


def _limits(max_rows: int, max_bytes: int, max_seconds: float) -> None:
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 1_000_000:
        raise ValueError("SQLite max_rows must be an integer from 1 to 1,000,000.")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= 512 * 1024 * 1024:
        raise ValueError("SQLite max_bytes must be from 1 byte to 512 MiB.")
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 300:
        raise ValueError("SQLite max_seconds must be greater than 0 and at most 300.")


def _open(path: str | Path) -> sqlite3.Connection:
    selected = Path(path).expanduser().resolve(strict=True)
    if not selected.is_file():
        raise ValueError("Choose an existing SQLite database file.")
    # URI encoding preserves spaces, literal '?' and '#' in database names.
    connection = sqlite3.connect(selected.as_uri() + "?mode=ro", uri=True, timeout=0.1)
    connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_QUERY_LENGTH)
    connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, MAX_COLUMNS)
    # Do not lower SQLITE_LIMIT_LENGTH: built-ins such as printf can return NULL
    # instead of raising when that limit is reduced, changing query semantics.
    # SELECT-only authorization also rejects temporary writes and ATTACH, which
    # opening the primary database read-only would not prevent on its own.
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

    def authorize(action, _first, second, _database, _trigger):
        if action not in allowed or (action == sqlite3.SQLITE_FUNCTION and str(second).lower() == "load_extension"):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(authorize)
    return connection


def _check(cancelled: Callable[[], bool] | None, deadline: float) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("SQLite operation cancelled.")
    if time.monotonic() >= deadline:
        raise TimeoutError("SQLite operation exceeded its time limit. Narrow the query and try again.")


def _progress(connection: sqlite3.Connection, cancelled, deadline: float) -> None:
    def stop():
        return int((cancelled is not None and cancelled()) or time.monotonic() >= deadline)

    connection.set_progress_handler(stop, 1_000)


def _database_error(exc: sqlite3.Error, cancelled, deadline: float) -> None:
    _check(cancelled, deadline)
    raise ValueError(f"Unable to read SQLite: {exc}") from exc


def _row_size(row: tuple) -> int:
    size = sys.getsizeof(row) + 8
    for value in row:
        memory = sys.getsizeof(value)
        if memory > MAX_CELL_BYTES:
            raise ValueError(f"SQLite result contains a cell above the {MAX_CELL_BYTES:,}-byte limit. Narrow the query.")
        encoded = len(value.encode("utf-8")) if isinstance(value, str) else 0
        if encoded > MAX_CELL_BYTES:
            raise ValueError(f"SQLite result contains a cell above the {MAX_CELL_BYTES:,}-byte limit. Narrow the query.")
        size += memory + encoded
    return size


def sqlite_catalog(
    path: str | Path, *, cancelled: Callable[[], bool] | None = None, max_seconds: float = 15,
) -> SQLiteCatalogResult:
    """List user tables and views without querying their contents."""
    _limits(DEFAULT_MAX_ROWS, DEFAULT_MAX_BYTES, max_seconds)
    deadline = time.monotonic() + max_seconds
    _check(cancelled, deadline)
    try:
        with closing(_open(path)) as connection:
            _progress(connection, cancelled, deadline)
            cursor = connection.execute(
                "SELECT name, type FROM sqlite_schema WHERE type IN ('table', 'view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name COLLATE NOCASE, name"
            )
            rows = []
            size = sys.getsizeof(rows)
            while len(rows) <= 1_000:
                row = cursor.fetchone()
                if row is None:
                    break
                size += _row_size(row)
                if size > DEFAULT_MAX_BYTES:
                    raise ValueError("SQLite catalog exceeds the 64-MiB output limit.")
                rows.append(row)
                _check(cancelled, deadline)
            _check(cancelled, deadline)
    except sqlite3.Error as exc:
        _database_error(exc, cancelled, deadline)
    if len(rows) > 1_000:
        raise ValueError("The SQLite database has more than 1,000 tables/views; use a narrower database.")
    return SQLiteCatalogResult(str(Path(path).expanduser().resolve()), [{"name": name, "kind": kind} for name, kind in rows])


def _series(values: list, name: str) -> pd.Series:
    present = [value for value in values if value is not None]
    kinds = {type(value) for value in present}
    if not kinds:
        return pd.Series(values, dtype=object)
    if kinds <= {int}:
        return pd.Series(values, dtype="Int64")
    if kinds <= {int, float}:
        if any(isinstance(value, int) and int(float(value)) != value for value in present):
            raise ValueError(f"Column {name!r} mixes REAL values and integers that cannot be preserved as floats. Cast explicitly in SQL.")
        if any(isinstance(value, float) and not math.isfinite(value) for value in present):
            raise ValueError(f"Column {name!r} contains non-finite numbers; handle them explicitly in SQL.")
        return pd.Series(values, dtype="Float64")
    if kinds <= {str}:
        return pd.Series(values, dtype="string")
    if bytes in kinds:
        raise ValueError(f"Column {name!r} contains BLOB values. Select scalar text/numeric columns or convert the BLOB explicitly in SQL.")
    raise ValueError(f"Column {name!r} mixes incompatible SQLite value types. Cast explicitly in SQL.")


def load_sqlite_dataset(
    path: str | Path,
    query: str,
    *,
    preview: bool = False,
    preview_rows: int = 20,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_seconds: float = 60,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[Dataset, ImportReport, dict[str, Any]]:
    """Execute one read-only query and return dataset, import report and source.

    Full imports fail rather than silently truncate. Preview is explicitly
    incomplete when more rows exist; it does not assert a full-query row count.
    SQL order is preserved, but a query needs ORDER BY to promise a stable order
    across separate executions or changes to its database. Byte/cell limits cap
    retained query output, not SQLite's transient evaluation or peak process
    memory. These limits do not constitute a sandbox.
    """
    _limits(max_rows, max_bytes, max_seconds)
    if not isinstance(preview, bool):
        raise ValueError("SQLite preview must be boolean.")
    if isinstance(preview_rows, bool) or not isinstance(preview_rows, int) or not 1 <= preview_rows <= 1_000:
        raise ValueError("SQLite preview_rows must be from 1 to 1,000.")
    if not isinstance(query, str) or not query.strip() or len(query.encode("utf-8")) > MAX_QUERY_LENGTH:
        raise ValueError("Enter one SELECT query of at most 65,536 UTF-8 bytes.")
    deadline = time.monotonic() + max_seconds
    _check(cancelled, deadline)
    cap = preview_rows if preview else max_rows
    rows: list[tuple] = []
    size = sys.getsizeof(rows)
    try:
        with closing(_open(path)) as connection:
            _progress(connection, cancelled, deadline)
            cursor = connection.execute(query)
            if cursor.description is None:
                raise ValueError("The SQLite query must return a table.")
            names = [column[0] for column in cursor.description]
            _validate_columns(names)
            size += _row_size(tuple(names))
            while len(rows) <= cap:
                _check(cancelled, deadline)
                row = cursor.fetchone()
                if row is None:
                    break
                # Include container/scalar overhead so many empty cells cannot
                # evade the budget. A row can allocate before these checks; do
                # not alter SQLite evaluation limits, which can change results.
                size += _row_size(row)
                if size > max_bytes:
                    raise ValueError(f"SQLite result exceeds the {max_bytes:,}-byte import limit. Narrow the query.")
                rows.append(row)
            _check(cancelled, deadline)
    except sqlite3.Error as exc:
        _database_error(exc, cancelled, deadline)
    has_more = len(rows) > cap
    if has_more and not preview:
        raise ValueError(f"SQLite query returns more than the {max_rows:,}-row import cap. Narrow the query or raise the cap explicitly.")
    rows = rows[:cap]
    if not rows and not preview:
        raise ValueError("The SQLite query returned no data rows.")
    columns = {}
    for index, name in enumerate(names):
        _check(cancelled, deadline)
        columns[name] = _series([row[index] for row in rows], name)
    frame = pd.DataFrame(columns)
    notes = []
    parsed_dates = []
    kept_as_text = []
    for name in names:
        _check(cancelled, deadline)
        values = frame[name].dropna()
        if len(frame) and values.empty:
            notes.append(f"Column {name!r} is entirely NULL; its statistical role cannot be inferred from values.")
        if pd.api.types.is_string_dtype(frame[name]) and values.str.match(r"^[+-]?0\d").any():
            kept_as_text.append(name)
            notes.append(f"Kept {name!r} as text, including leading-zero identifiers.")
        parsed = _iso_dates(frame[name])
        if parsed is not None:
            frame[name] = parsed
            parsed_dates.append(name)
            notes.append(f"Parsed {name!r} as dates (every value is an ISO-8601 date/time).")
    if preview:
        notes.append(f"Preview only: first {len(frame):,} returned rows" + ("; more rows exist." if has_more else "."))
    notes.append("Query result order is retained. Use ORDER BY for reproducible ordering across database queries.")
    selected = str(Path(path).expanduser().resolve())
    source = {"kind": "sqlite", "path": selected, "query": query, "preview": preview, "has_more": has_more,
              "max_rows": max_rows, "retrieved_utc": datetime.now(UTC).isoformat(), "import_notes": list(notes)}
    _check(cancelled, deadline)
    report = ImportReport(Path(path).name, "sqlite", len(frame), len(names), [], kept_as_text, parsed_dates, notes)
    return Dataset(frame), report, source
