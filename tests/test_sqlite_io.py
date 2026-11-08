"""SQLite sources preserve values, enforce read-only access and import snapshots."""

import json
import sqlite3
from contextlib import closing

import pandas as pd
import pytest

from mlcookbook.io import load_dataset
from mlcookbook.sqlite_io import load_sqlite_dataset, sqlite_catalog


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "a #% database.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute('CREATE TABLE "measurements" (id TEXT, value REAL, count INTEGER, measured TEXT)')
        connection.executemany("INSERT INTO measurements VALUES (?, ?, ?, ?)", [
            (f"{number:05d}", float(number) if number != 4 else None, number if number != 5 else None, f"2026-01-{number + 1:02d}")
            for number in range(30)
        ])
        connection.execute("CREATE VIEW positive AS SELECT * FROM measurements WHERE value > 10")
        connection.commit()
    return path


def test_catalog_and_query_preserve_identifiers_nulls_dates_and_provenance(database):
    catalog = sqlite_catalog(database)
    assert catalog.objects == [{"name": "measurements", "kind": "table"}, {"name": "positive", "kind": "view"}]
    json.dumps(catalog.to_dict(), allow_nan=False)
    dataset, report, source = load_sqlite_dataset(database, "SELECT * FROM measurements ORDER BY id")
    assert dataset.df.id.iloc[0] == "00000"
    assert dataset.df["count"].iloc[29] == 29
    assert pd.isna(dataset.df["count"].iloc[5])
    assert pd.isna(dataset.df.value.iloc[4])
    assert dataset.schema().var_type("id") == "text"
    assert dataset.schema().var_type("measured") == "datetime"
    assert report.parsed_dates == ["measured"] and report.kept_as_text == ["id"]
    assert report.file_format == "sqlite"
    assert source["query"].endswith("ORDER BY id") and not source["preview"] and not source["has_more"]
    assert source["import_notes"] == report.notes
    assert any("leading-zero" in note for note in source["import_notes"])
    assert any("dates" in note for note in source["import_notes"])
    json.dumps(source, allow_nan=False)
    json.dumps(report.to_dict(), allow_nan=False)


def test_preview_does_not_truncate_explicit_full_import(database):
    preview, report, source = load_sqlite_dataset(database, "SELECT * FROM measurements ORDER BY id", preview=True)
    assert len(preview.df) == report.n_rows == 20
    assert source["preview"] and source["has_more"]
    full, _, source = load_sqlite_dataset(database, "SELECT * FROM measurements ORDER BY id")
    assert len(full.df) == 30 and not source["has_more"]
    assert len(full.df) != len(preview.df)
    with pytest.raises(ValueError, match="more than the 10-row import cap"):
        load_sqlite_dataset(database, "SELECT * FROM measurements", max_rows=10)
    explicit, _, _ = load_sqlite_dataset(database, "SELECT * FROM measurements ORDER BY id LIMIT 10", max_rows=10)
    assert len(explicit.df) == 10


@pytest.mark.parametrize("query", [
    "DELETE FROM measurements",
    "UPDATE measurements SET value=9",
    "CREATE TEMP TABLE forbidden(x)",
    "ATTACH DATABASE ':memory:' AS attached",
    "PRAGMA user_version=3",
    "SELECT load_extension('anything')",
    "SELECT * FROM measurements; DELETE FROM measurements",
])
def test_read_only_rejects_database_and_temporary_mutations(database, query):
    with pytest.raises(ValueError, match="Unable to read SQLite|must return"):
        load_sqlite_dataset(database, query)
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT count(*) FROM measurements").fetchone()[0] == 30
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0


def test_query_validation_duplicate_columns_mixed_values_and_blobs(database):
    for query, message in [
        ("SELECT id, id FROM measurements", "duplicate"),
        ("SELECT '' AS ''", "non-empty"),
        ("SELECT x'1234' AS blob", "BLOB"),
        ("SELECT 1 AS value UNION ALL SELECT 'text'", "incompatible"),
        ("SELECT 9007199254740993 AS value UNION ALL SELECT 1.5", "cannot be preserved"),
        ("SELECT 1e999 AS value", "non-finite"),
    ]:
        with pytest.raises(ValueError, match=message):
            load_sqlite_dataset(database, query)
    with pytest.raises(ValueError, match="no data rows"):
        load_sqlite_dataset(database, "SELECT * FROM measurements WHERE 0")
    preview, _, _ = load_sqlite_dataset(database, "SELECT * FROM measurements WHERE 0", preview=True)
    assert preview.df.empty


def test_large_nullable_integers_are_not_silently_converted_to_float(database):
    dataset, _, _ = load_sqlite_dataset(database, "SELECT 9223372036854775807 AS n UNION ALL SELECT NULL")
    assert dataset.df.n.dtype == pd.Int64Dtype()
    assert int(dataset.df.n.iloc[0]) == 9223372036854775807
    assert pd.isna(dataset.df.n.iloc[1])


def test_payload_and_time_are_bounded_and_query_can_be_cancelled(database):
    with pytest.raises(ValueError, match="limit|too big"):
        load_sqlite_dataset(database, "SELECT printf('%010000d', 1) AS large", max_bytes=1_000)
    expensive = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n"
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks > 10

    with pytest.raises(InterruptedError, match="cancelled"):
        load_sqlite_dataset(database, expensive, cancelled=cancelled)
    with pytest.raises(TimeoutError, match="time limit"):
        load_sqlite_dataset(database, expensive, max_seconds=0.01)
    with pytest.raises(InterruptedError):
        sqlite_catalog(database, cancelled=lambda: True)


def test_empty_cells_still_count_toward_retained_memory_limit(database):
    query = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x < 100) SELECT '' AS empty FROM n"
    with pytest.raises(ValueError, match="byte import limit"):
        load_sqlite_dataset(database, query, max_bytes=1_000)


def test_csv_parquet_and_sqlite_have_equivalent_values_and_statistical_roles(database, tmp_path):
    sql, _, _ = load_sqlite_dataset(database, "SELECT * FROM measurements ORDER BY id")
    csv = tmp_path / "same.csv"
    parquet = tmp_path / "same.parquet"
    sql.df.to_csv(csv, index=False)
    sql.df.to_parquet(parquet, index=False)
    for loaded in (load_dataset(csv), load_dataset(parquet)):
        assert [column.var_type for column in loaded.schema().columns] == [column.var_type for column in sql.schema().columns]
        # CSV has no stored nullable dtype contract; compare common semantic
        # dtypes, including the same missing sentinel, rather than NaN vs pd.NA.
        normalized = loaded.df.copy()
        for name in sql.df.columns:
            normalized[name] = normalized[name].astype(sql.df[name].dtype)
        pd.testing.assert_frame_equal(normalized, sql.df)
    pd.testing.assert_frame_equal(load_dataset(parquet).df, sql.df)


def test_saved_query_snapshot_reopens_without_database(database, tmp_path):
    # Persistence is already an ordinary typed-Parquet snapshot. A source dict
    # can be retained verbatim without executing its query when reading bytes.
    dataset, report, source = load_sqlite_dataset(database, "SELECT * FROM measurements ORDER BY id")
    capture = tmp_path / "snapshot.parquet"
    dataset.df.to_parquet(capture)
    database.unlink()
    restored = load_dataset(capture)
    pd.testing.assert_frame_equal(restored.df, dataset.df)
    assert json.loads(json.dumps(source))["query"] == source["query"]
    assert json.loads(json.dumps(source))["import_notes"] == report.notes
    assert report.n_rows == len(restored.df)


@pytest.mark.parametrize("options", [
    {"max_rows": 0}, {"max_rows": True}, {"max_rows": 1_000_001},
    {"max_bytes": 0}, {"max_seconds": 0}, {"max_seconds": float("nan")},
    {"preview_rows": 0}, {"preview": "yes"},
])
def test_limits_are_validated_before_database_work(database, options):
    with pytest.raises(ValueError):
        load_sqlite_dataset(database, "SELECT * FROM measurements", **options)


def test_nonexistent_database_is_never_created(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        sqlite_catalog(path)
    assert not path.exists()
