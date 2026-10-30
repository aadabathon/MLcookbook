"""Dataset downloads retain complete values and provenance as one capture."""

import csv
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from mlcookbook import gathering_io
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.fingerprint import dataset_fingerprint


@pytest.fixture
def snapshot():
    frame = pd.DataFrame({
        "code": pd.array([f"{index:05d}" for index in range(45)], dtype="string"),
        "value": pd.array([None if index % 7 == 0 else index / 10 for index in range(45)], dtype="Float64"),
        "group": pd.array(["Café, daily" if index % 2 else None for index in range(45)], dtype="string"),
        "date": pd.date_range("2024-01-01", periods=45, tz="UTC"),
    }, index=[index // 2 for index in range(45)])
    roles = {"code": "unsupported", "value": "numeric", "group": "categorical", "date": "datetime"}
    dataset = Dataset(frame, overrides=roles)
    metadata = {"kind": "api", "dataset_fingerprint": dataset_fingerprint(dataset), "gathering": {
        "source": "exoplanets", "title": "NASA exoplanets", "query": {"limit": 45, "columns": list(frame.columns)},
        "source_url": "https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html",
        "retrieved_at": "2026-10-29T12:34:56+00:00", "n_rows": 45, "n_columns": 4,
        "roles": roles, "units": {"code": "", "value": "Earth masses", "group": "", "date": "UTC"},
        "request_urls": ["https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=json"],
        "warnings": ["Composite values may use inferred measurements."], "truncated": True,
    }}
    return dataset, metadata


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_download_saves_every_row_and_provenance_without_mutating_snapshot(tmp_path, snapshot, format):
    dataset, metadata = snapshot
    frame, original_metadata = dataset.df.copy(deep=True), deepcopy(metadata)
    root = tmp_path / "data"
    output = gathering_io.save_gathered_dataset(dataset, metadata, root, format=format)
    assert output.parent.parent == root
    assert output.parent.name.startswith("exoplanets-")
    assert {item.name for item in output.parent.iterdir()} == {f"dataset.{format}", "source.json"}
    manifest = json.loads((output.parent / "source.json").read_text(encoding="utf-8"))
    assert {key: manifest[key] for key in original_metadata} == original_metadata
    assert manifest["export"] == {"file": f"dataset.{format}", "format": format, "n_rows": 45,
                                  "n_columns": 4, "index_included": format == "parquet",
                                  "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()}}
    assert json.loads(json.dumps(manifest)) == manifest
    if format == "parquet":
        pd.testing.assert_frame_equal(pd.read_parquet(output), frame)
    else:
        with output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 45 and rows[0]["code"] == "00000" and rows[-1]["code"] == "00044"
        assert rows[0]["value"] == "" and rows[1]["group"] == "Café, daily"
        assert list(rows[0]) == list(frame.columns)  # No synthetic CSV index column.
    pd.testing.assert_frame_equal(dataset.df, frame)
    assert metadata == original_metadata


def test_repeated_downloads_are_independent_and_do_not_overwrite_existing_capture(tmp_path, snapshot, monkeypatch):
    dataset, metadata = snapshot
    first = gathering_io.save_gathered_dataset(dataset, metadata, tmp_path)
    original = first.read_bytes()
    names = iter([first.parent.name, "exoplanets-second-capture"])
    monkeypatch.setattr(gathering_io, "_capture_name", lambda _: next(names))
    second = gathering_io.save_gathered_dataset(dataset, metadata, tmp_path)
    assert second.parent != first.parent and first.read_bytes() == second.read_bytes() == original
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("failure", ["data", "metadata", "rename", "collision"])
def test_failed_download_removes_partial_capture_and_retains_existing_files(tmp_path, snapshot, monkeypatch, failure):
    dataset, metadata = snapshot
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "kept.txt").write_text("kept", encoding="utf-8")
    if failure == "data":
        def fail_data(frame, path, **kwargs):
            Path(path).write_text("partial", encoding="utf-8")
            raise OSError("disk full")
        monkeypatch.setattr(pd.DataFrame, "to_csv", fail_data)
    elif failure == "metadata":
        original = Path.write_text

        def fail_metadata(path, *args, **kwargs):
            if path.name == "source.json":
                raise OSError("disk full")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "write_text", fail_metadata)
    elif failure == "rename":
        def fail_rename(*args):
            raise OSError("rename refused")
        monkeypatch.setattr(Path, "rename", fail_rename)
    else:
        monkeypatch.setattr(gathering_io, "_capture_name", lambda _: "existing")
    with pytest.raises(OSError):
        gathering_io.save_gathered_dataset(dataset, metadata, tmp_path)
    assert list(tmp_path.iterdir()) == [existing]
    assert (existing / "kept.txt").read_text(encoding="utf-8") == "kept"


def test_credentials_are_removed_from_nested_fields_and_request_urls(tmp_path, snapshot):
    dataset, metadata = snapshot
    metadata["api_key"] = "top-secret"
    metadata["gathering"]["query"]["API_KEY"] = "query-secret"
    metadata["gathering"]["request_urls"] = ["https://api.census.gov/data?get=NAME&key=url-secret&token=token-secret"]
    metadata["gathering"]["nested"] = [{"access_token": "nested-secret", "kept": "value"}]
    output = gathering_io.save_gathered_dataset(dataset, metadata, tmp_path)
    raw = (output.parent / "source.json").read_text(encoding="utf-8")
    assert "secret" not in raw
    saved = json.loads(raw)
    assert saved["gathering"]["request_urls"] == ["https://api.census.gov/data?get=NAME"]
    assert saved["gathering"]["nested"] == [{"kept": "value"}]
    assert metadata["api_key"] == "top-secret"  # Redaction does not alter the caller's state.


@pytest.mark.parametrize("change", ["format", "kind", "source", "roles", "units", "rows", "columns", "fingerprint", "nonfinite", "url_auth"])
def test_invalid_download_requests_create_no_directory(tmp_path, snapshot, change):
    dataset, metadata = snapshot
    format = "csv"
    if change == "format":
        format = "pickle"
    elif change == "kind":
        metadata["kind"] = "file"
    elif change == "source":
        metadata["gathering"]["source"] = "../../outside"
    elif change == "roles":
        metadata["gathering"]["roles"]["code"] = "numeric"
    elif change == "units":
        metadata["gathering"]["units"] = None
    elif change == "rows":
        metadata["gathering"]["n_rows"] = 20
    elif change == "columns":
        metadata["gathering"]["n_columns"] = 1
    elif change == "fingerprint":
        metadata["dataset_fingerprint"] = "outdated"
    elif change == "nonfinite":
        metadata["gathering"]["query"]["bad"] = float("nan")
    else:
        metadata["gathering"]["request_urls"] = ["https://user:password@example.com/data"]
    directory = tmp_path / "data"
    with pytest.raises(ValueError):
        gathering_io.save_gathered_dataset(dataset, metadata, directory, format=format)
    assert not directory.exists()
