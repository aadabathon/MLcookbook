"""Save an explicitly fetched snapshot and its provenance without another request."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4

from mlcookbook._data_http import safe_url
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.fingerprint import dataset_fingerprint

_SOURCES = frozenset({"usgs", "food", "exoplanets", "census", "weather"})
_CREDENTIALS = frozenset({"key", "api_key", "token", "access_token"})


def _redacted(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Download provenance must use string field names.")
        return {key: _redacted(item) for key, item in value.items() if key.lower() not in _CREDENTIALS}
    if isinstance(value, (list, tuple)):
        return [_redacted(item) for item in value]
    if isinstance(value, str) and value.startswith(("https://", "http://")):
        parsed = urlsplit(value)
        if parsed.username or parsed.password:
            raise ValueError("Download provenance must not contain URL credentials.")
        if any(key.lower() in _CREDENTIALS for key, _ in parse_qsl(parsed.query)):
            return safe_url(value)
    return value


def _capture_name(source: str) -> str:
    return f"{source}-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}-{uuid4().hex[:8]}"


def save_gathered_dataset(dataset: Dataset, metadata: dict, directory: str | Path, *, format: str = "csv") -> Path:
    """Save all rows as CSV or typed Parquet, alongside redacted ``source.json``.

    ``metadata`` is the completed preview's ``source`` dictionary, containing
    ``kind``, ``gathering`` and ``dataset_fingerprint``. Each explicit save makes
    a separate capture folder. Its data and sidecar become visible together
    only after both writes succeed, and existing captures are never replaced.
    CSV omits the pandas index and does not encode dtypes; Parquet retains both.
    """
    if format not in {"csv", "parquet"}:
        raise ValueError("Choose CSV or Parquet for the dataset download.")
    if not isinstance(dataset, Dataset):
        raise TypeError("A dataset download requires a Dataset snapshot.")
    if not isinstance(metadata, dict) or metadata.get("kind") != "api" or not isinstance(metadata.get("gathering"), dict):
        raise ValueError("A dataset download requires completed API provenance.")
    source = metadata["gathering"].get("source")
    if not isinstance(source, str) or source not in _SOURCES:
        raise ValueError("The dataset download source is not supported.")
    roles = {column.name: column.var_type.value for column in dataset.schema().columns}
    gathering = metadata["gathering"]
    if (gathering.get("n_rows") != len(dataset.df) or gathering.get("n_columns") != len(dataset.df.columns)
            or gathering.get("roles") != roles or not isinstance(gathering.get("units"), dict)
            or set(gathering["units"]) != set(roles)
            or metadata.get("dataset_fingerprint") != dataset_fingerprint(dataset)):
        raise ValueError("Download provenance does not match the fetched dataset.")
    filename = f"dataset.{format}"
    manifest = _redacted(metadata)
    manifest["export"] = {"file": filename, "format": format, "n_rows": len(dataset.df),
                          "n_columns": len(dataset.df.columns), "index_included": format == "parquet",
                          "dtypes": {str(column): str(dtype) for column, dtype in dataset.df.dtypes.items()}}
    serialized = json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".gathering-", dir=directory))
    try:
        destination = temporary / filename
        if format == "csv":
            dataset.df.to_csv(destination, index=False, encoding="utf-8")
        else:
            dataset.df.to_parquet(destination, index=True, engine="pyarrow")
        (temporary / "source.json").write_text(serialized, encoding="utf-8")
        # Include both a UTC timestamp and a random suffix; even an accidental
        # name collision must leave the existing capture untouched.
        for _ in range(10):
            capture = directory / _capture_name(source)
            if not capture.exists():
                temporary.rename(capture)
                return capture / filename
        raise FileExistsError("Could not allocate a new dataset download folder.")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
