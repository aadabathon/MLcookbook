"""Portable, readable reproduction of a completed linear-regression request."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import to_jsonable


def regression_source(request: dict, *, roles: dict[str, str], snapshot_sha256: str, versions: dict | None = None) -> str:
    """Return executable Python for the recorded request, never current UI edits."""
    if request.get("task") != "regression" or not request.get("target") or not request.get("features"):
        raise ValueError("A completed regression request with target and predictors is required.")
    if request.get("params", {}).get("penalty", "none") not in ("none", "ridge", "lasso"):
        raise ValueError("Unknown linear-regression penalty.")
    effective_roles = {**roles, **request.get("overrides", {})}
    params = request.get("params", {})
    settings = {"target": request["target"], "features": request["features"],
                "test_size": 0.25, "random_state": 0,
                "penalty": params.get("penalty", "none"), "alpha": params.get("alpha", 1.0),
                "cv_folds": request.get("cv_folds", 5)}
    # repr preserves Python literals safely, including unusual user column names.
    lines = ["# Reproduces the completed ML Cookbook linear-regression run.",
             "# Recorded versions: " + json.dumps(versions or {}, sort_keys=True),
             "from pathlib import Path", "import hashlib", "import json", "import pandas as pd",
             "from mlcookbook import Dataset, PreprocessingConfig, linear_regression", "",
             "snapshot = Path(__file__).resolve().with_name('data.parquet')",
             f"expected_sha256 = {snapshot_sha256!r}",
             "if hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected_sha256:",
             "    raise ValueError('Dataset snapshot checksum does not match.')",
             f"data = Dataset(pd.read_parquet(snapshot), overrides={effective_roles!r})",
             f"settings = {settings!r}"]
    if request.get("preprocessing") is not None:
        lines.append(f"settings['preprocessing'] = PreprocessingConfig(**{request['preprocessing']!r})")
    lines.extend(["result = linear_regression(data, **settings)",
                  "print(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False, indent=2))", ""])
    return "\n".join(lines)


def export_regression(path: str | Path, dataset: Dataset, request: dict, *, versions: dict | None = None) -> Path:
    """Atomically publish a new directory with run.py, typed rows and provenance."""
    path = Path(path)
    if path.exists():
        raise ValueError("Choose a new reproduction folder; existing work is never replaced.")
    snapshot = dataset.df.to_parquet(index=True, engine="pyarrow")
    fingerprint = hashlib.sha256(snapshot).hexdigest()
    roles = {column.name: column.var_type.value for column in dataset.schema().columns}
    source = regression_source(request, roles=roles, snapshot_sha256=fingerprint, versions=versions)
    temporary = Path(tempfile.mkdtemp(prefix=".reproduction-", dir=path.parent))
    try:
        (temporary / "data.parquet").write_bytes(snapshot)
        (temporary / "run.py").write_text(source, encoding="utf-8")
        (temporary / "run.json").write_text(json.dumps(to_jsonable({
            "format": "mlcookbook-python-reproduction", "version": 1, "recipe": "linear-regression",
            "request": request, "roles": roles, "snapshot_sha256": fingerprint, "versions": versions or {},
        }), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return path
