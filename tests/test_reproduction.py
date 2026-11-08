"""Portable code reproduces the recorded request, including stale UI settings."""
import json
import subprocess
import sys

import pandas as pd
import pytest

from mlcookbook import Dataset, PreprocessingConfig, linear_regression
from mlcookbook.reproduction import export_regression


def test_exported_python_matches_recorded_regression_and_preserves_snapshot(tmp_path, linear_df):
    frame = linear_df.copy()
    frame.index = [4] * len(frame)
    frame["code"] = pd.Series(["001", "002"] * (len(frame) // 2), index=frame.index, dtype="string")
    data = Dataset(frame, overrides={"code": "categorical"})
    request = {"task": "regression", "target": "y", "features": ["x1", "x2", "code"],
               "overrides": {column.name: column.var_type.value for column in data.schema().columns},
               "params": {"penalty": "ridge", "alpha": 3.0}, "cv_folds": 3,
               "preprocessing": {"numeric_impute": "mean", "scaling": "robust"}}
    expected = linear_regression(data, "y", request["features"], penalty="ridge", alpha=3.0, cv_folds=3,
                                 preprocessing=PreprocessingConfig(**request["preprocessing"])).to_dict()
    folder = export_regression(tmp_path / "reproduction", data, request, versions={"python": "recorded"})
    request["params"]["alpha"] = 999
    completed = subprocess.run([sys.executable, str(folder / "run.py")], cwd=tmp_path,
                               capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == expected
    restored = pd.read_parquet(folder / "data.parquet")
    pd.testing.assert_frame_equal(restored, frame)
    assert json.loads((folder / "run.json").read_text(encoding="utf-8"))["request"]["params"]["alpha"] == 3
    with pytest.raises(ValueError, match="never replaced"):
        export_regression(folder, data, request)


def test_reproduction_rejects_modified_snapshot(tmp_path, linear_df):
    data = Dataset(linear_df)
    request = {"task": "regression", "target": "y", "features": ["x1"], "params": {}, "cv_folds": 0}
    folder = export_regression(tmp_path / "reproduction", data, request)
    (folder / "data.parquet").write_bytes(b"altered")
    completed = subprocess.run([sys.executable, str(folder / "run.py")], capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0 and "checksum" in completed.stderr
