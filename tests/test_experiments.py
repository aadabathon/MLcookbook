"""Self-contained, versioned experiment persistence and recovery."""
import json
import zipfile

import pandas as pd
import pytest

from mlcookbook import Dataset
from mlcookbook.experiments import load_experiment, runtime_versions, save_experiment


def rewrite(path, change_manifest=None, snapshot=None):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("experiment.json"))
        original = archive.read("data.parquet")
    if change_manifest:
        change_manifest(manifest)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("experiment.json", json.dumps(manifest))
        archive.writestr("data.parquet", original if snapshot is None else snapshot)


def test_round_trip_keeps_typed_snapshot_roles_and_state(tmp_path):
    frame = pd.DataFrame({"code": ["001", "002", None], "x": pd.Series([1, None, 3], dtype="Int64"),
                          "flag": pd.Series([True, None, False], dtype="boolean"),
                          "category": pd.Categorical(["a", "b", None]),
                          "when": pd.date_range("2024-01-01", periods=3, tz="UTC")})
    frame.index = [10, 10, 21]
    data = Dataset(frame, overrides={"code": "text"})
    state = {"report": {"summary": "Saved result"}, "request": {"params": {"random_state": 17}}}
    source = {"path": str(tmp_path / "deleted.csv"), "delimiter": ";", "missing_markers": ["?"]}
    path = tmp_path / "analysis.mlcb"
    fingerprint = save_experiment(path, data, recipe="model-comparison", name="Study", state=state, source=source)
    loaded = load_experiment(path)
    pd.testing.assert_frame_equal(loaded.dataset.df, frame)
    assert loaded.dataset.schema().var_type("code").value == "text"
    assert loaded.fingerprint == fingerprint and len(fingerprint) == 64
    assert loaded.name == "Study" and loaded.recipe == "model-comparison"
    assert loaded.state == state and loaded.source == source
    assert loaded.versions == runtime_versions() and not loaded.warnings


def test_ollama_unrun_engine_change_keeps_pytorch_report_versions(tmp_path):
    """Switching engine without rerunning must not invent version mismatches."""
    state = {"params": {"backend": "ollama"},
             "report": {"provenance": {"versions": runtime_versions(include_llm=True)}}}
    path = tmp_path / "changed-engine.mlcb"
    save_experiment(path, Dataset(pd.DataFrame({"x": [1, 2]})), recipe="llm", name="Changed engine", state=state)
    loaded = load_experiment(path)
    assert not loaded.warnings
    assert loaded.state == state


@pytest.mark.parametrize("change,match", [
    (lambda m: m.update(version=999), "unsupported format"),
    (lambda m: m.update(format="other"), "not an MLcookbook"),
    (lambda m: m.update(state=[]), "state must be"),
    (lambda m: m.update(roles={}), "column roles"),
    (lambda m: m.update(snapshot_sha256="bad"), "fingerprint"),
])
def test_invalid_experiments_are_rejected(tmp_path, linear_df, change, match):
    path = tmp_path / "analysis.mlcb"
    save_experiment(path, Dataset(linear_df), recipe="linear-regression", name="Study", state={})
    rewrite(path, change)
    with pytest.raises(ValueError, match=match):
        load_experiment(path)


def test_corrupt_snapshot_and_invalid_archives(tmp_path, linear_df):
    path = tmp_path / "analysis.mlcb"
    path.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="Could not open"):
        load_experiment(path)
    save_experiment(path, Dataset(linear_df), recipe="linear-regression", name="Study", state={})
    rewrite(path, snapshot=b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        load_experiment(path)


def test_version_change_warns_without_discarding_result(tmp_path, linear_df):
    path = tmp_path / "analysis.mlcb"
    state = {"report": {"summary": "A completed run", "provenance": {"versions": {"pandas": "0.0"}}}}
    save_experiment(path, Dataset(linear_df), recipe="linear-regression", name="Study", state=state)
    rewrite(path, lambda m: m["versions"].update({"scikit-learn": "0.0"}))
    loaded = load_experiment(path)
    assert loaded.state == state
    assert any("scikit-learn 0.0" in message for message in loaded.warnings)
    assert any("pandas 0.0" in message for message in loaded.warnings)


def test_failed_save_preserves_existing_experiment(tmp_path, linear_df, monkeypatch):
    import mlcookbook.experiments as experiments

    path = tmp_path / "analysis.mlcb"
    data = Dataset(linear_df)
    save_experiment(path, data, recipe="linear-regression", name="Original", state={})
    original = path.read_bytes()

    def fail(*args):
        raise OSError("Disk failure")

    monkeypatch.setattr(experiments.os, "replace", fail)
    with pytest.raises(OSError, match="Disk failure"):
        save_experiment(path, data, recipe="linear-regression", name="Replacement", state={})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
