import csv
import hashlib
import json

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from mlcookbook import Dataset, evaluate_wearable, load_wearable_study
from mlcookbook.experiments import load_experiment, save_experiment
from mlcookbook.wearable import INDEX, META, STATISTICS


def make_export(root):
    root.mkdir()
    rng = np.random.default_rng(7)
    axes = ["ax", "ay", "az", "gx", "gy", "gz"]
    features = [f"{axis}_{stat}" for axis in [*axes, "accel_mag", "gyro_mag"] for stat in STATISTICS]
    groups = {f"p{i:02}": "train" if i < 4 else "validation" if i == 4 else "test" for i in range(6)}
    labels = ["jumping", "standing", "walking"]
    counts = {}
    position = 0
    for split in ("train", "validation", "test"):
        values, indexes = [], []
        counts[split] = dict.fromkeys(labels, 0)
        for subject, assigned in groups.items():
            if assigned != split:
                continue
            for label_number, label in enumerate(labels):
                for window in range(4):
                    data = label_number * 8 + rng.normal(0, 0.05, len(features))
                    values.append([*data, label])
                    indexes.append({"source_window": position, "session_id": f"session-{subject}", "subject": subject,
                                    "split": split, "label": label, "run": label_number,
                                    "start_ts_ms": label_number * 10000 + window * 500,
                                    "end_ts_ms": label_number * 10000 + window * 500 + 1980})
                    counts[split][label] += 1
                    position += 1
        with (root / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([*features, "label"])
            writer.writerows(values)
        with (root / f"{split}_index.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=INDEX)
            writer.writeheader()
            writer.writerows(indexes)
    manifest = {"schema_version": 1, "target": "label", "features": features, "source_axes": axes,
                "source_units": ["g"] * 3 + ["deg/s"] * 3, "sample_rate_hz": 50,
                "window_samples": 100, "stride_samples": 25, "labels": labels, "group_by": "subject",
                "group_splits": groups, "class_window_counts": counts, "source_dataset_sha256": "0" * 64,
                "artifacts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}}
    (root / "export.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture
def study(tmp_path):
    return make_export(tmp_path / "study")


def test_verified_import_grouped_cv_predictions_and_json(study):
    dataset, imported = load_wearable_study(study / "export.json")
    result = evaluate_wearable(dataset, final_test=True)
    assert len(imported.features) == 64
    assert set(result.features).isdisjoint(META.values())
    assert set(result.predictions) == {"validation", "test"}
    json.dumps(imported.to_dict(), allow_nan=False)
    json.dumps(result.to_dict(), allow_nan=False)
    assert result.dataset_fingerprint
    for candidate in result.candidates.values():
        assert len(candidate["grouped_cv"]) == 4
        for fold in candidate["grouped_cv"]:
            assert not set(fold["train_groups"]) & set(fold["validation_groups"])
            assert not set(fold["train_source_windows"]) & set(fold["validation_source_windows"])
    assert len(result.predictions["test"]) == 12
    for row in result.predictions["test"]:
        assert row["split"] == "test" and row["subject"] == "p05"
        assert sum(row[f"prob_{label}"] for label in result.labels) == pytest.approx(1)
    assert result.final_test_evaluated


def test_test_labels_cannot_select_model_or_change_validation(study):
    dataset, _ = load_wearable_study(study)
    first = evaluate_wearable(dataset, cv_folds=0)
    frame = dataset.df
    test = frame[META["split"]] == "test"
    frame.loc[test, "label"] = frame.loc[test, "label"].map({"standing": "walking", "walking": "jumping", "jumping": "standing"})
    second = evaluate_wearable(Dataset(frame, overrides={c.name: c.var_type for c in dataset.schema().columns}), cv_folds=0)
    assert first.selected_model == second.selected_model
    assert first.predictions["validation"] == second.predictions["validation"]
    assert "test" not in first.predictions
    assert all("test" not in value for value in first.candidates.values())


def test_preprocessing_never_fits_holdout_rows(study, monkeypatch):
    dataset, _ = load_wearable_study(study)
    frame = dataset.df
    features = [c for c in frame.columns if c not in META.values() and c != "label"]
    frame.loc[frame[META["split"]] != "train", features] += 10000
    configured = Dataset(frame, overrides={c.name: c.var_type for c in dataset.schema().columns})
    observed = []
    original = StandardScaler.fit

    def fit(self, values, *args, **kwargs):
        observed.append(np.asarray(values).copy())
        return original(self, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", fit)
    evaluate_wearable(configured)
    assert len(observed) == 5
    assert all(values.max() < 100 for values in observed)
    assert sorted(len(values) for values in observed) == [36, 36, 36, 36, 48]


@pytest.mark.parametrize("mode", ["checksum", "units", "lineage", "roles", "nonfinite"])
def test_corruption_and_predictor_leakage_rejected(study, mode):
    if mode == "checksum":
        (study / "train.csv").write_text("changed", encoding="utf-8")
        with pytest.raises(ValueError, match="checksum"):
            load_wearable_study(study)
    elif mode == "units":
        manifest = json.loads((study / "export.json").read_text())
        manifest["source_units"] = ["m/s^2"] * 6
        (study / "export.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="contract"):
            load_wearable_study(study)
    else:
        dataset, _ = load_wearable_study(study)
        if mode == "roles":
            dataset.set_type(META["subject"], "categorical")
        elif mode == "lineage":
            frame = dataset.df
            frame.loc[frame[META["split"]] == "test", META["subject"]] = "p00"
            dataset = Dataset(frame, overrides={c.name: c.var_type for c in dataset.schema().columns})
        else:
            frame = dataset.df
            frame.loc[0, "ax_mean"] = np.inf
            dataset = Dataset(frame, overrides={c.name: c.var_type for c in dataset.schema().columns})
        with pytest.raises(ValueError):
            evaluate_wearable(dataset)


def test_saved_snapshot_reruns_without_original_files(study, tmp_path):
    dataset, imported = load_wearable_study(study)
    result = evaluate_wearable(dataset, cv_folds=0)
    saved = tmp_path / "study.mlcb"
    save_experiment(saved, dataset, recipe="wearable-study", name="pilot",
                    state={"report": {"wearable_evaluation": result.to_dict()}},
                    source={"kind": "wearable_study", "wearable_import": imported.to_dict()})
    for file in study.iterdir():
        file.unlink()
    loaded = load_experiment(saved)
    restored = evaluate_wearable(loaded.dataset, cv_folds=0)
    assert restored.predictions == result.predictions
    assert restored.dataset_fingerprint == result.dataset_fingerprint


def test_missing_group_cv_warns_without_row_fallback(study):
    dataset, _ = load_wearable_study(study)
    frame = dataset.df
    frame.loc[frame[META["split"]] == "train", META["subject"]] = "one-person"
    configured = Dataset(frame, overrides={c.name: c.var_type for c in dataset.schema().columns})
    result = evaluate_wearable(configured)
    assert all(not value["grouped_cv"] for value in result.candidates.values())
    assert any("one training group" in warning for warning in result.warnings)
