"""Verified movement-study imports and fixed-group classical baselines.

Human labels and immutable split/clock lineage travel in the Dataset snapshot.
Metadata roles are unsupported predictors. No fitted estimators or CNN weights
are serialized, and the final test holdout is used only when requested.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.evaluation.metrics import classification_metrics
from mlcookbook.experiments import runtime_versions
from mlcookbook.preprocessing.builders import build_preprocessor
from mlcookbook.preprocessing.config import PreprocessingConfig

SPLITS = ("train", "validation", "test")
INDEX = ("source_window", "session_id", "subject", "split", "label", "run", "start_ts_ms", "end_ts_ms")
META = {name: "__wearable_" + name for name in (*INDEX[:4], *INDEX[5:], "group_by", "export_sha256", "synthetic")}
STATISTICS = ("mean", "std", "min", "max", "rms", "median", "iqr", "mean_abs_delta")


@dataclass(frozen=True)
class WearableImportReport(Result):
    source_export_sha256: str
    source_dataset_sha256: str
    features: list[str]
    labels: list[str]
    group_by: str
    group_splits: dict[str, str]
    class_window_counts: dict
    artifact_hashes: dict[str, str]
    warnings: list[str]


@dataclass(frozen=True)
class WearableEvaluationResult(Result):
    schema_version: int
    created_utc: str
    selected_model: str
    selection: str
    features: list[str]
    labels: list[str]
    seed: int
    group_by: str
    group_splits: dict[str, str]
    candidates: dict
    predictions: dict
    final_test_evaluated: bool
    source_export_sha256: str
    dataset_fingerprint: str
    versions: dict[str, str]
    warnings: list[str]
    limitations: list[str]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv(data: bytes, name: str) -> pd.DataFrame:
    text = data.decode("utf-8-sig")
    header = next(csv.reader(io.StringIO(text)), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f"Missing or duplicate column headers: {name}")
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def _contract(manifest: dict) -> list[str]:
    if (manifest.get("schema_version") != 1 or manifest.get("target") != "label"
            or manifest.get("group_by") not in ("subject", "session_id")):
        raise ValueError("Unsupported movement-study export or grouping.")
    axes = manifest.get("source_axes")
    if axes not in (["ax", "ay", "az"], ["ax", "ay", "az", "gx", "gy", "gz"]):
        raise ValueError("Movement study must retain three or six IMU axes.")
    signals = axes + ["accel_mag"] + (["gyro_mag"] if len(axes) == 6 else [])
    features = [f"{axis}_{stat}" for axis in signals for stat in STATISTICS]
    units = ["g"] * 3 + (["deg/s"] * 3 if len(axes) == 6 else [])
    labels = manifest.get("labels")
    if (manifest.get("features") != features or manifest.get("source_units") != units
            or manifest.get("sample_rate_hz") != 50 or not isinstance(labels, list)
            or not 2 <= len(labels) <= 50 or not all(isinstance(v, str) for v in labels) or labels != sorted(set(labels))
            or any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,15}", v)
                   or v in ("unknown", "unlabeled") for v in labels)):
        raise ValueError("Movement-study predictor, units, sample rate or label contract is invalid.")
    return features


def load_wearable_study(path: str | Path) -> tuple[Dataset, WearableImportReport]:
    """Load a verified wearable export.json and its fixed split/index CSVs.

    File bytes are read once, hashed, then parsed from those same bytes. The
    returned snapshot includes provenance needed for offline save/reopen/rerun.
    """
    path = Path(path)
    manifest_path = path / "export.json" if path.is_dir() else path
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("Movement-study manifest must be an object.")
    features = _contract(manifest)
    if type(manifest.get("synthetic", False)) is not bool:
        raise ValueError("Invalid synthetic-data marker.")
    frames, hashes = [], {}
    for split in SPLITS:
        tables = []
        for name in (f"{split}.csv", f"{split}_index.csv"):
            payload = (manifest_path.parent / name).read_bytes()
            hashes[name] = _sha(payload)
            if hashes[name] != manifest.get("artifacts_sha256", {}).get(name):
                raise ValueError(f"Movement-study checksum mismatch: {name}")
            tables.append(_csv(payload, name))
        table, index = tables
        if (table.columns.tolist() != features + ["label"] or set(index.columns) != set(INDEX)
                or len(table) == 0 or len(table) != len(index)
                or not table["label"].equals(index["label"]) or not (index["split"] == split).all()):
            raise ValueError("Movement-study feature/index shape or label alignment mismatch.")
        for col in features:
            table[col] = pd.to_numeric(table[col], errors="raise").astype(float)
        for col, target in META.items():
            if col in index:
                table[target] = index[col].to_numpy()
        table[META["group_by"]] = manifest["group_by"]
        table[META["export_sha256"]] = _sha(raw)
        table[META["synthetic"]] = manifest.get("synthetic", False)
        frames.append(table)
    combined = pd.concat(frames, ignore_index=True)
    for name in ("source_window", "run", "start_ts_ms", "end_ts_ms"):
        values = combined[META[name]]
        if not values.str.fullmatch(r"\d+").all():
            raise ValueError("Movement-study clock/window positions must be nonnegative integers.")
        combined[META[name]] = pd.to_numeric(values, errors="raise").astype("int64")
    dataset = Dataset(combined, overrides={**{v: "unsupported" for v in META.values()}, "label": "categorical"})
    _, _, groups, counts = _validate(dataset)
    if groups != manifest.get("group_splits") or counts != manifest.get("class_window_counts"):
        raise ValueError("Movement-study recorded groups/class coverage differs from its tables.")
    report = WearableImportReport(_sha(raw), manifest.get("source_dataset_sha256", ""), features,
                                  manifest["labels"], manifest["group_by"], groups, counts, hashes,
                                  ["Original acquisition sessions remain separate evidence; this snapshot contains window features."])
    return dataset, report


def _validate(dataset: Dataset):
    frame = dataset.df
    if not set(META.values()) | {"label"} <= set(frame.columns):
        raise ValueError("Open a verified movement study; ordinary CSV files have no fixed group lineage.")
    features = [c for c in frame.columns if c not in META.values() and c != "label"]
    expected = [[f"{axis}_{stat}" for axis in axes for stat in STATISTICS]
                for axes in (["ax", "ay", "az", "accel_mag"],
                             ["ax", "ay", "az", "gx", "gy", "gz", "accel_mag", "gyro_mag"])]
    if features not in expected:
        raise ValueError("Invalid movement-study predictor columns.")
    if (any(dataset.schema().var_type(c).value != "numeric" for c in features)
            or dataset.schema().var_type("label").value != "categorical"
            or any(dataset.schema().var_type(c).value != "unsupported" for c in META.values())):
        raise ValueError("Movement-study roles changed: predictors are numeric; lineage cannot be a predictor.")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all() or frame.isna().any().any():
        raise ValueError("Nonfinite or missing movement-study values.")
    if not frame["label"].map(lambda x: isinstance(x, str)).all():
        raise ValueError("Movement-study labels must be strings.")
    if not pd.api.types.is_bool_dtype(frame[META["synthetic"]]) or frame[META["synthetic"]].nunique() != 1:
        raise ValueError("Invalid synthetic-data marker in saved snapshot.")
    labels = sorted(frame["label"].unique().tolist())
    if not 2 <= len(labels) <= 50 or any(v in ("unknown", "unlabeled") for v in labels):
        raise ValueError("Movement study requires 2..50 known classes.")
    if set(frame[META["split"]]) != set(SPLITS):
        raise ValueError("Movement study needs the fixed train, validation and test splits.")
    grouping = frame[META["group_by"]].unique().tolist()
    if len(grouping) != 1 or grouping[0] not in ("subject", "session_id"):
        raise ValueError("Invalid movement-study grouping.")
    fingerprints = frame[META["export_sha256"]].unique().tolist()
    if len(fingerprints) != 1 or not isinstance(fingerprints[0], str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprints[0]):
        raise ValueError("Movement study must refer to one verified export.")
    for name in ("source_window", "run", "start_ts_ms", "end_ts_ms"):
        if not pd.api.types.is_integer_dtype(frame[META[name]]) or (frame[META[name]] < 0).any():
            raise ValueError("Invalid movement-study window/clock positions.")
    if (frame[META["source_window"]].duplicated().any()
            or frame.duplicated([META[n] for n in ("session_id", "run", "start_ts_ms", "end_ts_ms")]).any()
            or (frame[META["end_ts_ms"]] <= frame[META["start_ts_ms"]]).any()):
        raise ValueError("Duplicate or invalid movement-study window lineage.")
    group_splits, sessions = {}, {}
    for row in frame.to_dict("records"):
        group, session, split = row[META[grouping[0]]], row[META["session_id"]], row[META["split"]]
        if any(not isinstance(v, str) or not v or v == "unknown" for v in (group, session)):
            raise ValueError("Movement study requires known grouping/session IDs.")
        if group_splits.setdefault(group, split) != split or sessions.setdefault(session, split) != split:
            raise ValueError("Movement-study group/session leakage across fixed splits.")
    counts = {split: frame.loc[frame[META["split"]] == split, "label"].value_counts().to_dict() for split in SPLITS}
    if any(set(counts[split]) != set(labels) for split in SPLITS):
        raise ValueError("Every fixed split must contain every movement class.")
    return features, labels, group_splits, counts


def check_wearable_study(dataset: Dataset) -> None:
    """Validate a saved snapshot without fitting or reading its original files."""
    _validate(dataset)


def evaluate_wearable(dataset: Dataset, *, cv_folds: int = 5, final_test: bool = False,
                      random_state: int = 42) -> WearableEvaluationResult:
    """Compare movement baselines on fixed validation groups; test is opt-in."""
    if type(cv_folds) is not int or cv_folds not in (0, 2, 3, 5, 10):
        raise ValueError("cv_folds must be 0, 2, 3, 5 or 10.")
    if type(final_test) is not bool or type(random_state) is not int or not 0 <= random_state <= 2147483647:
        raise ValueError("Invalid final-test choice or random seed.")
    features, labels, group_splits, _ = _validate(dataset)
    frame = dataset.df
    group_by = frame[META["group_by"]].iloc[0]
    splits = {split: frame.loc[frame[META["split"]] == split] for split in SPLITS}
    train = splits["train"]
    groups = train[META[group_by]].to_numpy()
    warnings = []
    if frame[META["synthetic"]].iloc[0]:
        warnings.append("Synthetic fixture data: scores only check the software integration and do not measure movement recognition.")
    if group_by == "session_id":
        warnings.append("Session-disjoint evaluation does not establish unseen-person generalization.")
    if len(set(groups)) < 5:
        warnings.append("Few training groups: treat this as a pilot baseline, not established generalization.")
    for issue in dataset.describe().issues:
        if issue.column in features and issue.kind in ("constant", "all_missing"):
            warnings.append(issue.message)
    folds = []
    if cv_folds and len(set(groups)) >= 2:
        for a, b in GroupKFold(n_splits=min(cv_folds, len(set(groups)))).split(train[features], train.label, groups):
            if set(train.iloc[a].label) != set(labels) or set(train.iloc[b].label) != set(labels):
                warnings.append("Grouped CV unavailable: a fold lacks class coverage; no row-wise fallback.")
                folds = []
                break
            folds.append((a, b))
    elif cv_folds:
        warnings.append("Grouped CV unavailable with one training group; fixed validation groups remain separate.")

    def build(name):
        config = PreprocessingConfig(scaling="standard" if name == "logistic_regression" else "none")
        model = (LogisticRegression(max_iter=2000, random_state=random_state) if name == "logistic_regression" else
                 RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, n_jobs=1, random_state=random_state))
        return Pipeline([("preprocess", build_preprocessor(dataset, features, config=config)), ("classifier", model)])

    def score(model, table):
        predicted = model.predict(table[features])
        probability = model.predict_proba(table[features])[:, [list(model.classes_).index(v) for v in labels]]
        metrics = classification_metrics(table.label, predicted, probability[:, 1] if len(labels) == 2 else probability, labels).to_dict()
        metrics["macro_f1"] = float(f1_score(table.label, predicted, labels=labels, average="macro", zero_division=0))
        return metrics, predicted, probability

    models, candidates = {}, {}
    for name in ("logistic_regression", "random_forest"):
        cv = []
        for a, b in folds:
            model = build(name)
            model.fit(train.iloc[a][features], train.iloc[a].label)
            metrics, _, _ = score(model, train.iloc[b])
            cv.append({"train_source_windows": train.iloc[a][META["source_window"]].tolist(),
                       "validation_source_windows": train.iloc[b][META["source_window"]].tolist(),
                       "train_groups": sorted(set(groups[a])), "validation_groups": sorted(set(groups[b])), "metrics": metrics})
        model = build(name)
        model.fit(train[features], train.label)
        metrics, _, _ = score(model, splits["validation"])
        candidates[name] = {"validation": metrics, "grouped_cv": cv,
                            "estimator_parameters": model.named_steps["classifier"].get_params(),
                            "scaling": "standard" if name == "logistic_regression" else "none"}
        models[name] = model
    selected = max(candidates, key=lambda name: candidates[name]["validation"]["macro_f1"])
    predictions = {}
    for split in (("validation", "test") if final_test else ("validation",)):
        metrics, predicted, probability = score(models[selected], splits[split])
        if split == "test":
            candidates[selected]["test"] = metrics
        records = []
        for position, row in enumerate(splits[split].to_dict("records")):
            records.append({**{key: row[META[key]] for key in INDEX if key != "label"}, "label": row["label"],
                            "predicted_label": predicted[position], "confidence": float(probability[position].max()),
                            **{f"prob_{label}": float(probability[position, j]) for j, label in enumerate(labels)}})
        predictions[split] = records
    snapshot_hash = _sha(frame.to_parquet(index=True, engine="pyarrow"))
    return WearableEvaluationResult(1, datetime.now(UTC).isoformat(), selected,
        "macro F1 on fixed validation groups; stable tie-break; test excluded", features, labels, random_state,
        group_by, group_splits, candidates, predictions, final_test, frame[META["export_sha256"]].iloc[0],
        snapshot_hash, runtime_versions(), warnings,
        ["Classical window baseline, not the deployed temporal CNN.",
         "Window metrics are not fall-event recall or false alerts per monitored hour."])
