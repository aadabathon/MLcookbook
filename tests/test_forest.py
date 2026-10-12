import json
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc, confusion_matrix
from sklearn.model_selection import train_test_split

from mlcookbook import Dataset, RandomForestResult, random_forest
import mlcookbook.classification.forest as forest_module


def test_forest_learns_signal_and_exposes_plain_data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    data = Dataset(pd.DataFrame({"signal": x, "noise": rng.normal(size=300), "y": np.where(x > 0, "yes", "no")}))
    result = random_forest(data, "y", n_estimators=40, max_depth=4)
    assert isinstance(result, RandomForestResult)
    assert result.test_metrics.accuracy > 0.95
    assert result.test_metrics.roc_auc > 0.98
    assert result.feature_importances["signal"] > result.feature_importances["noise"]
    assert sum(result.feature_importances.values()) == pytest.approx(1)
    assert len(result.tree_depths) == len(result.tree_leaf_counts) == 40
    assert max(result.tree_depths) <= 4
    assert all(1 <= count <= 16 for count in result.tree_leaf_counts)
    assert len(result.test_probability) == len(result.test_actual) == len(result.test_predicted) == result.n_test
    assert all(0 <= p <= 1 for p in result.test_probability)
    assert result.roc_fpr[0] == result.roc_tpr[0] == 0
    assert result.roc_fpr[-1] == result.roc_tpr[-1] == 1
    assert auc(result.roc_fpr, result.roc_tpr) == pytest.approx(result.test_metrics.roc_auc)
    assert confusion_matrix(result.test_actual, result.test_predicted, labels=result.classes).tolist() == result.test_metrics.confusion_matrix
    assert any("not causal" in w for w in result.warnings)


def test_defaults_serialization_and_frozen_result(logistic_df):
    result = random_forest(Dataset(logistic_df), "y")
    assert result.n_estimators == len(result.tree_depths) == 100
    assert result.max_depth is None and result.min_samples_leaf == 5 and result.max_features == "sqrt"
    assert result.n_train == 300 and result.n_test == 100
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["method"] == "random_forest"
    assert payload["classes"] == ["no", "yes"]
    assert any(name.startswith("group_") for name in payload["feature_importances"])
    with pytest.raises(FrozenInstanceError):
        result.n_estimators = 2


@pytest.mark.parametrize("labels", [("no", "yes"), (0, 1), (False, True)])
def test_positive_class_switches_probability_and_metric_order(logistic_df, labels):
    logistic_df["y"] = logistic_df["y"].map(dict(zip(["no", "yes"], labels)))
    data = Dataset(logistic_df)
    a = random_forest(data, "y", positive_class=labels[1], n_estimators=20)
    b = random_forest(data, "y", positive_class=labels[0], n_estimators=20)
    assert b.classes == list(reversed(a.classes))
    assert a.test_actual == b.test_actual and a.test_predicted == b.test_predicted
    np.testing.assert_allclose(a.test_probability, 1 - np.array(b.test_probability))
    assert a.test_metrics.confusion_matrix == [row[::-1] for row in b.test_metrics.confusion_matrix[::-1]]
    assert b.test_metrics.positive_class == labels[0]


def test_repeatable_with_same_seed(logistic_df):
    data = Dataset(logistic_df)
    a = random_forest(data, "y", n_estimators=20, random_state=23)
    b = random_forest(data, "y", n_estimators=20, random_state=23)
    assert a.to_dict() == b.to_dict()


@pytest.mark.parametrize("kwargs", [
    {"n_estimators": 0}, {"n_estimators": 2.5}, {"max_depth": 0},
    {"min_samples_leaf": 0}, {"max_features": "wrong"}, {"test_size": 1.0},
])
def test_invalid_settings(logistic_df, kwargs):
    with pytest.raises(ValueError):
        random_forest(Dataset(logistic_df), "y", **kwargs)


@pytest.mark.parametrize("case,message", [
    ("multiclass", "binary"), ("oneclass", "binary"), ("rare", "at least 2"),
    ("positive", "positive_class"), ("target_feature", "also be a feature"),
    ("no_features", "No usable"), ("text_feature", "unsupported type"),
])
def test_shared_validation(logistic_df, case, message):
    kwargs = {"n_estimators": 5}
    if case == "multiclass":
        logistic_df.loc[0, "y"] = "third"
    elif case == "oneclass":
        logistic_df["y"] = "yes"
    elif case == "rare":
        logistic_df["y"] = "yes"
        logistic_df.loc[0, "y"] = "no"
    elif case == "positive":
        kwargs["positive_class"] = "absent"
    elif case == "target_feature":
        kwargs["features"] = ["x1", "y"]
    elif case == "no_features":
        kwargs["features"] = []
    else:
        logistic_df["notes"] = [f"note {i}" for i in range(len(logistic_df))]
        kwargs["features"] = ["notes"]
    with pytest.raises(ValueError, match=message):
        random_forest(Dataset(logistic_df), "y", **kwargs)


def test_missing_targets_features_and_skipped_columns_warn(logistic_df):
    logistic_df.loc[:9, "y"] = None
    logistic_df.loc[20:24, "x1"] = np.nan
    logistic_df["when"] = pd.date_range("2026-01-01", periods=len(logistic_df))
    result = random_forest(Dataset(logistic_df), "y", n_estimators=10)
    assert result.n_train + result.n_test == 390
    assert "when" not in result.features
    assert any("missing target" in w for w in result.warnings)
    assert any("5 missing feature" in w for w in result.warnings)
    assert any("Skipped columns" in w for w in result.warnings)


def test_training_only_preprocessing_and_unseen_test_category(logistic_df, monkeypatch):
    train, test = train_test_split(logistic_df.index, test_size=.25, random_state=0, stratify=logistic_df.y)
    logistic_df["group"] = "training"
    logistic_df.loc[test, "group"] = "held_out_only"
    logistic_df.loc[test, "x1"] = 1_000_000
    logistic_df.loc[train[:3], "x1"] = np.nan
    observed = {}
    original = forest_module.Pipeline.fit

    def fit(pipeline, X, y=None, **kwargs):
        result = original(pipeline, X, y, **kwargs)
        if "forest" in pipeline.named_steps:
            observed["rows"] = set(X.index)
            numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
            observed["numeric_steps"] = list(numeric.named_steps)
            observed["median"] = numeric.named_steps["impute"].statistics_[0]
        return result

    monkeypatch.setattr(forest_module.Pipeline, "fit", fit)
    result = random_forest(Dataset(logistic_df), "y", features=["x1", "group"], n_estimators=10)
    assert observed["rows"] == set(train) and observed["rows"].isdisjoint(test)
    assert "scale" not in observed["numeric_steps"]
    assert observed["median"] == pytest.approx(logistic_df.loc[train, "x1"].median())
    assert "group_held_out_only" not in result.feature_importances
    assert "group_training" in result.feature_importances


def test_single_class_test_split_has_no_roc_and_json_null_auc():
    data = Dataset(pd.DataFrame({"x": range(40), "y": ["a"] * 38 + ["b"] * 2}))
    result = random_forest(data, "y", test_size=.1, n_estimators=10)
    assert len(set(result.test_actual)) == 1
    assert result.roc_fpr == result.roc_tpr == []
    assert result.to_dict()["test_metrics"]["roc_auc"] is None
    assert any("ROC curve" in w for w in result.warnings)
    assert any("imbalance" in w for w in result.warnings)
    json.dumps(result.to_dict(), allow_nan=False)


def test_constant_features_return_zero_importances_with_warning():
    data = Dataset(pd.DataFrame({"x": [1.0] * 40, "y": ["a", "b"] * 20}))
    result = random_forest(data, "y", n_estimators=10)
    assert result.feature_importances == {"x": 0.0}
    assert result.tree_depths == [0] * 10
    assert any("Constant" in w for w in result.warnings)


def test_all_missing_features_rejected():
    data = Dataset(pd.DataFrame({"x": [np.nan] * 40, "y": ["a", "b"] * 20}))
    with pytest.raises(ValueError, match="No observed feature"):
        random_forest(data, "y", n_estimators=10)


def test_overfit_warning_on_noise():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=300) for i in range(5)})
    frame["y"] = rng.choice(["a", "b"], 300)
    result = random_forest(Dataset(frame), "y", n_estimators=50, min_samples_leaf=1)
    assert any("overfitting" in w for w in result.warnings)
