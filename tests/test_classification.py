import json

import numpy as np
import pytest

from mlcookbook import Dataset, logistic_regression
from mlcookbook.evaluation.metrics import classification_metrics


def test_recovers_logistic_relationship(logistic_df):
    result = logistic_regression(Dataset(logistic_df), target="y", features=["x1", "x2"])
    assert result.classes == ["no", "yes"]
    assert result.positive_class == "yes"
    assert result.test_metrics.roc_auc > 0.8
    assert result.coefficients["x1"] > 0 > result.coefficients["x2"]
    assert result.odds_ratios["x1"] == pytest.approx(np.exp(result.coefficients["x1"]))
    assert result.n_train + result.n_test == len(logistic_df)


def test_split_is_stratified(logistic_df):
    result = logistic_regression(Dataset(logistic_df), target="y")
    share_all = (logistic_df["y"] == "yes").mean()
    m = result.test_metrics
    actual_yes = sum(m.confusion_matrix[1])
    assert actual_yes / m.n_obs == pytest.approx(share_all, abs=0.02)


def test_positive_class_flips_coefficients(logistic_df):
    data = Dataset(logistic_df)
    yes = logistic_regression(data, target="y", features=["x1", "x2"], positive_class="yes")
    no = logistic_regression(data, target="y", features=["x1", "x2"], positive_class="no")
    assert no.positive_class == "no"
    assert no.classes == ["yes", "no"]
    assert no.coefficients["x1"] == pytest.approx(-yes.coefficients["x1"])
    assert no.intercept == pytest.approx(-yes.intercept)
    assert no.test_metrics.roc_auc == pytest.approx(yes.test_metrics.roc_auc)


def test_boolean_target(logistic_df):
    df = logistic_df.assign(y=logistic_df["y"] == "yes")
    result = logistic_regression(Dataset(df), target="y", features=["x1", "x2"])
    assert result.positive_class is True or result.positive_class == True  # noqa: E712
    assert result.test_metrics.accuracy > 0.6


def test_categorical_features_are_onehot_encoded(logistic_df):
    result = logistic_regression(Dataset(logistic_df), target="y")
    assert any(name.startswith("group_") for name in result.coefficients)


def test_missing_target_rows_dropped(logistic_df):
    df = logistic_df.copy()
    df.loc[:9, "y"] = np.nan
    result = logistic_regression(Dataset(df), target="y", features=["x1", "x2"])
    assert result.n_train + result.n_test == len(df) - 10
    assert any("missing target" in w for w in result.warnings)


def test_imbalance_warning(logistic_df):
    df = logistic_df.copy()
    df["y"] = "no"
    df.loc[:9, "y"] = "yes"
    result = logistic_regression(Dataset(df), target="y", features=["x1", "x2"])
    assert any("imbalance" in w for w in result.warnings)


def test_positive_class_only_for_two_classes(mixed_df):
    with pytest.raises(ValueError, match="two-class"):
        logistic_regression(Dataset(mixed_df), target="city", positive_class="Oslo")


def test_rejects_continuous_target(mixed_df):
    with pytest.raises(ValueError, match="regression"):
        logistic_regression(Dataset(mixed_df), target="income")


def test_rejects_unusable_target_type(mixed_df):
    with pytest.raises(ValueError, match="requires a categorical"):
        logistic_regression(Dataset(mixed_df), target="signup")


def test_rejects_unknown_positive_class(logistic_df):
    with pytest.raises(ValueError, match="positive_class"):
        logistic_regression(Dataset(logistic_df), target="y", positive_class="maybe")


def test_rejects_target_in_features(logistic_df):
    with pytest.raises(ValueError, match="cannot also be a feature"):
        logistic_regression(Dataset(logistic_df), target="y", features=["x1", "y"])


def test_result_is_json_serializable(logistic_df):
    payload = logistic_regression(Dataset(logistic_df), target="y").to_dict()
    json.dumps(payload)
    assert payload["train_metrics"]["n_obs"] == payload["n_train"]
    assert payload["test_metrics"]["labels"] == ["no", "yes"]
    assert isinstance(payload["odds_ratios"], dict)
    assert len(payload["test_actual"]) == len(payload["test_probability"]) == payload["n_test"]
    assert all(0.0 <= p <= 1.0 for p in payload["test_probability"])


def test_classification_metrics_known_values():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    y_score = [0.1, 0.6, 0.7, 0.9]
    m = classification_metrics(y_true, y_pred, y_score, labels=["a", "b"])
    assert m.accuracy == 0.75
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == 1.0
    assert m.confusion_matrix == [[1, 1], [0, 2]]
    assert m.roc_auc == 1.0


def test_classification_metrics_single_class_auc_is_nan_and_serializable():
    m = classification_metrics(["a", "a"], ["a", "a"], [0.1, 0.2], labels=["a", "b"])
    assert np.isnan(m.roc_auc)
    assert m.to_dict()["roc_auc"] is None
    json.dumps(m.to_dict())
