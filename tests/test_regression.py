import json

import numpy as np
import pytest

from mlcookbook import Dataset, linear_regression


def test_recovers_linear_relationship(linear_df):
    result = linear_regression(Dataset(linear_df), target="y", features=["x1", "x2"])
    assert result.test_metrics.r2 > 0.99
    # Numeric features are standardized, so coefficients are per-SD of X;
    # with SD(x) ~= 1 here they should be close to the true 3 and -2.
    assert result.coefficients["x1"] == pytest.approx(3.0, abs=0.3)
    assert result.coefficients["x2"] == pytest.approx(-2.0, abs=0.3)
    assert result.n_train + result.n_test == len(linear_df)


def test_default_features_skip_unusable_columns(mixed_df):
    result = linear_regression(Dataset(mixed_df), target="income")
    assert "signup" not in result.features
    assert "notes" not in result.features
    assert set(result.features) == {"age", "city", "is_member", "constant"}
    assert any("Skipped" in w for w in result.warnings)


def test_categorical_features_are_onehot_encoded(linear_df):
    result = linear_regression(Dataset(linear_df), target="y")
    assert any(name.startswith("group_") for name in result.coefficients)


def test_missing_target_rows_dropped(linear_df):
    df = linear_df.copy()
    df.loc[:9, "y"] = np.nan
    result = linear_regression(Dataset(df), target="y", features=["x1", "x2"])
    assert result.n_train + result.n_test == len(df) - 10
    assert any("missing target" in w for w in result.warnings)


def test_rejects_non_numeric_target(mixed_df):
    with pytest.raises(ValueError, match="numeric target"):
        linear_regression(Dataset(mixed_df), target="city")


def test_rejects_target_in_features(linear_df):
    with pytest.raises(ValueError, match="cannot also be a feature"):
        linear_regression(Dataset(linear_df), target="y", features=["x1", "y"])


def test_result_is_json_serializable(linear_df):
    payload = linear_regression(Dataset(linear_df), target="y").to_dict()
    json.dumps(payload)
    assert payload["train_metrics"]["n_obs"] == payload["n_train"]
    assert isinstance(payload["coefficients"], dict)
