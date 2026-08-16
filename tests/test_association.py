import json

import numpy as np
import pandas as pd
import pytest

from mlcookbook import Dataset, correlation


def test_pearson_detects_strong_linear_relationship(linear_df):
    data = Dataset(linear_df.assign(x1_scaled=linear_df["x1"] * 2 + 1))
    result = correlation(data, "x1", "x1_scaled", method="pearson")
    assert result.statistic == pytest.approx(1.0)
    assert result.p_value < 1e-10
    assert result.n_obs == len(linear_df)
    assert result.warnings == []


def test_spearman_handles_monotonic_nonlinear():
    x = np.linspace(1, 10, 30)
    data = Dataset(pd.DataFrame({"x": x, "y": np.exp(x)}))
    result = correlation(data, "x", "y", method="spearman")
    assert result.statistic == pytest.approx(1.0)


def test_missing_values_dropped_with_warning():
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, np.nan], "b": [2.0, 4, 6, 8, 10]})
    result = correlation(Dataset(df), "a", "b")
    assert result.n_obs == 4
    assert any("Dropped 1" in w for w in result.warnings)


def test_rejects_non_numeric_column(mixed_df):
    with pytest.raises(ValueError, match="numeric"):
        correlation(Dataset(mixed_df), "age", "city")


def test_rejects_too_few_observations():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="at least"):
        correlation(Dataset(df), "a", "b")


def test_result_is_json_serializable(linear_df):
    result = correlation(Dataset(linear_df), "x1", "y")
    payload = result.to_dict()
    json.dumps(payload)
    assert payload["method"] == "pearson"
    assert isinstance(payload["statistic"], float)
