"""Fair folds, training-only preprocessing, selection, and reproducibility."""
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer

from mlcookbook import Dataset, PreprocessingConfig, compare_regressors


def test_comparison_round_trip_and_reproducibility(linear_df):
    data = Dataset(linear_df)
    result = compare_regressors(data, "y", n_estimators=10, cv_folds=3)
    repeat = compare_regressors(data, "y", n_estimators=10, cv_folds=3)
    assert result.to_dict() == repeat.to_dict()
    assert result.selected_model == "ridge"
    assert result.test_metrics.r2 > 0.98
    assert result.selection_metric == "rmse"
    assert len(result.train_rows) == 150 and len(result.test_rows) == 50
    assert set(result.train_rows).isdisjoint(result.test_rows)
    assert sorted(result.train_rows + result.test_rows) == list(range(200))
    assert sorted(row for fold in result.folds for row in fold["validation_rows"]) == sorted(result.train_rows)
    assert result.test_actual == linear_df.iloc[result.test_rows].y.tolist()
    json.dumps(result.to_dict(), allow_nan=False)
    for candidate in result.candidates:
        json.dumps(candidate.to_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        result.selected_model = "different"
    changed = compare_regressors(data, "y", n_estimators=10, cv_folds=3, random_state=41)
    assert changed.test_rows != result.test_rows


def test_candidates_fit_identical_folds_without_test_leakage(linear_df, monkeypatch):
    frame = linear_df.copy()
    frame.index = np.arange(len(frame)) + 1000
    frame.loc[1002, "y"] = np.nan
    frame.loc[1005:1010, "x1"] = np.nan
    fitted = []
    original = ColumnTransformer.fit_transform

    def record(self, X, *args, **kwargs):
        result = original(self, X, *args, **kwargs)
        numeric = self.named_transformers_["numeric"]
        assert numeric.named_steps["impute"].statistics_[0] == pytest.approx(X.x1.median())
        fitted.append(tuple(int(i - 1000) for i in X.index))
        return result

    monkeypatch.setattr(ColumnTransformer, "fit_transform", record)
    result = compare_regressors(Dataset(frame), "y", n_estimators=10, cv_folds=3)
    assert len(fitted) == 7  # each candidate's three folds, then only the selected model
    assert fitted[:3] == fitted[3:6]
    for actual, fold in zip(fitted[:3], result.folds):
        assert list(actual) == fold["train_rows"]
        assert set(actual).isdisjoint(fold["validation_rows"])
        assert set(actual) | set(fold["validation_rows"]) == set(result.train_rows)
    assert list(fitted[-1]) == result.train_rows
    assert all(set(rows).isdisjoint(result.test_rows) for rows in fitted)
    assert 2 not in result.train_rows + result.test_rows
    assert any("Dropped 1" in warning for warning in result.warnings)


def test_selection_does_not_consult_holdout_targets(linear_df):
    first = compare_regressors(Dataset(linear_df), "y", n_estimators=10, cv_folds=3)
    changed = linear_df.copy()
    changed.loc[first.test_rows, "y"] += 100_000
    second = compare_regressors(Dataset(changed), "y", n_estimators=10, cv_folds=3)
    assert first.candidates == second.candidates
    assert first.selected_model == second.selected_model
    assert first.test_predicted == second.test_predicted
    assert second.test_metrics.rmse > 90_000


def test_comparison_records_preprocessing_and_duplicate_indices(linear_df):
    frame = linear_df.copy()
    frame.index = [0] * len(frame)
    config = PreprocessingConfig(numeric_impute="mean", scaling="robust", missing_indicators=True)
    result = compare_regressors(Dataset(frame), "y", n_estimators=10, cv_folds=3, preprocessing=config)
    assert result.preprocessing.config == config
    assert sorted(result.train_rows + result.test_rows) == list(range(len(frame)))


@pytest.mark.parametrize("kwargs,match", [
    ({"cv_folds": 0}, "at least 2"), ({"cv_folds": 1}, "at least 2"),
    ({"test_size": 1}, "between"), ({"random_state": -1}, "random_state"),
    ({"alpha": float("nan")}, "positive"), ({"alpha": 0}, "positive"),
])
def test_invalid_comparison_settings(linear_df, kwargs, match):
    with pytest.raises(ValueError, match=match):
        compare_regressors(Dataset(linear_df), "y", **kwargs)


def test_comparison_requires_regression_target_and_enough_rows(linear_df):
    with pytest.raises(ValueError, match="numeric target"):
        compare_regressors(Dataset(linear_df), "group")
    with pytest.raises(ValueError, match="at least 4 training"):
        compare_regressors(Dataset(linear_df.head(5)), "y", test_size=0.5)
    result = compare_regressors(Dataset(linear_df.head(12)), "y", n_estimators=10, cv_folds=10)
    assert len(result.folds) == 4
    assert any("4 folds instead of 10" in warning for warning in result.warnings)


@pytest.mark.parametrize("splits,match", [
    ([(np.array([0]), np.array([1]))], "fold count"),
    ([(np.array([0, 1]), np.array([1, 2]))] * 2, "must not overlap"),
    ([(np.array([0, 0]), np.array([1]))] * 2, "unique positions"),
    ([(np.array([-1]), np.array([1]))] * 2, "unique positions"),
    ([(np.array([0]), np.array([999]))] * 2, "unique positions"),
])
def test_shared_cv_rejects_invalid_indices(linear_df, splits, match):
    from sklearn.linear_model import Ridge

    from mlcookbook.evaluation.cross_validation import cross_validate_pipeline

    with pytest.raises(ValueError, match=match):
        cross_validate_pipeline(Ridge(), linear_df[["x1"]], linear_df.y, "regression", folds=2, splits=splits)
