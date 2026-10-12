"""Configurable preprocessing, preprocessing summaries, data quality, and cross-validation."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from mlcookbook import Dataset, decision_tree, linear_regression, logistic_regression, random_forest
from mlcookbook.core.quality import column_issues
from mlcookbook.evaluation import cross_validate_pipeline
from mlcookbook.preprocessing import (
    PreprocessingConfig,
    build_preprocessor,
    describe_preprocessor,
    resolve_features,
)
from mlcookbook.preprocessing.builders import resolve_config


@pytest.mark.parametrize("kwargs", [
    {"numeric_impute": "mode"}, {"scaling": "zscore"}, {"categorical_encoding": "target"},
    {"categorical_impute": "drop"}, {"min_category_frequency": 1.5}, {"min_category_frequency": 0},
    {"min_category_frequency": True},
])
def test_config_rejects_unknown_options(kwargs):
    with pytest.raises(ValueError):
        PreprocessingConfig(**kwargs)


def test_resolve_config_fills_workflow_default_scaling():
    assert resolve_config(None, scale=True).scaling == "standard"
    assert resolve_config(None, scale=False).scaling == "none"
    assert resolve_config(PreprocessingConfig(scaling="robust"), scale=False).scaling == "robust"


@pytest.mark.parametrize("scaling,check", [
    ("standard", lambda x: abs(x.mean()) < 1e-9 and abs(x.std() - 1) < 1e-9),
    ("minmax", lambda x: x.min() == 0 and x.max() == 1),
    ("robust", lambda x: abs(np.median(x)) < 1e-9),
    ("none", lambda x: x.min() >= 20),
])
def test_scaling_options(scaling, check):
    frame = pd.DataFrame({"age": np.linspace(20, 80, 61)})
    out = build_preprocessor(Dataset(frame), ["age"], config=PreprocessingConfig(scaling=scaling)).fit_transform(frame)
    assert check(out[:, 0])


def test_mean_imputation_and_missing_indicator():
    frame = pd.DataFrame({"x": [1.0, 2.0, np.nan, 9.0], "y": [1.0, 2.0, 3.0, 4.0]})
    config = PreprocessingConfig(numeric_impute="mean", scaling="none", missing_indicators=True)
    pre = build_preprocessor(Dataset(frame), ["x", "y"], config=config)
    out = pre.fit_transform(frame)
    # x, y, then an indicator for x only (y had nothing missing)
    assert list(pre.get_feature_names_out()) == ["x", "y", "missingindicator_x"]
    assert out[2, 0] == pytest.approx(4.0)
    assert out[:, 2].tolist() == [0, 0, 1, 0]
    summary = describe_preprocessor(pre, frame, Dataset(frame), config)
    assert summary.columns[0].output_features == ["x", "missingindicator_x"]
    assert summary.columns[1].output_features == ["y"]


def test_missing_category_keeps_missingness_visible():
    frame = pd.DataFrame({"code": pd.Series([1, 2, None, 2], dtype="Int64").astype("category")})
    config = PreprocessingConfig(categorical_impute="missing_category")
    pre = build_preprocessor(Dataset(frame), ["code"], config=config)
    pre.fit(frame)
    assert list(pre.get_feature_names_out()) == ["code_(missing)", "code_1", "code_2"]


def test_rare_levels_are_pooled_and_unseen_levels_map_to_them():
    frame = pd.DataFrame({"city": ["Oslo"] * 10 + ["Bergen"] * 8 + ["Moss", "Hamar"]})
    pre = build_preprocessor(Dataset(frame), ["city"], config=PreprocessingConfig(min_category_frequency=3))
    pre.fit(frame)
    names = list(pre.get_feature_names_out())
    assert names == ["city_Bergen", "city_Oslo", "city_infrequent_sklearn"]
    unseen = pre.transform(pd.DataFrame({"city": ["Narvik"]}))
    assert unseen.tolist() == [[0, 0, 1]]


def test_ordinal_encoding_and_unknown_code():
    frame = pd.DataFrame({"size": ["s", "m", "l", "m"]})
    pre = build_preprocessor(Dataset(frame), ["size"], config=PreprocessingConfig(categorical_encoding="ordinal"))
    pre.fit(frame)
    assert pre.transform(frame)[:, 0].tolist() == [2, 1, 0, 1]  # sorted: l, m, s
    assert pre.transform(pd.DataFrame({"size": ["xl"]}))[0, 0] == -1


def test_datetime_columns_expand_to_numeric_parts():
    frame = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-03-15", None, "2025-07-06"])})
    pre = build_preprocessor(Dataset(frame), ["when"], scale=False)
    out = pre.fit_transform(frame)
    assert list(pre.get_feature_names_out()) == ["when_days", "when_year", "when_month", "when_dayofweek"]
    assert out[0].tolist() == [19723.0, 2024.0, 1.0, 0.0]  # 2024-01-01 was a Monday
    assert not np.isnan(out).any()  # the missing date was imputed part by part


def test_timezone_aware_dates_are_converted_to_utc():
    frame = pd.DataFrame({"when": pd.to_datetime(["2024-01-01T23:00:00-02:00", "2024-01-02T00:00:00+00:00"], utc=True)})
    out = build_preprocessor(Dataset(frame), ["when"], scale=False).fit_transform(frame)
    assert out[0, 0] == pytest.approx(19724 + 1 / 24)


def test_datetime_role_requires_parsed_dates():
    frame = pd.DataFrame({"when": ["2024-01-01", "2024-02-01"]})
    with pytest.raises(ValueError, match="not stored as dates"):
        build_preprocessor(Dataset(frame, overrides={"when": "datetime"}), ["when"])


def test_summary_reports_values_learned_from_training_rows_only():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.normal(10, 2, 100), "city": rng.choice(["a", "b"], 100)})
    frame.loc[:4, "x"] = np.nan
    frame.loc[5, "city"] = None
    train, test = frame.iloc[:80], frame.iloc[80:]
    data = Dataset(frame)
    config = resolve_config(None)
    pre = build_preprocessor(data, ["x", "city"], config=config).fit(train)
    summary = describe_preprocessor(pre, train, data, config)
    assert summary.n_rows_fitted == 80
    assert summary.n_output_features == 3
    x, city = summary.columns
    assert x.column == "x" and x.n_missing_fit == 5 and x.output_features == ["x"]
    assert f"{train['x'].median():.4g}" in x.steps[0]
    assert f"{train['x'].mean():.4g}" in x.steps[1]
    assert city.n_missing_fit == 1 and city.output_features == ["city_a", "city_b"]
    assert "one-hot" in city.steps[-1]
    json.dumps(summary.to_dict(), allow_nan=False)


def test_quality_issues_flag_misleading_columns():
    n = 40
    frame = pd.DataFrame({
        "row_id": range(1000, 1000 + n),
        "same": [1.0] * n,
        "empty": [np.nan] * n,
        "sparse": [1.0, 2.0] * 5 + [np.nan] * 30,
        "measure": np.linspace(0, 1, n),  # unique, but floats are measurements
        "zip": pd.Series([f"{i:05d}" for i in range(n)], dtype="category"),
    })
    data = Dataset(frame)
    kinds = {(i.column, i.kind) for i in column_issues(data.df, data.schema())}
    assert ("row_id", "likely_id") in kinds and ("zip", "likely_id") in kinds
    assert ("same", "constant") in kinds and ("empty", "all_missing") in kinds
    assert ("sparse", "high_missing") in kinds
    assert not any(column == "measure" for column, _ in kinds)
    assert data.describe().issues  # surfaced by Dataset.describe()


def test_resolve_features_validation_and_quality_warnings():
    frame = pd.DataFrame({"id": range(30), "x": np.linspace(0, 1, 30), "y": [0, 1] * 15})
    features, warnings = resolve_features(Dataset(frame), "y", None)
    assert features == ["id", "x"]
    assert any("row identifier" in w for w in warnings)
    with pytest.raises(ValueError, match="unique"):
        resolve_features(Dataset(frame), "y", ["x", "x"])


def test_cross_validation_reports_every_fold(linear_df):
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    data = Dataset(linear_df)
    model = Pipeline([("preprocess", build_preprocessor(data, ["x1", "x2"])), ("m", LinearRegression())])
    cv = cross_validate_pipeline(model, linear_df[["x1", "x2"]], linear_df["y"], "regression", folds=4)
    assert cv.n_folds == len(cv.folds) == 4 and cv.strategy == "k-fold"
    assert sum(f.n_validation for f in cv.folds) == len(linear_df)
    assert cv.mean["r2"] > 0.99 and cv.mean["rmse"] > 0  # error metrics reported positive
    assert cv.mean["rmse"] == pytest.approx(np.mean([f.scores["rmse"] for f in cv.folds]))
    assert cv.std["r2"] == pytest.approx(np.std([f.scores["r2"] for f in cv.folds], ddof=1))
    json.dumps(cv.to_dict(), allow_nan=False)


def test_cross_validation_rejects_bad_arguments(linear_df):
    with pytest.raises(ValueError, match="task"):
        cross_validate_pipeline(None, linear_df, linear_df["y"], "clustering")
    with pytest.raises(ValueError, match="2 folds"):
        cross_validate_pipeline(None, linear_df, linear_df["y"], "regression", folds=1)


@pytest.mark.parametrize("workflow,fixture,task", [
    (linear_regression, "linear_df", "regression"),
    (logistic_regression, "logistic_df", "binary"),
    (decision_tree, "logistic_df", "binary"),
    (random_forest, "logistic_df", "binary"),
])
def test_workflows_cross_validate_on_training_rows(request, workflow, fixture, task):
    frame = request.getfixturevalue(fixture)
    result = workflow(Dataset(frame), target="y")
    cv = result.cross_validation
    assert cv.n_folds == 5 and cv.n_rows == result.n_train
    assert cv.strategy == ("k-fold" if task == "regression" else "stratified k-fold")
    assert "r2" in cv.mean if task == "regression" else {"accuracy", "f1", "roc_auc"} <= set(cv.mean)
    assert result.preprocessing.n_rows_fitted == result.n_train
    assert workflow(Dataset(frame), target="y", cv_folds=0).cross_validation is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_cross_validation_shrinks_folds_for_small_classes(logistic_df):
    logistic_df["y"] = "no"
    logistic_df.loc[:3, "y"] = "yes"  # 3 of the 4 "yes" rows land in training
    result = logistic_regression(Dataset(logistic_df), target="y", features=["x1"], cv_folds=5)
    assert result.cross_validation.n_folds == 3
    assert any("3 folds instead of 5" in w for w in result.warnings)


def test_workflows_record_the_preprocessing_they_used(logistic_df):
    config = PreprocessingConfig(scaling="robust", categorical_encoding="ordinal", min_category_frequency=0.05)
    result = logistic_regression(Dataset(logistic_df), target="y", preprocessing=config)
    assert result.preprocessing.config == config
    assert result.coefficients.keys() == {"x1", "x2", "group"}
    assert any("Ordinal encoding" in w for w in result.warnings)
    tree = decision_tree(Dataset(logistic_df), target="y")
    assert tree.preprocessing.config.scaling == "none"
    assert any("not scaled" in step for c in tree.preprocessing.columns for step in c.steps)


def test_datetime_features_flow_through_a_workflow():
    rng = np.random.default_rng(4)
    when = pd.date_range("2020-01-01", periods=300, freq="D")
    frame = pd.DataFrame({"when": when, "y": np.arange(300) * 2.0 + rng.normal(0, 1, 300)})
    result = linear_regression(Dataset(frame), target="y")
    assert result.features == ["when"]
    assert {"when_days", "when_year", "when_month", "when_dayofweek"} == set(result.coefficients)
    assert result.test_metrics.r2 > 0.99  # the trend lives in when_days


def test_preprocessing_is_refit_inside_each_fold(linear_df, monkeypatch):
    """No fold's validation rows may influence that fold's fitted preprocessing."""
    import mlcookbook.regression.linear as linear_module

    train, _ = train_test_split(linear_df.index, test_size=0.25, random_state=0)
    seen = []
    original = linear_module.Pipeline.fit

    def fit(pipeline, X, y=None, **kwargs):
        seen.append(set(X.index))
        return original(pipeline, X, y, **kwargs)

    monkeypatch.setattr(linear_module.Pipeline, "fit", fit)
    result = linear_regression(Dataset(linear_df), target="y", cv_folds=4)
    assert seen[0] == set(train)
    assert len(seen) == 5 and all(len(rows) < len(train) for rows in seen[1:])
    assert result.cross_validation.n_folds == 4


def test_summary_attributes_grouped_outputs_to_each_column():
    frame = pd.DataFrame({
        "city": ["Oslo"] * 10 + ["Bergen"] * 8 + ["Moss", "Hamar"],
        "size": ["s", "m"] * 10,
        "when": pd.date_range("2024-01-01", periods=20),
        "x": np.arange(20.0),
    })
    data = Dataset(frame)
    config = resolve_config(PreprocessingConfig(min_category_frequency=3))
    pre = build_preprocessor(data, list(frame.columns), config=config).fit(frame)
    summary = describe_preprocessor(pre, frame, data, config)
    outputs = {c.column: c.output_features for c in summary.columns}
    assert [c.column for c in summary.columns] == ["city", "size", "when", "x"]  # input order
    assert outputs["city"] == ["city_Bergen", "city_Oslo", "city_infrequent_sklearn"]
    assert outputs["size"] == ["size_m", "size_s"]
    assert outputs["when"] == ["when_days", "when_year", "when_month", "when_dayofweek"]
    assert sum(len(o) for o in outputs.values()) == summary.n_output_features == len(pre.get_feature_names_out())
