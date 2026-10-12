"""Ridge/lasso, residual diagnostics, and tree-based regression."""
import json

import numpy as np
import pandas as pd
import pytest

from mlcookbook import (
    Dataset,
    PreprocessingConfig,
    decision_tree_regression,
    linear_regression,
    random_forest_regression,
)
from mlcookbook.regression import residual_diagnostics


@pytest.fixture
def step_df() -> pd.DataFrame:
    """y jumps from 10 to 30 at x = 5 (a step a tree finds exactly); noise is irrelevant."""
    rng = np.random.default_rng(8)
    n = 400
    x = rng.uniform(0, 10, n)
    return pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": np.where(x > 5, 30.0, 10.0) + rng.normal(0, 1, n)})


def test_ridge_shrinks_and_lasso_zeroes(linear_df):
    rng = np.random.default_rng(3)
    frame = linear_df.assign(**{f"junk{i}": rng.normal(size=len(linear_df)) for i in range(5)})
    data = Dataset(frame)
    ols = linear_regression(data, target="y", features=["x1", "x2", "junk0", "junk1", "junk2", "junk3", "junk4"])
    ridge = linear_regression(data, target="y", features=ols.features, penalty="ridge", alpha=50)
    lasso = linear_regression(data, target="y", features=ols.features, penalty="lasso", alpha=0.2)
    assert ols.penalty == "none" and ols.alpha is None and ols.n_zero_coefficients == 0
    assert abs(ridge.coefficients["x1"]) < abs(ols.coefficients["x1"])
    assert lasso.n_zero_coefficients >= 3
    assert lasso.coefficients["x1"] != 0
    assert ridge.alpha == 50 and lasso.penalty == "lasso"
    json.dumps(lasso.to_dict(), allow_nan=False)


@pytest.mark.parametrize("kwargs,message", [({"penalty": "elastic"}, "penalty"), ({"penalty": "ridge", "alpha": 0}, "alpha")])
def test_penalty_validation(linear_df, kwargs, message):
    with pytest.raises(ValueError, match=message):
        linear_regression(Dataset(linear_df), target="y", **kwargs)


def test_penalty_on_unscaled_features_warns(linear_df):
    result = linear_regression(Dataset(linear_df), target="y", penalty="ridge",
                               preprocessing=PreprocessingConfig(scaling="none"))
    assert any("unscaled" in w for w in result.warnings)


def test_residual_diagnostics_flag_heteroscedasticity_and_skew():
    rng = np.random.default_rng(1)
    fitted = np.linspace(1, 100, 500)
    skewed = fitted + rng.exponential(fitted / 5)  # errors grow with the fit and are right-skewed
    warnings = []
    d = residual_diagnostics(skewed, fitted, warnings)
    assert d.normality_test == "shapiro" and d.normality_p_value < 0.05
    assert d.spread_correlation > 0.3 and d.spread_p_value < 0.05 and d.skewness > 0
    assert any("normality" in w for w in warnings) and any("heteroscedastic" in w for w in warnings)


def test_well_behaved_residuals_do_not_warn(linear_df):
    result = linear_regression(Dataset(linear_df), target="y", features=["x1", "x2"])
    d = result.residual_diagnostics
    assert d.n_obs == result.n_train and abs(d.mean) < 1e-9
    assert not any("heteroscedastic" in w for w in result.warnings)


def test_degenerate_residuals_are_not_tested():
    d = residual_diagnostics([1.0, 2.0], [1.0, 2.0], [])
    assert d.normality_test == "none" and np.isnan(d.normality_p_value)


def test_regression_tree_finds_the_step(step_df):
    result = decision_tree_regression(Dataset(step_df), target="y", max_depth=1)
    root = result.nodes[0]
    assert root.feature == "x" and root.threshold == pytest.approx(5, abs=0.3)
    left, right = result.nodes[root.left], result.nodes[root.right]
    assert left.value == pytest.approx(10, abs=0.5) and right.value == pytest.approx(30, abs=0.5)
    assert result.test_metrics.r2 > 0.95
    assert [r.prediction for r in result.rules] == [left.value, right.value]
    assert all(r.std == pytest.approx(1, abs=0.3) for r in result.rules)  # the noise level
    assert sum(result.feature_importances.values()) == pytest.approx(1)
    assert [d.max_depth for d in result.depth_curve] == list(range(1, 9))
    assert result.cross_validation.strategy == "k-fold"
    json.dumps(result.to_dict(), allow_nan=False)


def test_regression_tree_node_bookkeeping(step_df):
    result = decision_tree_regression(Dataset(step_df), target="y", max_depth=3)
    assert result.nodes[0].n_samples == result.n_train
    for node in result.nodes:
        if node.left is not None:
            assert result.nodes[node.left].n_samples + result.nodes[node.right].n_samples == node.n_samples
    assert len(result.rules) == result.n_leaves


def test_regression_tree_validation(mixed_df, step_df):
    with pytest.raises(ValueError, match="numeric target"):
        decision_tree_regression(Dataset(mixed_df), target="city")
    with pytest.raises(ValueError, match="max_depth"):
        decision_tree_regression(Dataset(step_df), target="y", max_depth=0)


def test_regression_forest_learns_and_reports(step_df):
    result = random_forest_regression(Dataset(step_df), target="y", n_estimators=40)
    assert result.test_metrics.r2 > 0.95
    assert result.feature_importances["x"] > result.feature_importances["noise"]
    assert len(result.tree_depths) == 40 and result.max_features == 1.0
    assert len(result.test_actual) == len(result.test_predicted) == result.n_test
    assert any("not causal" in w for w in result.warnings)
    json.dumps(result.to_dict(), allow_nan=False)


def test_regression_forest_overfit_warning_on_noise():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=300) for i in range(5)})
    frame["y"] = rng.normal(size=300)
    result = random_forest_regression(Dataset(frame), target="y", n_estimators=30, min_samples_leaf=1)
    assert any("overfitting" in w for w in result.warnings)


def test_two_valued_numeric_target_suggests_a_classifier(logistic_df):
    frame = logistic_df.assign(y=(logistic_df["y"] == "yes").astype(float))
    result = linear_regression(Dataset(frame), target="y")
    assert any("classifier" in w for w in result.warnings)
