"""Support vector machines and neural networks (classification and regression)."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc

from mlcookbook import (
    Dataset,
    PreprocessingConfig,
    neural_network,
    neural_network_regression,
    svm,
    svm_regression,
)


@pytest.fixture
def circle_df() -> pd.DataFrame:
    """'inside' iff a point lies within radius 1: no straight line separates the classes."""
    rng = np.random.default_rng(12)
    xy = rng.uniform(-1.6, 1.6, (400, 2))
    inside = (xy ** 2).sum(axis=1) < 1
    return pd.DataFrame({"a": xy[:, 0], "b": xy[:, 1], "y": np.where(inside, "inside", "outside")})


@pytest.fixture
def curve_df() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    x = rng.uniform(-3, 3, 400)
    return pd.DataFrame({"x": x, "noise": rng.normal(size=400), "y": 50 * np.sin(x) + 200 + rng.normal(0, 2, 400)})


def test_rbf_svm_draws_curved_boundaries_a_linear_one_cannot(circle_df):
    data = Dataset(circle_df)
    rbf = svm(data, target="y")
    linear = svm(data, target="y", kernel="linear")
    assert rbf.test_metrics.accuracy > 0.9
    assert linear.test_metrics.accuracy < 0.8
    assert rbf.kernel == "rbf" and rbf.C == 1.0 and rbf.coefficients == {}
    assert set(linear.coefficients) == {"a", "b"}
    assert sum(rbf.support_vectors.values()) == round(rbf.support_vector_share * rbf.n_train)
    assert auc(rbf.roc_fpr, rbf.roc_tpr) == pytest.approx(rbf.test_metrics.roc_auc)
    assert len(rbf.test_probability) == rbf.n_test
    json.dumps(rbf.to_dict(), allow_nan=False)


def test_linear_svm_coefficients_follow_the_positive_class(logistic_df):
    data = Dataset(logistic_df)
    yes = svm(data, target="y", features=["x1", "x2"], kernel="linear", positive_class="yes")
    no = svm(data, target="y", features=["x1", "x2"], kernel="linear", positive_class="no")
    assert yes.coefficients["x1"] > 0 > yes.coefficients["x2"]
    assert no.coefficients["x1"] == pytest.approx(-yes.coefficients["x1"])


def test_svm_multiclass_and_overlap_warning(logistic_df):
    rng = np.random.default_rng(1)
    noise = pd.DataFrame({"x": rng.normal(size=300), "y": rng.choice(["a", "b", "c"], 300)})
    result = svm(Dataset(noise), target="y")
    assert result.positive_class is None and result.roc_fpr == []
    assert set(result.support_vectors) == {"a", "b", "c"}
    assert any("support vectors" in w for w in result.warnings)


@pytest.mark.parametrize("kwargs,message", [({"kernel": "sigmoid"}, "kernel"), ({"C": 0}, "C must")])
def test_svm_validation(logistic_df, kwargs, message):
    with pytest.raises(ValueError, match=message):
        svm(Dataset(logistic_df), target="y", **kwargs)
    with pytest.raises(ValueError, match=message):
        svm_regression(Dataset(logistic_df), target="x1", **kwargs)


def test_svm_regression_fits_a_curve_in_any_target_units(curve_df):
    result = svm_regression(Dataset(curve_df), target="y", C=10)
    assert result.test_metrics.r2 > 0.95  # the target is standardized, so C does not depend on its units
    assert result.epsilon == 0.1 and 0 < result.support_vector_share <= 1
    assert result.cross_validation.n_folds == 5
    json.dumps(result.to_dict(), allow_nan=False)


def test_svm_warns_without_scaling(circle_df):
    result = svm(Dataset(circle_df), target="y", preprocessing=PreprocessingConfig(scaling="none"))
    assert any("without scaling" in w for w in result.warnings)


def test_neural_network_learns_a_nonlinear_boundary(circle_df):
    result = neural_network(Dataset(circle_df), target="y", hidden_layers=(32, 16), max_epochs=800)
    t = result.training
    assert result.test_metrics.accuracy > 0.85
    assert t.architecture == [2, 32, 16, 1]
    assert t.n_parameters == 2 * 32 + 32 + 32 * 16 + 16 + 16 + 1
    assert t.epochs == len(t.loss_curve) == len(t.validation_curve)
    assert t.loss_curve[-1] < t.loss_curve[0]
    assert t.early_stopping and t.validation_metric == "accuracy"
    assert result.classes == ["inside", "outside"]
    assert set(result.test_predicted) <= {"inside", "outside"}  # decoded from the network's class codes
    json.dumps(result.to_dict(), allow_nan=False)


def test_neural_network_multiclass_and_positive_class(three_class_frame):
    result = neural_network(Dataset(three_class_frame), target="species")
    assert result.training.architecture[-1] == 3
    assert result.test_metrics.accuracy > 0.85 and result.test_probability == []
    assert np.array(result.test_class_probabilities).shape == (result.n_test, 3)


def test_neural_network_positive_class_orders_probabilities(logistic_df):
    data = Dataset(logistic_df)
    yes = neural_network(data, target="y", features=["x1", "x2"], positive_class="yes", cv_folds=0)
    no = neural_network(data, target="y", features=["x1", "x2"], positive_class="no", cv_folds=0)
    np.testing.assert_allclose(yes.test_probability, 1 - np.array(no.test_probability), atol=1e-6)
    assert yes.test_metrics.roc_auc == pytest.approx(no.test_metrics.roc_auc)


def test_neural_network_reports_non_convergence(logistic_df):
    result = neural_network(Dataset(logistic_df), target="y", max_epochs=2, early_stopping=False, cv_folds=0)
    assert not result.training.converged and result.training.validation_curve == []
    assert any("epoch limit" in w for w in result.warnings)


@pytest.mark.parametrize("kwargs,message", [
    ({"hidden_layers": ()}, "hidden_layers"), ({"hidden_layers": (0,)}, "hidden_layers"),
    ({"activation": "gelu"}, "activation"), ({"alpha": -1}, "alpha"),
    ({"learning_rate": 0}, "learning_rate"), ({"max_epochs": 0}, "max_epochs"),
])
def test_neural_network_validation(logistic_df, kwargs, message):
    with pytest.raises(ValueError, match=message):
        neural_network(Dataset(logistic_df), target="y", **kwargs)


def test_neural_network_regression_learns_a_curve(curve_df):
    result = neural_network_regression(Dataset(curve_df), target="y", max_epochs=1000)
    assert result.test_metrics.r2 > 0.9
    assert result.training.architecture == [2, 64, 32, 1] and result.training.validation_metric == "r2"
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.fixture
def three_class_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    species = rng.choice(["setosa", "versicolor", "virginica"], 300)
    centre = pd.Series(species).map({"setosa": -3.0, "versicolor": 0.0, "virginica": 3.0}).to_numpy()
    return pd.DataFrame({"x1": centre + rng.normal(0, 0.6, 300), "x2": rng.normal(0, 1, 300), "species": species})
