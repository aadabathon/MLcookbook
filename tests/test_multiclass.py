"""Multiclass targets across the three classifiers and the shared metrics."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from mlcookbook import Dataset, decision_tree, logistic_regression, random_forest
from mlcookbook.evaluation.metrics import classification_metrics


@pytest.fixture
def three_class_df() -> pd.DataFrame:
    """Three well-separated groups along x1; x2 is noise."""
    rng = np.random.default_rng(11)
    n = 300
    species = rng.choice(["setosa", "versicolor", "virginica"], n)
    centre = pd.Series(species).map({"setosa": -3.0, "versicolor": 0.0, "virginica": 3.0}).to_numpy()
    return pd.DataFrame({"x1": centre + rng.normal(0, 0.6, n), "x2": rng.normal(0, 1, n), "species": species})


@pytest.mark.parametrize("workflow", [logistic_regression, decision_tree, random_forest])
def test_classifiers_learn_three_classes(three_class_df, workflow):
    kwargs = {"n_estimators": 30} if workflow is random_forest else {}
    result = workflow(Dataset(three_class_df), target="species", **kwargs)
    m = result.test_metrics
    assert result.classes == ["setosa", "versicolor", "virginica"]
    assert result.positive_class is None and m.positive_class is None
    assert m.average == "macro" and len(m.confusion_matrix) == 3
    assert m.accuracy > 0.9 and m.roc_auc > 0.95
    assert [c.label for c in m.per_class] == result.classes
    assert sum(c.support for c in m.per_class) == m.n_obs == result.n_test
    assert result.cross_validation.strategy == "stratified k-fold"
    assert {"accuracy", "f1_macro", "roc_auc_ovr"} == set(result.cross_validation.mean)
    json.dumps(result.to_dict(), allow_nan=False)


def test_multinomial_logistic_exposes_per_class_coefficients(three_class_df):
    result = logistic_regression(Dataset(three_class_df), target="species")
    assert result.intercept is None and result.coefficients == {} and result.test_probability == []
    coefs = result.class_coefficients
    assert set(coefs) == {"setosa", "versicolor", "virginica"}
    assert coefs["virginica"]["x1"] > coefs["versicolor"]["x1"] > coefs["setosa"]["x1"]
    probs = np.array(result.test_class_probabilities)
    assert probs.shape == (result.n_test, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    predicted = [result.classes[i] for i in probs.argmax(axis=1)]
    assert predicted == result.test_predicted
    assert confusion_matrix(result.test_actual, result.test_predicted, labels=result.classes).tolist() == \
        result.test_metrics.confusion_matrix


def test_multiclass_tree_nodes_count_every_class(three_class_df):
    result = decision_tree(Dataset(three_class_df), target="species", max_depth=2)
    root = result.nodes[0]
    assert len(root.class_counts) == 3 and sum(root.class_counts) == pytest.approx(root.n_samples)
    assert root.feature == "x1"
    assert {r.prediction for r in result.rules} == set(result.classes)


def test_multiclass_forest_has_no_binary_roc(three_class_df):
    result = random_forest(Dataset(three_class_df), target="species", n_estimators=20)
    assert result.roc_fpr == result.roc_tpr == [] and result.test_probability == []
    assert len(result.test_class_probabilities) == result.n_test


def test_numeric_class_codes_are_classes_not_a_regression():
    rng = np.random.default_rng(2)
    grade = rng.integers(1, 4, 240)
    frame = pd.DataFrame({"x": grade + rng.normal(0, 0.3, 240), "grade": grade})
    result = logistic_regression(Dataset(frame), target="grade")
    assert result.classes == [1, 2, 3]


def test_multiclass_imbalance_warning(three_class_df):
    common = three_class_df[three_class_df.species != "virginica"]  # about 100 rows per class
    rare = three_class_df[three_class_df.species == "virginica"].head(12)
    result = logistic_regression(Dataset(pd.concat([common, rare])), target="species")
    assert any("Class imbalance" in w and "macro F1" in w for w in result.warnings)


def test_multiclass_metrics_known_values():
    labels = ["a", "b", "c"]
    y_true = ["a", "b", "c", "c"]
    y_pred = ["a", "b", "b", "c"]
    scores = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.5, 0.4], [0.1, 0.2, 0.7]]
    m = classification_metrics(y_true, y_pred, scores, labels)
    assert m.accuracy == 0.75
    assert m.recall == pytest.approx((1 + 1 + 0.5) / 3)  # macro recall
    assert m.balanced_accuracy == pytest.approx(m.recall)
    assert m.confusion_matrix == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert [c.support for c in m.per_class] == [1, 1, 2]
    with pytest.raises(ValueError, match="Shape"):
        classification_metrics(y_true, y_pred, [0.1, 0.2, 0.3, 0.4], labels)


def test_multiclass_auc_undefined_when_a_class_is_absent():
    m = classification_metrics(["a", "b"], ["a", "b"], [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]], ["a", "b", "c"])
    assert np.isnan(m.roc_auc) and m.to_dict()["roc_auc"] is None
