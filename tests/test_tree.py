import json

import numpy as np
import pandas as pd
import pytest

from mlcookbook import Dataset, decision_tree
from mlcookbook.preprocessing.builders import build_preprocessor


@pytest.fixture
def threshold_df() -> pd.DataFrame:
    """'high' iff age > 50 (with a little label noise); age is in real-world units."""
    rng = np.random.default_rng(3)
    n = 300
    age = rng.uniform(20, 80, n)
    noise = rng.normal(0, 1, n)
    y = np.where(age + rng.normal(0, 2, n) > 50, "high", "low")
    return pd.DataFrame({"age": age, "noise": noise, "y": y})


def test_finds_the_true_split_in_real_units(threshold_df):
    result = decision_tree(Dataset(threshold_df), target="y", max_depth=1)
    root = result.nodes[0]
    assert root.feature == "age"
    assert root.threshold == pytest.approx(50, abs=4)  # not a standardized value
    assert result.test_metrics.accuracy > 0.9
    assert result.feature_importances["age"] > 0.9


def test_node_structure_is_consistent(logistic_df):
    result = decision_tree(Dataset(logistic_df), target="y", max_depth=3)
    nodes = result.nodes
    assert nodes[0].n_samples == result.n_train
    leaves = [n for n in nodes if n.left is None]
    assert len(leaves) == result.n_leaves == len(result.rules)
    for n in nodes:
        assert sum(n.class_counts) == pytest.approx(n.n_samples)
        if n.left is not None:
            assert nodes[n.left].n_samples + nodes[n.right].n_samples == n.n_samples
            assert nodes[n.left].depth == nodes[n.right].depth == n.depth + 1
            assert n.feature is not None and n.threshold is not None
    assert sum(r.n_samples for r in result.rules) == result.n_train
    assert result.depth <= 3


def test_max_depth_is_respected(logistic_df):
    shallow = decision_tree(Dataset(logistic_df), target="y", max_depth=2)
    assert shallow.depth <= 2
    assert max(n.depth for n in shallow.nodes) == shallow.depth


def test_feature_importances_sum_to_one(logistic_df):
    result = decision_tree(Dataset(logistic_df), target="y")
    assert sum(result.feature_importances.values()) == pytest.approx(1.0)


def test_depth_curve_shape(logistic_df):
    result = decision_tree(Dataset(logistic_df), target="y", max_depth=3)
    assert [d.max_depth for d in result.depth_curve] == list(range(1, 9))
    # Training accuracy can only rise with depth; that is the overfitting story.
    train = [d.train_accuracy for d in result.depth_curve]
    assert train[-1] >= train[0]


def test_positive_class_orders_counts_and_labels(logistic_df):
    data = Dataset(logistic_df)
    yes = decision_tree(data, target="y", positive_class="yes")
    no = decision_tree(data, target="y", positive_class="no")
    assert yes.classes == ["no", "yes"] and no.classes == ["yes", "no"]
    assert yes.nodes[0].class_counts == list(reversed(no.nodes[0].class_counts))


def test_overfitting_warning_on_noise():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({f"x{i}": rng.normal(size=300) for i in range(5)})
    df["y"] = rng.choice(["a", "b"], 300)  # pure noise
    result = decision_tree(Dataset(df), target="y", max_depth=20, min_samples_leaf=1)
    assert any("overfitting" in w for w in result.warnings)


def test_categorical_features_supported(logistic_df):
    result = decision_tree(Dataset(logistic_df), target="y")
    assert any(name.startswith("group_") for name in result.feature_importances)


def test_missing_target_rows_dropped(logistic_df):
    df = logistic_df.copy()
    df.loc[:9, "y"] = np.nan
    result = decision_tree(Dataset(df), target="y", features=["x1", "x2"])
    assert result.n_train + result.n_test == len(df) - 10
    assert any("missing target" in w for w in result.warnings)


def test_rejects_multiclass_and_bad_depth(mixed_df, logistic_df):
    with pytest.raises(ValueError, match="binary"):
        decision_tree(Dataset(mixed_df), target="city")
    with pytest.raises(ValueError, match="max_depth"):
        decision_tree(Dataset(logistic_df), target="y", max_depth=0)


def test_result_is_json_serializable(logistic_df):
    payload = decision_tree(Dataset(logistic_df), target="y").to_dict()
    json.dumps(payload)
    assert payload["nodes"][0]["id"] == 0
    assert payload["nodes"][0]["left"] is not None
    assert len(payload["depth_curve"]) >= 1
    assert payload["rules"][0]["conditions"]


def test_preprocessor_scale_flag(threshold_df):
    data = Dataset(threshold_df)
    scaled = build_preprocessor(data, ["age"]).fit_transform(threshold_df[["age"]])
    raw = build_preprocessor(data, ["age"], scale=False).fit_transform(threshold_df[["age"]])
    assert abs(scaled.mean()) < 1e-6
    assert raw.min() >= 20


def test_rule_text_is_ascii_so_windows_consoles_can_print_it(logistic_df):
    result = decision_tree(Dataset(logistic_df), target="y")
    for rule in result.rules:
        for condition in rule.conditions:
            condition.encode("cp1252")  # would raise on e.g. a unicode "≤"
