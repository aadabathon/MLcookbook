"""Workflow discovery shares validation, preserves data and never fits."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlcookbook.core.dataset import Dataset
from mlcookbook.workflows import discover_workflows


@pytest.fixture
def data():
    return Dataset(pd.DataFrame({
        "row_id": range(40), "signal": np.linspace(0, 1, 40),
        "outcome": np.linspace(0, 4, 40) ** 2, "label": ["yes", "no"] * 20,
        "group": ["a", "b", "c", "d"] * 10,
    }))


def option(result, key, task=None):
    return next(item for item in result.workflows if item.key == key and (task is None or item.task == task))


def test_discovery_is_json_safe_frozen_and_never_fits_or_changes_data(data, monkeypatch):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("Discovery must not fit preprocessing or models.")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", forbidden)
    before = data.df.copy(deep=True)
    result = discover_workflows(data, target="outcome", features=["signal"])
    assert option(result, "linear-regression").available
    assert option(result, "pca").available
    assert_frame_equal(before, data.df)
    json.dumps(result.to_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        result.warnings = []


def test_suggested_inputs_exclude_identifiers_but_explicit_choice_remains_possible(data):
    initial = discover_workflows(data)
    identifier = next(column for column in initial.columns if column.name == "row_id")
    assert identifier.identifier_like and not identifier.suggested
    assert "identifier" in identifier.reason
    assert "row_id" not in initial.context["features"]
    explicit = discover_workflows(data, target="outcome", features=["signal", "row_id"])
    regression = option(explicit, "linear-regression")
    assert regression.available
    assert any("row identifier" in warning for warning in regression.warnings)
    assert data.schema().var_type("row_id").value == "numeric"


def test_missing_target_is_a_specific_requirement_and_discovery_still_offers_other_work(data):
    result = discover_workflows(data)
    regression = option(result, "linear-regression")
    classification = option(result, "logistic-regression")
    assert not regression.available and "Select a numeric target" in regression.reason
    assert not classification.available and "Select a categorical" in classification.reason
    assert option(result, "association-scan").available
    assert option(result, "kmeans").available


def test_target_validation_uses_existing_shared_problem_rules(data):
    regression = discover_workflows(data, target="label", features=["signal"])
    assert "requires a numeric target" in option(regression, "linear-regression").reason
    classification = discover_workflows(data, target="outcome", features=["signal"])
    assert not option(classification, "logistic-regression").available
    assert "looks continuous" in option(classification, "logistic-regression").reason
    valid = discover_workflows(data, target="label", features=["signal"])
    assert option(valid, "logistic-regression").available
    assert option(valid, "decision-tree", "classification").available


def test_split_feasibility_and_missing_targets_are_checked():
    small = Dataset(pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4], "y": ["a", "b", "c", "c"]}))
    result = discover_workflows(small, target="y", features=["x"])
    assert not option(result, "logistic-regression").available
    assert "Each class needs at least 2 rows" in option(result, "logistic-regression").reason
    uneven = Dataset(pd.DataFrame({"x": np.linspace(0, 1, 6), "y": ["a", "a", "b", "b", "c", "c"]}))
    result = discover_workflows(uneven, target="y", features=["x"])
    assert not option(result, "logistic-regression").available
    assert "test_size" in option(result, "logistic-regression").reason
    missing = Dataset(pd.DataFrame({"x": np.linspace(0, 1, 10), "y": [0.1, 0.2, *([np.nan] * 8)]}))
    result = discover_workflows(missing, target="y", features=["x"])
    assert not option(result, "linear-regression").available
    assert "training" in option(result, "linear-regression").reason


@pytest.mark.parametrize("context", [{"group_column": "group"}, {"time_column": "row_id"}])
def test_declared_dependence_blocks_rowwise_prediction_and_tests_but_allows_description(data, context):
    result = discover_workflows(data, target="outcome", features=["signal"], **context)
    for candidate in result.workflows:
        if candidate.intent != "structure":
            assert not candidate.available
            assert "grouped or temporal evaluator" in candidate.reason
    assert option(result, "pca").available
    assert result.warnings


def test_declared_context_columns_cannot_leak_into_features(data):
    result = discover_workflows(data, features=["signal", "group"], group_column="group")
    assert not option(result, "pca").available
    assert "Exclude the declared" in option(result, "pca").reason


def test_unknown_columns_duplicate_features_and_target_leakage_are_explicit(data):
    with pytest.raises(ValueError, match="Unknown target"):
        discover_workflows(data, target="not here")
    with pytest.raises(ValueError, match="Unknown group"):
        discover_workflows(data, group_column="not here")
    result = discover_workflows(data, target="outcome", features=["outcome"])
    assert "cannot also be a feature" in option(result, "linear-regression").reason
    result = discover_workflows(data, features=["signal", "signal"])
    assert not option(result, "pca").available
    assert "unique" in option(result, "pca").reason


def test_empty_constant_and_nonoverlapping_observations_have_reasons():
    empty = discover_workflows(Dataset(pd.DataFrame({"x": pd.Series(dtype=float)})))
    assert all(not candidate.available for candidate in empty.workflows)
    constant = discover_workflows(Dataset(pd.DataFrame({"a": [1.0] * 10, "b": [2.0] * 10})), features=["a", "b"])
    assert not option(constant, "pca").available
    assert not option(constant, "association-scan").available
    disjoint = Dataset(pd.DataFrame({"a": [0.1, 0.2, 0.3, *([np.nan] * 3)],
                                    "b": [*([np.nan] * 3), 1.1, 1.2, 1.3]}))
    assert "No selected pair" in option(discover_workflows(disjoint), "association-scan").reason


def test_small_structure_dataset_distinguishes_default_cluster_requirement():
    data = Dataset(pd.DataFrame({"x": [0.1, 0.2, 0.3]}))
    result = discover_workflows(data)
    assert not option(result, "kmeans").available
    assert option(result, "pca").available


def test_explicit_source_role_is_described_without_converting_values(data):
    data.set_type("row_id", "categorical")
    result = discover_workflows(data)
    column = next(column for column in result.columns if column.name == "row_id")
    assert column.role == "categorical" and "Explicit source role" in column.reason
    assert data.df.row_id.dtype.kind in "iu"
