"""Open typed compositions, positional lineage, target-descendant guards and full artifacts."""
import json
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.fingerprint import dataset_fingerprint
from mlcookbook.kitchen import (
    KitchenDocument,
    KitchenEdge,
    KitchenGraph,
    KitchenNode,
    default_settings,
    node_states,
    run_graph,
)
from mlcookbook.kitchen_artifacts import VectorArtifact
from mlcookbook.kitchen_io import load_kitchen, save_kitchen


def node(key, kind, **settings):
    return KitchenNode(key, kind, default_settings(kind) | settings)


def edge(source, output, target, incoming):
    return KitchenEdge(source, output, target, incoming)


@pytest.fixture
def data():
    x, z = np.arange(1., 41.), np.linspace(.5, 2.5, 40)
    return Dataset(pd.DataFrame({"x": x, "z": z, "y": 4 * (x * z + 2) + np.sin(x)}, index=[7, 7, 3, 2] * 10))


def feature_graph():
    return KitchenGraph([
        node("a", "dataset", snapshot="own"), node("x", "column", column="x"), node("z", "column", column="z"),
        node("y", "column", column="y"), node("b", "combine", operation="multiply"),
        node("c", "map", operation="add", value=2.0), node("d", "build", name="signal"),
        node("e", "add_column", name="outcome"),
        node("f", "linear", target="outcome", features=["signal"], cv_folds=2), node("g", "report"),
        node("s", "summary"), node("r", "correlation", method="spearman"),
    ], [edge("a", "data", key, "data") for key in ("x", "z", "y")] + [
        edge("x", "vector", "b", "left"), edge("z", "vector", "b", "right"),
        edge("b", "vector", "c", "vector"), edge("c", "vector", "d", "vector"),
        edge("d", "data", "e", "data"), edge("y", "vector", "e", "vector"),
        edge("e", "data", "f", "data"), edge("f", "result", "g", "result"),
        edge("c", "vector", "s", "vector"), edge("x", "vector", "r", "left"), edge("y", "vector", "r", "right"),
    ])


def document(plan, data, execution):
    doc = KitchenDocument(plan, {"own": data}, {"own": {"name": "measurements.csv", "source": {"path": "gone.csv"}}})
    doc.accept_execution(execution)
    return doc


def test_branching_vector_function_assembles_a_real_model_and_other_results(data):
    execution = run_graph(feature_graph(), {"own": data})
    expected = data.df.x * data.df.z + 2
    table = execution.datasets["e"]
    np.testing.assert_allclose(table.df.signal.to_numpy(dtype=float), expected.to_numpy())
    pd.testing.assert_index_equal(table.df.index, data.df.index)
    np.testing.assert_allclose(table.df.outcome.to_numpy(dtype=float), data.df.y.to_numpy())
    result = execution.run.records["f"].payload
    assert result["test_metrics"]["r2"] > .999
    assert set(result["train_rows"]).isdisjoint(result["test_rows"])
    assert result["preprocessing"]["n_rows_fitted"] == len(result["train_rows"])
    assert execution.run.records["s"].payload["statistics"]["mean"] == pytest.approx(expected.mean())
    assert execution.run.records["r"].payload["statistic"] > .99
    json.dumps(execution.run.to_dict(), allow_nan=False)


def test_saved_stale_model_retains_full_older_derived_data_without_reexecution(data, tmp_path, monkeypatch):
    import mlcookbook.kitchen as kitchen

    plan = feature_graph()
    first = run_graph(plan, {"own": data})
    doc = document(plan, data, first)
    old_input = first.run.records["f"].payload["input_artifacts"]["data"]
    old_values = first.artifacts[old_input].dataset.df.copy()
    changed = replace(plan, nodes=[replace(n, settings=n.settings | {"value": 50.0}) if n.id == "c" else n for n in plan.nodes])
    doc.graph = changed
    doc.accept_execution(run_graph(changed, doc.snapshots, previous=doc.run, to_node="c"))
    path = tmp_path / "function.mlck"
    save_kitchen(path, doc)
    monkeypatch.setattr(kitchen, "linear_regression", lambda *a, **kw: pytest.fail("Load must not fit"))
    monkeypatch.setattr(kitchen, "map_vector", lambda *a, **kw: pytest.fail("Load must not execute maps"))
    loaded = load_kitchen(path)
    assert node_states(changed, {"own": dataset_fingerprint(data)}, loaded.run)["f"] == "stale"
    pd.testing.assert_frame_equal(loaded.artifacts[old_input].dataset.df, old_values)
    current_vector = loaded.artifacts[loaded.run.records["c"].payload["outputs"]["vector"]]
    assert len(current_vector.values) == len(data.df)
    np.testing.assert_allclose(current_vector.values.to_numpy(dtype=float), data.df.x * data.df.z + 50)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("kitchen.json"))
    assert manifest["version"] == 2 and manifest["artifacts"]
    assert "values" not in loaded.run.records["c"].payload
    assert len(loaded.run.records["c"].payload["vectors"]["vector"]["preview"]) == 20


def test_same_labels_from_different_snapshots_cannot_be_combined(data):
    plan = KitchenGraph([node("a", "dataset", snapshot="own"), node("b", "dataset", snapshot="other"),
                         node("x", "column", column="x"), node("z", "column", column="z"), node("sum", "combine")],
                        [edge("a", "data", "x", "data"), edge("b", "data", "z", "data"),
                         edge("x", "vector", "sum", "left"), edge("z", "vector", "sum", "right")])
    with pytest.raises(ValueError, match="Row lineage differs"):
        run_graph(plan, {"own": data, "other": data})


def test_different_equal_length_filtered_positions_cannot_align_on_duplicate_indices():
    data = Dataset(pd.DataFrame({"x": range(40)}, index=["same"] * 40))
    plan = KitchenGraph([node("a", "dataset", snapshot="own"), node("x", "column", column="x"),
                         node("low", "map", operation="less_than", value=20), node("high", "map", operation="greater_than", value=19),
                         node("l", "filter"), node("h", "filter"), node("lv", "column", column="x"),
                         node("hv", "column", column="x"), node("sum", "combine")],
        [edge("a", "data", "x", "data"), edge("x", "vector", "low", "vector"), edge("x", "vector", "high", "vector"),
         edge("a", "data", "l", "data"), edge("a", "data", "h", "data"), edge("low", "vector", "l", "mask"),
         edge("high", "vector", "h", "mask"), edge("l", "data", "lv", "data"), edge("h", "data", "hv", "data"),
         edge("lv", "vector", "sum", "left"), edge("hv", "vector", "sum", "right")])
    with pytest.raises(ValueError, match="Row lineage differs"):
        run_graph(plan, {"own": data})


def test_target_descendant_feature_is_rejected_but_ratio_target_and_ancestor_feature_are_valid(data):
    leaking = KitchenGraph([node("a", "dataset", snapshot="own"), node("y", "column", column="y"),
                            node("map", "map", operation="multiply", value=2), node("append", "add_column", name="leak"),
                            node("fit", "linear", target="y", features=["leak"], cv_folds=2)],
        [edge("a", "data", "y", "data"), edge("y", "vector", "map", "vector"),
         edge("a", "data", "append", "data"), edge("map", "vector", "append", "vector"), edge("append", "data", "fit", "data")])
    with pytest.raises(ValueError, match="derived from the selected target"):
        run_graph(leaking, {"own": data})
    ratio = KitchenGraph([node("a", "dataset", snapshot="own"), node("y", "column", column="y"),
                          node("x", "column", column="x"), node("ratio", "combine", operation="divide"),
                          node("append", "add_column", name="ratio"), node("fit", "linear", target="ratio", features=["x"], cv_folds=2)],
        [edge("a", "data", "y", "data"), edge("a", "data", "x", "data"), edge("y", "vector", "ratio", "left"),
         edge("x", "vector", "ratio", "right"), edge("a", "data", "append", "data"),
         edge("ratio", "vector", "append", "vector"), edge("append", "data", "fit", "data")])
    execution = run_graph(ratio, {"own": data})
    assert execution.run.attempts["fit"]["state"] == "completed"
    assert any("share source columns" in w for w in execution.run.records["fit"].warnings)


def test_domain_handling_preserves_rows_and_reports_invalid_values():
    data = Dataset(pd.DataFrame({"x": [-1., 0., 1., 2., None]}, index=[1, 1, 2, 2, 3]))
    plan = KitchenGraph([node("a", "dataset", snapshot="own"), node("x", "column", column="x"),
                         node("log", "map", operation="log"), node("s", "summary")],
                        [edge("a", "data", "x", "data"), edge("x", "vector", "log", "vector"), edge("log", "vector", "s", "vector")])
    with pytest.raises(ValueError, match="invalid/non-finite"):
        run_graph(plan, {"own": data})
    plan = replace(plan, nodes=[replace(n, settings=n.settings | {"invalid": "missing"}) if n.id == "log" else n for n in plan.nodes])
    run = run_graph(plan, {"own": data})
    vector = run.vectors["log", "vector"]
    assert len(vector.values) == 5 and int(vector.values.isna().sum()) == 3
    assert vector.index.equals(data.df.index)
    assert any("marked 2 invalid" in w for w in run.run.records["s"].warnings)


def test_source_port_reconnection_marks_summary_stale_and_records_consumed_port():
    data = Dataset(pd.DataFrame({"flag": [False] * 5}))
    plan = KitchenGraph([node("a", "dataset", snapshot="own"), node("col", "column", column="flag"), node("s", "summary")],
                        [edge("a", "data", "col", "data"), edge("col", "vector", "s", "vector")])
    first = run_graph(plan, {"own": data})
    changed = replace(plan, edges=[plan.edges[0], edge("col", "missing", "s", "vector")])
    assert node_states(changed, {"own": dataset_fingerprint(data)}, first.run)["s"] == "stale"
    assert first.run.records["s"].inputs["vector"]["port"] == "vector"
    second = run_graph(changed, {"own": data}, previous=first.run)
    assert second.run.records["s"].inputs["vector"]["port"] == "missing"
    assert second.run.records["s"].payload["name"] == "flag missing"


def test_categorical_summary_counts_observed_values_and_persists_domains(tmp_path):
    data = Dataset(pd.DataFrame({"label": pd.Categorical(["a", "a", None], categories=["a", "b"])}))
    plan = KitchenGraph([node("a", "dataset", snapshot="own"), node("c", "column", column="label"), node("s", "summary")],
                        [edge("a", "data", "c", "data"), edge("c", "vector", "s", "vector")])
    execution = run_graph(plan, {"own": data})
    assert execution.run.records["s"].payload["n_unique"] == 1
    assert execution.run.records["s"].payload["value_counts"] == [["a", 2]]
    path = tmp_path / "categories.mlck"
    save_kitchen(path, document(plan, data, execution))
    loaded = load_kitchen(path)
    key = loaded.run.records["c"].payload["outputs"]["vector"]
    vector = loaded.artifacts[key]
    assert isinstance(vector, VectorArtifact)
    assert list(vector.values.cat.categories) == ["a", "b"] and vector.values.isna().sum() == 1


def transform_graph():
    return KitchenGraph([node("a", "dataset", snapshot="own"), node("c", "column", column="x"),
                         node("m", "map", operation="add", value=1.0), node("n", "map", operation="square"),
                         node("t", "build", name="derived")],
                        [edge("a", "data", "c", "data"), edge("c", "vector", "m", "vector"),
                         edge("m", "vector", "n", "vector"), edge("n", "vector", "t", "vector")])


@pytest.mark.parametrize("when", ["before_first", "during_map"])
def test_preflight_cancellation_stops_between_kernels_without_exposing_unexecuted_outputs(data, monkeypatch, when):
    import mlcookbook.kitchen as kitchen

    plan = transform_graph()
    previous = run_graph(plan, {"own": data})
    calls, stop = [], [when == "before_first"]
    original_map = kitchen.map_vector

    def mapped(*args, **kwargs):
        calls.append("map")
        output = original_map(*args, **kwargs)
        stop[0] = True
        return output

    monkeypatch.setattr(kitchen, "map_vector", mapped)
    if when == "before_first":
        monkeypatch.setattr(kitchen, "root_table", lambda *a, **kw: pytest.fail("Cancelled validation must not copy source data"))
    cancelled = run_graph(plan, {"own": data}, previous=previous.run, stop_requested=lambda: stop[0])
    assert calls == ([] if when == "before_first" else ["map"])
    assert cancelled.run.records == previous.run.records
    assert cancelled.run.execution_order == []
    assert {a["state"] for a in cancelled.run.attempts.values()} == {"cancelled"}
    assert not cancelled.datasets and not cancelled.vectors and not cancelled.artifacts
    assert "input validation" in cancelled.run.warnings[0]


def test_cancelled_execution_retains_stale_outputs_and_only_returns_completed_artifacts(data, tmp_path):
    plan = transform_graph()
    first = run_graph(plan, {"own": data})
    doc = document(plan, data, first)
    changed = replace(plan, nodes=[replace(n, settings=n.settings | {"value": 20.0}) if n.id == "m" else n for n in plan.nodes])
    events = []
    stopped = run_graph(changed, {"own": data}, previous=first.run, on_progress=events.append,
                        stop_requested=lambda: any(e["node"] == "c" and e["state"] == "completed" for e in events))
    assert stopped.run.execution_order == ["a", "c"]
    assert set(stopped.datasets) == {"a"}
    assert set(stopped.vectors) == {("c", "vector"), ("c", "missing")}
    assert stopped.run.records["m"] == first.run.records["m"]
    assert stopped.run.attempts["m"]["state"] == "cancelled"
    doc.graph = changed
    doc.accept_execution(stopped)
    path = tmp_path / "cancelled.mlck"
    save_kitchen(path, doc)
    loaded = load_kitchen(path)
    assert node_states(changed, {"own": dataset_fingerprint(data)}, loaded.run)["m"] == "cancelled"
    old_id = first.run.records["m"].payload["outputs"]["vector"]
    np.testing.assert_array_equal(loaded.artifacts[old_id].values.to_numpy(dtype=float), data.df.x.to_numpy() + 1)
    assert loaded.run.records["m"].settings["value"] == 1.0


def test_numeric_summary_overflow_is_explicit_json_safe_and_persisted(tmp_path):
    import warnings

    data = Dataset(pd.DataFrame({"x": [1e308, 1e308]}))
    plan = KitchenGraph([node("a", "dataset", snapshot="own"), node("c", "column", column="x"), node("s", "summary")],
                        [edge("a", "data", "c", "data"), edge("c", "vector", "s", "vector")])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        execution = run_graph(plan, {"own": data})
    record = execution.run.records["s"]
    assert record.payload["statistics"]["mean"] is None
    assert record.payload["statistics"]["minimum"] == 1e308
    assert any("Summary mean" in w and "non-finite" in w for w in record.warnings)
    assert all(value is None or np.isfinite(value) for value in record.payload["statistics"].values())
    json.dumps(record.payload, allow_nan=False)
    path = tmp_path / "overflow.mlck"
    save_kitchen(path, document(plan, data, execution))
    loaded = load_kitchen(path)
    assert loaded.run.records["s"] == record


@pytest.mark.parametrize("values, message", [([2.0], "requires at least two"), ([None], "No non-missing numeric values")])
def test_numeric_summary_undefined_metrics_have_result_warnings(values, message):
    data = Dataset(pd.DataFrame({"x": pd.Series(values, dtype="Float64")}), overrides={"x": "numeric"})
    plan = KitchenGraph([node("a", "dataset", snapshot="own"), node("c", "column", column="x"), node("s", "summary")],
                        [edge("a", "data", "c", "data"), edge("c", "vector", "s", "vector")])
    execution = run_graph(plan, {"own": data})
    assert any(message in w for w in execution.run.records["s"].warnings)
    json.dumps(execution.run.to_dict(), allow_nan=False)


def test_stale_filter_records_consumed_port_and_full_original_rows_after_reconnection(tmp_path):
    data = Dataset(pd.DataFrame({"flag": pd.array([True, False, None], dtype="boolean"), "x": [1., 2., 3.]}, index=[7, 7, 7]))
    plan = KitchenGraph([node("a", "dataset", snapshot="own"), node("c", "column", column="flag"), node("f", "filter")],
                        [edge("a", "data", "c", "data"), edge("a", "data", "f", "data"), edge("c", "vector", "f", "mask")])
    first = run_graph(plan, {"own": data})
    doc = document(plan, data, first)
    doc.graph = replace(plan, edges=[*plan.edges[:-1], edge("c", "missing", "f", "mask")])
    path = tmp_path / "old-filter.mlck"
    save_kitchen(path, doc)
    loaded = load_kitchen(path)
    assert node_states(loaded.graph, {"own": dataset_fingerprint(data)}, loaded.run)["f"] == "stale"
    old_record = loaded.run.records["f"]
    assert old_record.inputs["mask"]["port"] == "vector"
    assert old_record.inputs["mask"]["artifact"] == old_record.payload["input_artifacts"]["mask"]
    assert loaded.artifacts[old_record.payload["outputs"]["data"]].rows[0][1] == 0
    current = run_graph(loaded.graph, loaded.snapshots, previous=loaded.run)
    assert current.run.records["f"].inputs["mask"]["port"] == "missing"
    assert current.datasets["f"].df.x.tolist() == [3.0]
    assert current.artifacts[current.run.records["f"].payload["outputs"]["data"]].rows[0][1] == 2


def test_target_based_row_filter_is_rejected_before_fitting(data, monkeypatch):
    import mlcookbook.kitchen as kitchen

    plan = KitchenGraph([node("a", "dataset", snapshot="own"), node("y", "column", column="y"),
                         node("mask", "map", operation="greater_than", value=20.), node("filtered", "filter"),
                         node("fit", "linear", target="y", features=["x"], cv_folds=2)],
                        [edge("a", "data", "y", "data"), edge("y", "vector", "mask", "vector"),
                         edge("a", "data", "filtered", "data"), edge("mask", "vector", "filtered", "mask"),
                         edge("filtered", "data", "fit", "data")])
    monkeypatch.setattr(kitchen, "linear_regression", lambda *a, **kw: pytest.fail("Target-dependent row selection must precede fitting validation"))
    with pytest.raises(ValueError, match="Row selection depends on the selected target"):
        run_graph(plan, {"own": data})


def test_foundational_v1_read_and_resave_preserve_offline_results_without_execution(data, tmp_path, monkeypatch):
    import mlcookbook.kitchen as kitchen

    plan = KitchenGraph([node("a", "dataset", snapshot="own"), node("b", "select", columns=["x", "y"])],
                        [edge("a", "data", "b", "data")])
    path = tmp_path / "foundation.mlck"
    save_kitchen(path, document(plan, data, run_graph(plan, {"own": data})))
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(contents["kitchen.json"])
    manifest["version"] = 1
    manifest.pop("artifacts")
    for record in manifest["run"]["records"].values():
        for field in ("outputs", "input_artifacts", "source_snapshot_ids", "dependencies"):
            record["payload"].pop(field, None)
        record["inputs"] = {port: value["fingerprint"] for port, value in record["inputs"].items()}
    members = {entry["member"] for entry in manifest["snapshots"].values()}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("kitchen.json", json.dumps(manifest))
        for member in members:
            archive.writestr(member, contents[member])
    monkeypatch.setattr(kitchen, "root_table", lambda *a, **kw: pytest.fail("Open/resave must not execute"))
    loaded = load_kitchen(path)
    pd.testing.assert_frame_equal(loaded.snapshots["own"].df, data.df)
    assert loaded.run.to_dict() == manifest["run"]
    assert loaded.artifacts == {} and any("preview-only" in w for w in loaded.warnings)
    resaved = tmp_path / "preserved-v2.mlck"
    save_kitchen(resaved, loaded)
    assert load_kitchen(resaved).run == loaded.run


@pytest.mark.parametrize("damage", ["artifact_checksum", "input_port", "input_artifact"])
def test_full_artifacts_and_historical_input_provenance_are_verified(data, tmp_path, damage):
    plan = transform_graph()
    execution = run_graph(plan, {"own": data})
    path = tmp_path / "valid.mlck"
    save_kitchen(path, document(plan, data, execution))
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(contents["kitchen.json"])
    if damage == "artifact_checksum":
        key = execution.run.records["m"].payload["outputs"]["vector"]
        contents[manifest["artifacts"][key]["member"]] = b"damaged typed output"
    elif damage == "input_port":
        manifest["run"]["records"]["m"]["inputs"]["vector"]["port"] = None
    else:
        manifest["run"]["records"]["m"]["inputs"]["vector"]["artifact"] = "0" * 64
    contents["kitchen.json"] = json.dumps(manifest).encode()
    broken = tmp_path / "damaged.mlck"
    with zipfile.ZipFile(broken, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError):
        load_kitchen(broken)
