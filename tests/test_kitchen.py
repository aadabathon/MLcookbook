"""Typed graph execution keeps existing statistical boundaries and old artifacts."""
import hashlib
import json
import zipfile
from dataclasses import replace

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
    graph_fingerprints,
    graph_from_dict,
    node_states,
    run_graph,
)
from mlcookbook.kitchen_io import load_kitchen, save_kitchen


@pytest.fixture
def data():
    return Dataset(pd.DataFrame({"id": [f"{i:03}" for i in range(40)], "x": range(40),
                                 "y": [3 * i + i % 3 for i in range(40)]}, index=[4, 4, 2, 9] * 10),
                   overrides={"id": "categorical"})


def graph():
    settings = default_settings("comparison") | {"target": "y", "features": ["x"], "n_estimators": 3, "cv_folds": 2}
    return KitchenGraph([
        KitchenNode("source", "dataset", {"snapshot": "own"}),
        KitchenNode("select", "select", {"columns": ["x", "y"], "roles": {}}),
        KitchenNode("compare", "comparison", settings),
        KitchenNode("report", "report", {}),
    ], [KitchenEdge("source", "data", "select", "data"), KitchenEdge("select", "data", "compare", "data"),
        KitchenEdge("compare", "result", "report", "result")])


def test_real_graph_preserves_lineage_shared_folds_and_serializes(data):
    execution = run_graph(graph(), {"own": data})
    json.dumps(execution.run.to_dict(), allow_nan=False)
    selected = execution.datasets["select"]
    pd.testing.assert_frame_equal(selected.df, data.df[["x", "y"]])
    comparison = execution.run.records["compare"].payload
    assert len(comparison["folds"]) == 2
    assert set(comparison["train_rows"]).isdisjoint(comparison["test_rows"])
    assert set(comparison["test_rows"]) | set(comparison["train_rows"]) == set(range(40))
    assert comparison["preprocessing"]["n_rows_fitted"] == len(comparison["train_rows"])
    for fold in comparison["folds"]:
        assert set(fold["train_rows"]).isdisjoint(fold["validation_rows"])
        assert set(fold["train_rows"] + fold["validation_rows"]) == set(comparison["train_rows"])
    assert execution.run.records["report"].payload == comparison
    assert execution.run.execution_order == ["source", "select", "compare", "report"]


@pytest.mark.parametrize("problem", ["cycle", "port", "double", "missing", "settings", "target", "column"])
def test_invalid_graph_is_rejected_before_any_model_call(data, monkeypatch, problem):
    import mlcookbook.kitchen as kitchen

    document = graph().to_dict()
    if problem == "cycle":
        document["edges"].append(KitchenEdge("select", "data", "select", "data").to_dict())
        document["edges"].pop(0)
    elif problem == "port":
        document["edges"][-1]["source_port"] = "data"
    elif problem == "double":
        document["edges"].append(document["edges"][0])
    elif problem == "missing":
        document["edges"].pop(1)
    elif problem == "settings":
        document["nodes"][2]["settings"]["n_estimators"] = True
    elif problem == "target":
        document["nodes"][2]["settings"]["target"] = "absent"
    else:
        document["nodes"][1]["settings"]["columns"] = ["unknown"]
    monkeypatch.setattr(kitchen, "compare_regressors", lambda *a, **kw: pytest.fail("Validation must precede fitting"))
    with pytest.raises(ValueError):
        run_graph(graph_from_dict(document), {"own": data})


def test_run_to_selection_ignores_unconfigured_other_branch_and_fits_once(data, monkeypatch):
    import mlcookbook.kitchen as kitchen

    original, calls = kitchen.compare_regressors, []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(kitchen, "compare_regressors", counted)
    plan = graph()
    plan = replace(plan, nodes=[*plan.nodes, KitchenNode("other", "comparison", default_settings("comparison"))])
    execution = run_graph(plan, {"own": data}, to_node="compare")
    assert len(calls) == 1 and "report" not in execution.run.records and "other" not in execution.run.records
    assert execution.run.execution_order == ["source", "select", "compare"]


def test_changes_mark_only_descendants_stale_and_revert_restores_current(data):
    plan = graph()
    run = run_graph(plan, {"own": data}).run
    hashes = {"own": dataset_fingerprint(data)}
    changed = replace(plan, nodes=[replace(n, settings=n.settings | {"alpha": 3.0}) if n.id == "compare" else n for n in plan.nodes])
    assert node_states(changed, hashes, run) == {"source": "completed", "select": "completed", "compare": "stale", "report": "stale"}
    assert set(node_states(plan, hashes, run).values()) == {"completed"}
    data.df.iloc[0, 1] = 999
    assert set(node_states(plan, {"own": dataset_fingerprint(data)}, run).values()) == {"stale"}
    assert run.records["compare"].settings["alpha"] == 1.0


def test_failure_and_cancellation_retain_completed_results_and_skip_descendants(data, monkeypatch):
    import mlcookbook.kitchen as kitchen

    plan = graph()
    previous = run_graph(plan, {"own": data}).run
    monkeypatch.setattr(kitchen, "compare_regressors", lambda *a, **kw: (_ for _ in ()).throw(ValueError("local fitting failed")))
    failed = run_graph(plan, {"own": data}, previous=previous).run
    assert failed.records["compare"] == previous.records["compare"]
    assert failed.attempts["compare"]["state"] == "failed" and failed.attempts["report"]["state"] == "blocked"
    assert "local fitting failed" in failed.attempts["compare"]["error"]
    states = []
    cancelled = run_graph(plan, {"own": data}, previous=previous,
                          stop_requested=lambda: any(s["node"] == "select" and s["state"] == "completed" for s in states),
                          on_progress=states.append).run
    assert cancelled.attempts["compare"]["state"] == "cancelled"
    assert cancelled.records["compare"] == previous.records["compare"]
    assert cancelled.execution_order == ["source", "select"]


def test_saved_stale_graph_reopens_without_sources_or_fitting(data, tmp_path, monkeypatch):
    import mlcookbook.kitchen as kitchen

    plan = graph()
    execution = run_graph(plan, {"own": data})
    run = execution.run
    changed = replace(plan, nodes=[replace(n, settings=n.settings | {"alpha": 5.0}) if n.id == "compare" else n for n in plan.nodes])
    doc = KitchenDocument(changed, {"own": data}, {"own": {"name": "my.csv", "source": {"path": "missing.csv"}}},
                           run, {"source": [10.0, -20.0]}, "report", artifacts=execution.artifacts)
    path = tmp_path / "research.mlck"
    save_kitchen(path, doc)
    monkeypatch.setattr(kitchen, "compare_regressors", lambda *a, **kw: pytest.fail("Open must not fit"))
    loaded = load_kitchen(path)
    pd.testing.assert_frame_equal(loaded.snapshots["own"].df, data.df)
    assert loaded.run == run and loaded.graph == changed and loaded.layout == doc.layout
    assert loaded.selected == "report"
    assert node_states(loaded.graph, {"own": dataset_fingerprint(loaded.snapshots["own"])}, loaded.run)["report"] == "stale"


def test_atomic_save_failure_preserves_existing_document(data, tmp_path, monkeypatch):
    import mlcookbook.kitchen_io as storage

    path = tmp_path / "work.mlck"
    doc = KitchenDocument(graph(), {"own": data}, {"own": {"name": "own", "source": {}}})
    save_kitchen(path, doc)
    before = path.read_bytes()
    monkeypatch.setattr(storage.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        save_kitchen(path, doc)
    assert path.read_bytes() == before and list(tmp_path.glob("*.tmp")) == []


def test_fingerprints_ignore_layout_but_include_roles_and_indices(data):
    copied = Dataset(data.df, overrides={"id": "text"})
    assert dataset_fingerprint(copied) != dataset_fingerprint(data)
    reordered = Dataset(data.df.iloc[::-1], overrides={"id": "categorical"})
    assert dataset_fingerprint(reordered) != dataset_fingerprint(data)
    hashes = {"own": dataset_fingerprint(data)}
    assert graph_fingerprints(graph_from_dict(json.loads(json.dumps(graph().to_dict()))), hashes) == graph_fingerprints(graph(), hashes)


def test_selection_changes_roles_without_converting_values_or_reordering_rows(data):
    plan = graph()
    changed = replace(plan, nodes=[replace(n, settings={"columns": ["y", "id", "x"], "roles": {"x": "text"}})
                                   if n.id == "select" else n for n in plan.nodes])
    output = run_graph(changed, {"own": data}, to_node="select").datasets["select"]
    pd.testing.assert_frame_equal(output.df, data.df[["y", "id", "x"]])
    assert output.schema().var_type("x").value == "text"
    assert output.schema().var_type("id").value == "categorical"


def test_shared_fingerprint_remains_compatible_with_legacy_llm_saved_runs(data):
    payload = {"parquet_sha256": hashlib.sha256(data.df.to_parquet(index=True, engine="pyarrow")).hexdigest(),
               "roles": {c.name: c.var_type.value for c in data.schema().columns}}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert dataset_fingerprint(data) == expected


@pytest.mark.parametrize("damage", ["checksum", "roles", "version", "snapshot", "selection", "nonfinite", "record"])
def test_damaged_documents_are_rejected_without_execution(data, tmp_path, monkeypatch, damage):
    import mlcookbook.kitchen as kitchen

    path = tmp_path / "original.mlck"
    doc = KitchenDocument(graph(), {"own": data}, {"own": {"name": "own.csv", "source": {}}})
    doc.accept_execution(run_graph(graph(), {"own": data}))
    save_kitchen(path, doc)
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(contents["kitchen.json"])
    if damage == "checksum":
        contents[manifest["snapshots"]["own"]["member"]] = b"broken parquet"
    elif damage == "roles":
        manifest["snapshots"]["own"]["roles"].pop("x")
    elif damage == "version":
        manifest["version"] = True
    elif damage == "snapshot":
        manifest["run"]["records"]["report"]["payload"]["snapshot_id"] = "missing"
    elif damage == "selection":
        manifest["selected"] = "absent"
    elif damage == "nonfinite":
        manifest["layout"] = {"source": [float("inf"), 0]}
    else:
        manifest["run"]["records"]["report"]["fingerprint"] = "invalid"
    contents["kitchen.json"] = json.dumps(manifest).encode()
    broken = tmp_path / "broken.mlck"
    with zipfile.ZipFile(broken, "w") as archive:
        for name, blob in contents.items():
            archive.writestr(name, blob)
    monkeypatch.setattr(kitchen, "compare_regressors", lambda *a, **kw: pytest.fail("Load must not execute"))
    with pytest.raises(ValueError):
        load_kitchen(broken)


@pytest.mark.parametrize("cancel", [False, True])
def test_latest_failed_or_cancelled_attempt_survives_reopen_beside_older_result(data, tmp_path, monkeypatch, cancel):
    import mlcookbook.kitchen as kitchen

    original = run_graph(graph(), {"own": data})
    previous = original.run
    changed = replace(graph(), nodes=[replace(n, settings=n.settings | {"alpha": 4.0}) if n.id == "compare" else n
                                      for n in graph().nodes])
    progress = []
    monkeypatch.setattr(kitchen, "compare_regressors", lambda *a, **kw: (_ for _ in ()).throw(ValueError("fit failed")))
    run = run_graph(changed, {"own": data}, previous=previous, on_progress=progress.append,
                    stop_requested=lambda: cancel and any(e["node"] == "select" and e["state"] == "completed" for e in progress)).run
    expected = "cancelled" if cancel else "failed"
    assert node_states(changed, {"own": dataset_fingerprint(data)}, run)["compare"] == expected
    path = tmp_path / "attempt.mlck"
    save_kitchen(path, KitchenDocument(changed, {"own": data}, {"own": {"name": "own", "source": {}}}, run, artifacts=original.artifacts))
    loaded = load_kitchen(path)
    assert node_states(loaded.graph, {"own": dataset_fingerprint(loaded.snapshots["own"])}, loaded.run)["compare"] == expected
    assert loaded.run.records["compare"] == previous.records["compare"]
    assert loaded.run.records["compare"].settings["alpha"] == 1.0
