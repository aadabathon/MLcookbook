"""The Kitchen: a small typed DAG of existing, auditable operations.

No Qt, arbitrary code, fitted-estimator artifacts or graph-level fitted transforms.
Dataset objects are runtime artifacts; completed records contain JSON-safe data.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings as python_warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from mlcookbook.comparison import compare_regressors
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.fingerprint import dataset_fingerprint
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType
from mlcookbook.experiments import runtime_versions
from mlcookbook.kitchen_artifacts import (
    COMBINATIONS,
    MAPS,
    PreparedKitchen,
    TableArtifact,
    VectorArtifact,
    add_column,
    aligned,
    artifact_roots,
    build_dataset,
    column_vectors,
    combine_vectors,
    filter_rows,
    map_vector,
    numeric_values,
    root_table,
    select_columns,
    validate_target_dependencies,
)
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.regression._problem import prepare_regression_problem
from mlcookbook.regression.linear import linear_regression
from mlcookbook.statistics.association import correlation

PORTS = {
    "dataset": ({}, {"data": "dataset"}),
    "select": ({"data": "dataset"}, {"data": "dataset"}),
    "column": ({"data": "dataset"}, {"vector": "vector", "missing": "vector"}),
    "map": ({"vector": "vector"}, {"vector": "vector"}),
    "combine": ({"left": "vector", "right": "vector"}, {"vector": "vector"}),
    "build": ({"vector": "vector"}, {"data": "dataset"}),
    "add_column": ({"data": "dataset", "vector": "vector"}, {"data": "dataset"}),
    "filter": ({"data": "dataset", "mask": "vector"}, {"data": "dataset"}),
    "summary": ({"vector": "vector"}, {"result": "analysis"}),
    "correlation": ({"left": "vector", "right": "vector"}, {"result": "analysis"}),
    "linear": ({"data": "dataset"}, {"result": "analysis"}),
    "comparison": ({"data": "dataset"}, {"result": "analysis"}),
    "report": ({"result": "analysis"}, {}),
}


def default_settings(kind: str) -> dict:
    settings = {
        "dataset": {"snapshot": ""},
        "select": {"columns": [], "roles": {}},
        "column": {"column": ""},
        "map": {"operation": "add", "value": 0.0, "invalid": "error"},
        "combine": {"operation": "add", "invalid": "error"},
        "build": {"name": "derived"}, "add_column": {"name": "derived"}, "filter": {}, "summary": {},
        "correlation": {"method": "pearson", "confidence_level": 0.95},
        "linear": {"target": "", "features": [], "penalty": "none", "alpha": 1.0, "test_size": 0.25,
                   "random_state": 0, "cv_folds": 5, "preprocessing": PreprocessingConfig().to_dict()},
        "comparison": {"target": "", "features": [], "alpha": 1.0, "n_estimators": 100,
                       "max_depth": None, "min_samples_leaf": 5, "test_size": 0.25,
                       "random_state": 0, "cv_folds": 5, "preprocessing": PreprocessingConfig().to_dict()},
        "report": {},
    }
    if kind not in settings:
        raise ValueError(f"Unknown Kitchen operation: {kind!r}.")
    return deepcopy(settings[kind])


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class KitchenNode(Result):
    id: str
    kind: str
    settings: dict


@dataclass(frozen=True)
class KitchenEdge(Result):
    source: str
    source_port: str
    target: str
    target_port: str


@dataclass(frozen=True)
class KitchenGraph(Result):
    nodes: list[KitchenNode] = field(default_factory=list)
    edges: list[KitchenEdge] = field(default_factory=list)
    version: int = 1


@dataclass(frozen=True)
class KitchenRecord(Result):
    node_id: str
    kind: str
    fingerprint: str
    settings: dict
    inputs: dict[str, str | dict]  # strings are legacy v1 records
    payload: dict
    completed_utc: str
    versions: dict[str, str]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KitchenRun(Result):
    records: dict[str, KitchenRecord] = field(default_factory=dict)
    attempts: dict[str, dict] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class KitchenExecution:
    """Runtime datasets deliberately sit outside the serializable result."""
    run: KitchenRun
    datasets: dict[str, Dataset]
    requested_nodes: list[str]
    artifacts: dict[str, TableArtifact | VectorArtifact] = field(default_factory=dict)
    vectors: dict[tuple[str, str], VectorArtifact] = field(default_factory=dict)


@dataclass
class KitchenDocument:
    graph: KitchenGraph = field(default_factory=KitchenGraph)
    snapshots: dict[str, Dataset] = field(default_factory=dict)
    snapshot_info: dict[str, dict] = field(default_factory=dict)
    run: KitchenRun = field(default_factory=KitchenRun)
    layout: dict[str, list[float]] = field(default_factory=dict)
    selected: str | None = None
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, TableArtifact | VectorArtifact] = field(default_factory=dict)

    def accept_execution(self, execution: KitchenExecution):
        """Retain complete typed outputs, including older artifacts used by stale records."""
        self.run = execution.run
        self.artifacts.update(execution.artifacts)


def graph_from_dict(value: dict) -> KitchenGraph:
    try:
        if not isinstance(value, dict) or set(value) != {"nodes", "edges", "version"}:
            raise ValueError("Expected a Kitchen graph with nodes, edges and version.")
        graph = KitchenGraph([KitchenNode(**node) for node in value["nodes"]],
                             [KitchenEdge(**edge) for edge in value["edges"]], value["version"])
        graph_order(graph, complete=False)
        return graph
    except (TypeError, KeyError) as exc:
        raise ValueError("Malformed Kitchen graph.") from exc


def _settings(node: KitchenNode):
    if not isinstance(node.kind, str):
        raise ValueError("Kitchen operation kinds must be strings.")
    defaults = default_settings(node.kind)
    if not isinstance(node.settings, dict) or set(node.settings) != set(defaults):
        raise ValueError(f"{node.id}: settings do not match {node.kind}.")
    # Reject non-JSON settings, including NaN, without coercing them.
    _hash(node.settings)
    s = node.settings
    if node.kind == "dataset" and not isinstance(s["snapshot"], str):
        raise ValueError(f"{node.id}: snapshot must be an identifier.")
    if node.kind == "select":
        if not isinstance(s["columns"], list) or not all(isinstance(c, str) for c in s["columns"]) \
                or len(set(s["columns"])) != len(s["columns"]):
            raise ValueError(f"{node.id}: select unique column names.")
        if not isinstance(s["roles"], dict) or not set(s["roles"]) <= set(s["columns"]):
            raise ValueError(f"{node.id}: role overrides must refer to selected columns.")
        for role in s["roles"].values():
            VarType(role)
    if node.kind == "column" and not isinstance(s["column"], str):
        raise ValueError(f"{node.id}: choose a column name.")
    if node.kind in ("map", "combine"):
        allowed = MAPS if node.kind == "map" else COMBINATIONS
        if s["operation"] not in allowed or s["invalid"] not in ("error", "missing"):
            raise ValueError(f"{node.id}: unknown numeric operation or invalid-value policy.")
        if node.kind == "map" and (type(s["value"]) not in (int, float) or not math.isfinite(s["value"])):
            raise ValueError(f"{node.id}: the scalar must be finite.")
    if node.kind in ("build", "add_column") and (not isinstance(s["name"], str) or len(s["name"]) > 200):
        raise ValueError(f"{node.id}: choose a nonempty output column name (at most 200 characters).")
    if node.kind == "correlation" and (s["method"] not in ("pearson", "spearman") or
            type(s["confidence_level"]) not in (float, int) or not 0 < s["confidence_level"] < 1):
        raise ValueError(f"{node.id}: choose Pearson/Spearman and a confidence level between 0 and 1.")
    if node.kind in ("comparison", "linear"):
        if not isinstance(s["target"], str) or not isinstance(s["features"], list) \
                or not all(isinstance(c, str) for c in s["features"]) or len(set(s["features"])) != len(s["features"]):
            raise ValueError(f"{node.id}: target/features must be unique column names.")
        integers = [("random_state", 0, 2**32 - 1), ("cv_folds", 2, 20)]
        if node.kind == "comparison":
            integers += [("n_estimators", 1, 1000), ("min_samples_leaf", 1, 1_000_000)]
        elif s["penalty"] not in ("none", "ridge", "lasso"):
            raise ValueError(f"{node.id}: choose OLS, ridge or lasso.")
        for key, low, high in integers:
            if type(s[key]) is not int or not low <= s[key] <= high:
                raise ValueError(f"{node.id}: {key} must be an integer from {low} to {high}.")
        if node.kind == "comparison" and s["max_depth"] is not None and (type(s["max_depth"]) is not int or not 1 <= s["max_depth"] <= 100):
            raise ValueError(f"{node.id}: max_depth must be 1–100 or unlimited.")
        for key in ("alpha", "test_size"):
            if type(s[key]) not in (int, float) or not math.isfinite(s[key]) or s[key] <= 0:
                raise ValueError(f"{node.id}: {key} must be finite and positive.")
        if s["test_size"] >= 1:
            raise ValueError(f"{node.id}: test_size must be below 1.")
        if not isinstance(s["preprocessing"], dict) or set(s["preprocessing"]) != set(PreprocessingConfig().to_dict()):
            raise ValueError(f"{node.id}: invalid preprocessing configuration.")
        if type(s["preprocessing"]["missing_indicators"]) is not bool:
            raise ValueError(f"{node.id}: missing_indicators must be boolean.")
        PreprocessingConfig(**s["preprocessing"])


def graph_order(graph: KitchenGraph, *, complete=True, to_node: str | None = None) -> list[str]:
    """Validate topology/types first; return deterministic ancestor/topological order."""
    if type(graph.version) is not int or graph.version != 1:
        raise ValueError("Unsupported Kitchen graph version.")
    if not isinstance(graph.nodes, list) or not isinstance(graph.edges, list) \
            or any(not isinstance(n, KitchenNode) or not isinstance(n.id, str) or not n.id or len(n.id) > 80 for n in graph.nodes):
        raise ValueError("Kitchen node IDs must be unique nonempty strings (at most 80 characters).")
    nodes = {n.id: n for n in graph.nodes}
    if len(nodes) != len(graph.nodes):
        raise ValueError("Kitchen node IDs must be unique.")
    if len(nodes) > 100 or len(graph.edges) > 200:
        raise ValueError("This Kitchen supports at most 100 nodes and 200 connections.")
    for node in graph.nodes:
        _settings(node)
    parents = {key: [] for key in nodes}
    children = {key: [] for key in nodes}
    ports = set()
    for edge in graph.edges:
        if not isinstance(edge, KitchenEdge) or any(not isinstance(value, str) for value in
                (edge.source, edge.target, edge.source_port, edge.target_port)):
            raise ValueError("Connection IDs and ports must be strings.")
        if edge.source not in nodes or edge.target not in nodes:
            raise ValueError("A connection refers to a missing node.")
        outputs, inputs = PORTS[nodes[edge.source].kind][1], PORTS[nodes[edge.target].kind][0]
        if edge.source_port not in outputs or edge.target_port not in inputs:
            raise ValueError("A connection refers to an unknown port.")
        if outputs[edge.source_port] != inputs[edge.target_port]:
            raise ValueError("Incompatible ports: connect datasets to datasets, comparison results to reports.")
        port = edge.target, edge.target_port
        if port in ports:
            raise ValueError(f"{edge.target}: each input accepts only one connection.")
        ports.add(port)
        parents[edge.target].append(edge.source)
        children[edge.source].append(edge.target)
    indegree = {key: len(value) for key, value in parents.items()}
    ready = sorted(key for key, count in indegree.items() if not count)
    order = []
    while ready:
        key = ready.pop(0)
        order.append(key)
        for child in sorted(children[key]):
            indegree[child] -= 1
            if not indegree[child]:
                ready.append(child)
                ready.sort()
    if len(order) != len(nodes):
        raise ValueError("Kitchen connections contain a cycle.")
    if to_node is not None:
        if to_node not in nodes:
            raise ValueError("Select a node to run to.")
        scope, pending = set(), [to_node]
        while pending:
            key = pending.pop()
            if key not in scope:
                scope.add(key)
                pending.extend(parents[key])
        order = [key for key in order if key in scope]
    if complete:
        for key in order:
            if set(PORTS[nodes[key].kind][0]) != {p for n, p in ports if n == key}:
                raise ValueError(f"{key}: connect its required input before running.")
    return order


def select_dataset(dataset: Dataset, settings: dict) -> Dataset:
    columns = settings["columns"]
    if not columns:
        raise ValueError("Select at least one column.")
    if not set(columns) <= set(dataset.df.columns):
        raise ValueError("Selected columns do not exist in the input dataset.")
    roles = {c.name: c.var_type.value for c in dataset.schema().columns if c.name in columns}
    roles.update(settings["roles"])
    return Dataset(dataset.df.loc[:, columns], overrides=roles)


def prepare_graph(graph: KitchenGraph, snapshots: dict[str, Dataset], *, to_node=None,
                  stop_requested: Callable[[], bool] | None = None) -> PreparedKitchen:
    """Preflight every operation in scope before any fitting, preserving row lineage."""
    order = graph_order(graph, to_node=to_node)
    nodes = {n.id: n for n in graph.nodes}
    prepared = PreparedKitchen(order)
    for key in order:
        if stop_requested and stop_requested():
            prepared.cancelled = True
            break
        node, s = nodes[key], nodes[key].settings
        try:
            if node.kind == "report":
                continue
            inputs = _artifact_inputs(graph, key, prepared.outputs)
            if node.kind == "dataset":
                if s["snapshot"] not in snapshots:
                    raise ValueError("Choose an ordinary dataset snapshot.")
                if not all(isinstance(c, str) and c for c in snapshots[s["snapshot"]].df.columns):
                    raise ValueError("Kitchen datasets require nonempty string column names.")
                table = root_table(s["snapshot"], snapshots[s["snapshot"]])
                prepared.roots[table.id] = table
                outputs = {"data": table}
            elif node.kind == "select":
                outputs = {"data": select_columns(inputs["data"], s)}
            elif node.kind == "column":
                outputs = column_vectors(inputs["data"], s["column"])
            elif node.kind == "map":
                outputs = {"vector": map_vector(inputs["vector"], s)}
            elif node.kind == "combine":
                outputs = {"vector": combine_vectors(inputs["left"], inputs["right"], s)}
            elif node.kind == "build":
                outputs = {"data": build_dataset(inputs["vector"], s["name"])}
            elif node.kind == "add_column":
                outputs = {"data": add_column(inputs["data"], inputs["vector"], s["name"])}
            elif node.kind == "filter":
                outputs = {"data": filter_rows(inputs["data"], inputs["mask"])}
            elif node.kind in ("comparison", "linear"):
                if not s["target"] or not s["features"]:
                    raise ValueError("Choose a numeric target and at least one feature.")
                prepared.analysis_notes[key] = validate_target_dependencies(inputs["data"], s["target"], s["features"])
                _, frame, _ = prepare_regression_problem(inputs["data"].dataset, s["target"], s["features"], "Kitchen model")
                train = len(frame) - math.ceil(len(frame) * s["test_size"])
                if train < 4 or len(frame) - train < 2:
                    raise ValueError("A model needs at least 4 training rows and 2 test rows.")
                continue
            elif node.kind == "correlation":
                aligned(inputs["left"], inputs["right"])
                a = numeric_values(inputs["left"].values, inputs["left"].role)
                b = numeric_values(inputs["right"].values, inputs["right"].role)
                if int((np.isfinite(a) & np.isfinite(b)).sum()) < 3:
                    raise ValueError("Correlation needs at least 3 complete numeric pairs.")
                continue
            else:  # summary is descriptive; value-count applicability is checked before any model fits
                inputs["vector"].values.value_counts(dropna=True)
                continue
            for port, artifact in outputs.items():
                prepared.outputs[key, port] = artifact
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{key}: {exc}") from exc
    if stop_requested and stop_requested():
        prepared.cancelled = True
    return prepared


def _artifact_inputs(graph, key, outputs):
    return {edge.target_port: outputs[edge.source, edge.source_port] for edge in graph.edges if edge.target == key}


def graph_fingerprints(graph: KitchenGraph, snapshot_hashes: dict[str, str]) -> dict[str, str]:
    """Cheap configuration lineage; positions and selection do not change results."""
    order = graph_order(graph, complete=False)
    nodes = {n.id: n for n in graph.nodes}
    fingerprints = {}
    for key in order:
        node = nodes[key]
        inputs = {e.target_port: {"node": e.source, "port": e.source_port, "fingerprint": fingerprints[e.source]}
                  for e in graph.edges if e.target == key}
        fingerprints[key] = _hash({"id": key, "kind": node.kind, "settings": node.settings, "inputs": inputs,
                                  "snapshot": snapshot_hashes.get(node.settings.get("snapshot"), "missing")})
    return fingerprints


def node_states(graph: KitchenGraph, snapshot_hashes: dict[str, str], run: KitchenRun) -> dict[str, str]:
    fingerprints = graph_fingerprints(graph, snapshot_hashes)
    states = {}
    for key, fingerprint in fingerprints.items():
        record, attempt = run.records.get(key), run.attempts.get(key, {})
        if attempt.get("fingerprint") == fingerprint and attempt.get("state") != "completed":
            states[key] = attempt["state"]
        elif record and record.fingerprint != fingerprint:
            states[key] = "stale"
        else:
            states[key] = "completed" if record else "unrun"
    return states


def _dataset_payload(dataset: Dataset) -> dict:
    return {"rows": len(dataset.df), "columns": [{"name": c.name, "type": c.var_type.value, "dtype": c.dtype}
                                                for c in dataset.schema().columns],
            "dataset_fingerprint": dataset_fingerprint(dataset),
            "preview": dataset.df.head(20).astype("string").fillna("—").values.tolist()}


def _vector_metadata(vector):
    return {"name": vector.name, "role": vector.role, "dtype": str(vector.values.dtype), "rows": len(vector.values),
            "missing": int(vector.values.isna().sum()), "dependencies": [list(dep) for dep in sorted(vector.dependencies)],
            "preview": vector.values.head(20).astype("string").fillna("—").tolist()}


def _summary(vector):
    present = vector.values.dropna()
    counts = present.value_counts(dropna=True)
    counts = counts[counts > 0]  # categorical domains may include unobserved levels
    notes = []
    if len(counts) > 50:
        notes.append("Value-count preview shows the 50 most frequent values; the complete vector is retained in its typed artifact.")
    stats = {}
    if vector.role == "numeric":
        if not len(present):
            notes.append("No non-missing numeric values; summary statistics are unavailable.")
        else:
            operations = {"mean": present.mean, "median": present.median,
                          "standard_deviation": lambda: present.std(ddof=1),
                          "minimum": present.min, "maximum": present.max}
            for metric, operation in operations.items():
                label = metric.replace("_", " ")
                if metric == "standard_deviation" and len(present) < 2:
                    stats[metric] = None
                    notes.append("Summary standard deviation requires at least two non-missing values.")
                    continue
                with python_warnings.catch_warnings(record=True) as caught:
                    python_warnings.simplefilter("always")
                    try:
                        value = float(operation())
                    except (OverflowError, FloatingPointError):
                        value = float("nan")
                if math.isfinite(value):
                    stats[metric] = value
                    notes.extend(f"Summary {label}: {warning.message}" for warning in caught)
                else:
                    stats[metric] = None
                    notes.append(f"Summary {label} is unavailable: aggregation produced a non-finite value.")
    return {"method": "vector_summary", "name": vector.name, "role": vector.role, "n_obs": len(present),
            "n_missing": int(vector.values.isna().sum()), "n_unique": int(present.nunique()), "statistics": stats,
            "value_counts": [[str(value), int(count)] for value, count in counts.head(50).items()], "warnings": notes}


def run_graph(graph: KitchenGraph, snapshots: dict[str, Dataset], *, previous: KitchenRun | None = None,
              to_node: str | None = None, stop_requested: Callable[[], bool] | None = None,
              on_progress: Callable[[dict], None] | None = None) -> KitchenExecution:
    """Explicitly rerun the requested closure, sequentially. Cancel between operations.

    Validation errors raise before fitting. Operation errors return a failed attempt
    while retaining completed records. A model already fitting finishes before stop.
    """
    order = graph_order(graph, to_node=to_node)
    if not order:
        raise ValueError("Add nodes before running the Kitchen.")
    nodes = {n.id: n for n in graph.nodes}
    fingerprints = graph_fingerprints(graph, {k: dataset_fingerprint(v) for k, v in snapshots.items()})
    previous = previous or KitchenRun()
    records = {k: deepcopy(v) for k, v in previous.records.items() if k in nodes}
    attempts = {k: deepcopy(v) for k, v in previous.attempts.items() if k in nodes}
    versions = runtime_versions()
    executed, warnings, artifacts, analysis = [], [], {}, {}

    def status(key, state, error=""):
        attempts[key] = {"fingerprint": fingerprints[key], "state": state, "error": error}
        if on_progress:
            on_progress({"node": key, "state": state, "error": error})

    prepared = prepare_graph(graph, snapshots, to_node=to_node, stop_requested=stop_requested)
    if prepared.cancelled:
        for key in order:
            status(key, "cancelled")
        warnings.append("Cancelled during input validation; previous completed node results were retained.")
        return KitchenExecution(KitchenRun(records, attempts, executed, warnings), {}, order)
    for key in order:
        status(key, "queued")
    for index, key in enumerate(order):
        if stop_requested and stop_requested():
            for remaining in order[index:]:
                status(remaining, "cancelled")
            warnings.append("Cancelled between operations; completed node results were retained.")
            break
        node, s = nodes[key], nodes[key].settings
        status(key, "running")
        try:
            if node.kind == "report":
                edge = next(e for e in graph.edges if e.target == key and e.target_port == "result")
                payload = deepcopy(analysis[edge.source, edge.source_port])
                notes = list(records[edge.source].warnings)
            elif node.kind in ("comparison", "linear", "summary", "correlation"):
                inputs = _artifact_inputs(graph, key, prepared.outputs)
                notes = list(dict.fromkeys(w for artifact in inputs.values() for w in artifact.warnings))
                notes.extend(prepared.analysis_notes.get(key, []))
                if node.kind in ("comparison", "linear"):
                    table = inputs["data"]
                    kwargs = {k: v for k, v in s.items() if k != "preprocessing"}
                    function = compare_regressors if node.kind == "comparison" else linear_regression
                    result = function(table.dataset, preprocessing=PreprocessingConfig(**s["preprocessing"]), **kwargs)
                    payload = result.to_dict()
                    payload["input_rows"] = len(table.dataset.df)
                    payload["input_dataset_fingerprint"] = dataset_fingerprint(table.dataset)
                elif node.kind == "summary":
                    payload = _summary(inputs["vector"])
                else:
                    left, right = inputs["left"], inputs["right"]
                    pairs = pd.DataFrame({"left": left.values.array, "right": right.values.array})
                    with python_warnings.catch_warnings(record=True) as caught:
                        python_warnings.simplefilter("always")
                        result = correlation(Dataset(pairs, overrides={"left": "numeric", "right": "numeric"}),
                                             "left", "right", **s)
                    notes.extend(str(w.message) for w in caught)
                    payload = result.to_dict()
                    points = pairs.dropna().astype(float).to_numpy()
                    indices = np.linspace(0, len(points) - 1, min(3000, len(points)), dtype=int)
                    payload["points"] = points[indices].tolist()
                    payload["left_name"], payload["right_name"] = left.name, right.name
                notes.extend(payload.get("warnings", []))
                payload["input_artifacts"] = {port: artifact.id for port, artifact in inputs.items()}
                payload["analysis_kind"] = node.kind
                roots = set().union(*(artifact_roots(value) for value in inputs.values()))
                payload["source_snapshot_ids"] = sorted({prepared.roots[root].root_snapshot for root in roots})
                payload["snapshot_id"] = payload["source_snapshot_ids"][0] if payload["source_snapshot_ids"] else ""
                payload["warnings"] = list(dict.fromkeys(notes))
                analysis[key, "result"] = payload
            else:
                consumed = _artifact_inputs(graph, key, prepared.outputs)
                outputs = {port: value for (identifier, port), value in prepared.outputs.items() if identifier == key}
                refs = {port: artifact.id for port, artifact in outputs.items()}
                notes = list(dict.fromkeys(w for artifact in outputs.values() for w in artifact.warnings))
                if "data" in outputs:
                    payload = _dataset_payload(outputs["data"].dataset)
                    payload["dependencies"] = {name: [list(dep) for dep in sorted(deps)] for name, deps in outputs["data"].dependencies.items()}
                else:
                    payload = {"vectors": {port: _vector_metadata(value) for port, value in outputs.items()}}
                roots = set().union(*(artifact_roots(value) for value in outputs.values()))
                payload["source_snapshot_ids"] = sorted({prepared.roots[root].root_snapshot for root in roots})
                payload["snapshot_id"] = payload["source_snapshot_ids"][0] if payload["source_snapshot_ids"] else ""
                payload["outputs"] = refs
                payload["input_artifacts"] = {port: value.id for port, value in consumed.items()}
                artifacts.update({artifact.id: artifact for artifact in outputs.values()})
                artifacts.update({root: prepared.roots[root] for root in roots})
            inputs = {e.target_port: {"node": e.source, "port": e.source_port, "fingerprint": fingerprints[e.source],
                                     "artifact": prepared.outputs[e.source, e.source_port].id
                                     if (e.source, e.source_port) in prepared.outputs else None}
                      for e in graph.edges if e.target == key}
            records[key] = KitchenRecord(key, node.kind, fingerprints[key], deepcopy(s), inputs, payload,
                                          datetime.now(UTC).isoformat(), versions, list(notes))
            executed.append(key)
            status(key, "completed")
        except Exception as exc:
            status(key, "failed", str(exc) or type(exc).__name__)
            for remaining in order[index + 1:]:
                status(remaining, "blocked")
            warnings.append(f"{key} failed; previous completed results were retained.")
            break
    completed = set(executed)
    datasets = {key: value.dataset for (key, _), value in prepared.outputs.items()
                if key in completed and isinstance(value, TableArtifact)}
    vectors = {key: value for key, value in prepared.outputs.items()
               if key[0] in completed and isinstance(value, VectorArtifact)}
    return KitchenExecution(KitchenRun(records, attempts, executed, warnings), datasets, order, artifacts, vectors)
