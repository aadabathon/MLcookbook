"""The Kitchen: a small typed DAG of existing, auditable operations.

No Qt, arbitrary code, fitted-estimator artifacts or graph-level fitted transforms.
Dataset objects are runtime artifacts; completed records contain JSON-safe data.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime

from mlcookbook.comparison import compare_regressors
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.fingerprint import dataset_fingerprint
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType
from mlcookbook.experiments import runtime_versions
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.regression._problem import prepare_regression_problem

PORTS = {
    "dataset": ({}, {"data": "dataset"}),
    "select": ({"data": "dataset"}, {"data": "dataset"}),
    "comparison": ({"data": "dataset"}, {"result": "comparison"}),
    "report": ({"result": "comparison"}, {}),
}


def default_settings(kind: str) -> dict:
    settings = {
        "dataset": {"snapshot": ""},
        "select": {"columns": [], "roles": {}},
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
    inputs: dict[str, str]
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


@dataclass
class KitchenDocument:
    graph: KitchenGraph = field(default_factory=KitchenGraph)
    snapshots: dict[str, Dataset] = field(default_factory=dict)
    snapshot_info: dict[str, dict] = field(default_factory=dict)
    run: KitchenRun = field(default_factory=KitchenRun)
    layout: dict[str, list[float]] = field(default_factory=dict)
    selected: str | None = None
    warnings: list[str] = field(default_factory=list)


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
    if node.kind == "comparison":
        if not isinstance(s["target"], str) or not isinstance(s["features"], list) \
                or not all(isinstance(c, str) for c in s["features"]) or len(set(s["features"])) != len(s["features"]):
            raise ValueError(f"{node.id}: target/features must be unique column names.")
        for key, low, high in (("n_estimators", 1, 1000), ("min_samples_leaf", 1, 1_000_000),
                               ("random_state", 0, 2**32 - 1), ("cv_folds", 2, 20)):
            if type(s[key]) is not int or not low <= s[key] <= high:
                raise ValueError(f"{node.id}: {key} must be an integer from {low} to {high}.")
        if s["max_depth"] is not None and (type(s["max_depth"]) is not int or not 1 <= s["max_depth"] <= 100):
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


def prepare_graph(graph: KitchenGraph, snapshots: dict[str, Dataset], *, to_node=None) -> tuple[list[str], dict[str, Dataset]]:
    """Preflight every operation in scope before any fitting, preserving row lineage."""
    order = graph_order(graph, to_node=to_node)
    nodes = {n.id: n for n in graph.nodes}
    parents = {e.target: e.source for e in graph.edges}
    data = {}
    for key in order:
        node, s = nodes[key], nodes[key].settings
        try:
            if node.kind == "dataset":
                if s["snapshot"] not in snapshots:
                    raise ValueError("Choose an ordinary dataset snapshot.")
                data[key] = snapshots[s["snapshot"]]
                if not all(isinstance(c, str) and c for c in data[key].df.columns):
                    raise ValueError("Kitchen datasets require nonempty string column names.")
            elif node.kind == "select":
                data[key] = select_dataset(data[parents[key]], s)
            elif node.kind == "comparison":
                if not s["target"] or not s["features"]:
                    raise ValueError("Choose a numeric target and at least one feature.")
                _, frame, _ = prepare_regression_problem(data[parents[key]], s["target"], s["features"], "Kitchen comparison")
                train = len(frame) - math.ceil(len(frame) * s["test_size"])
                if train < 4 or len(frame) - train < 2:
                    raise ValueError("Comparison needs at least 4 training rows and 2 test rows.")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{key}: {exc}") from exc
    return order, data


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


def run_graph(graph: KitchenGraph, snapshots: dict[str, Dataset], *, previous: KitchenRun | None = None,
              to_node: str | None = None, stop_requested: Callable[[], bool] | None = None,
              on_progress: Callable[[dict], None] | None = None) -> KitchenExecution:
    """Explicitly rerun the requested closure, sequentially. Cancel between operations.

    Validation errors raise before fitting. Operation errors return a failed attempt
    while retaining completed records. A model already fitting finishes before stop.
    """
    order, data = prepare_graph(graph, snapshots, to_node=to_node)
    if not order:
        raise ValueError("Add nodes before running the Kitchen.")
    nodes = {n.id: n for n in graph.nodes}
    parents = {e.target: e.source for e in graph.edges}
    fingerprints = graph_fingerprints(graph, {k: dataset_fingerprint(v) for k, v in snapshots.items()})
    previous = previous or KitchenRun()
    records = {k: deepcopy(v) for k, v in previous.records.items() if k in nodes}
    attempts = {k: deepcopy(v) for k, v in previous.attempts.items() if k in nodes}
    versions = runtime_versions()
    executed, warnings = [], []

    def status(key, state, error=""):
        attempts[key] = {"fingerprint": fingerprints[key], "state": state, "error": error}
        if on_progress:
            on_progress({"node": key, "state": state, "error": error})

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
            source_key = key
            while nodes[source_key].kind != "dataset":
                source_key = parents[source_key]
            snapshot_id = nodes[source_key].settings["snapshot"]
            if node.kind in ("dataset", "select"):
                payload, notes = _dataset_payload(data[key]), []
            elif node.kind == "comparison":
                kwargs = {k: v for k, v in s.items() if k != "preprocessing"}
                result = compare_regressors(data[parents[key]], preprocessing=PreprocessingConfig(**s["preprocessing"]), **kwargs)
                payload, notes = result.to_dict(), result.warnings
                payload["input_rows"] = len(data[parents[key]].df)
                payload["input_dataset_fingerprint"] = dataset_fingerprint(data[parents[key]])
            else:
                payload = deepcopy(records[parents[key]].payload)
                notes = list(records[parents[key]].warnings)
            payload["snapshot_id"] = snapshot_id
            inputs = {e.target_port: fingerprints[e.source] for e in graph.edges if e.target == key}
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
    return KitchenExecution(KitchenRun(records, attempts, executed, warnings), data, order)
