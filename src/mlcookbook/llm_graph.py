"""Bounded import and actual llama.cpp CPU operation-graph capture.

Captures are separate helper executions, never claims about a connected server's
live process. The helper is pinned to the verified Windows b11193 public ABI.
Opening recorded graphs requires neither this runtime nor the original model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, deque
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

FORMAT = "mlcookbook-llama-graph"
VERSION = 1
UPSTREAM_COMMIT = "4e7481175"
MAX_CAPTURE_BYTES = 64 * 1024 * 1024
MAX_NODES = 60_000
MAX_EDGES = 600_000
MAX_LOG_BYTES = 1024 * 1024
SUPPORTED_DLLS = {
    "llama.dll": "f99c04c7444ad30a7e306d1f1f5a2ad517e2bc7b753a6dfd654ae7931cd1bade",
    "ggml-base.dll": "7e4e8cf1edeeb0f8b40de183095a39a51bf6090ea09a1cf928512a250c481d60",
    "ggml.dll": "bb616de54ef57f8804d1d22a4bf104e2d98c8e8399b08374e70a7c096322b851",
}


def _check(stop_requested=None, deadline=None):
    if stop_requested is not None and stop_requested():
        raise InterruptedError("Graph capture cancelled; the prior completed capture is preserved.")
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("Graph capture reached its time limit; the prior completed capture is preserved.")


def _string(value, name, maximum=4096, *, empty=False):
    if not isinstance(value, str) or (not value and not empty) or len(value) > maximum:
        raise ValueError(f"Graph {name} is invalid or exceeds its budget.")


def _integer(value, name, low=0, high=2**63 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Graph {name} is invalid or exceeds its budget.")


def _bounded_json(value):
    try:
        raw = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise ValueError("A graph capture must contain finite JSON data only.") from exc
    if len(raw) > MAX_CAPTURE_BYTES:
        raise ValueError("Graph capture exceeds the 64 MiB file budget.")


def validate_graph_capture(value: dict, *, stop_requested=None) -> dict:
    """Validate a portable declarative capture, including dependency integrity.

    Unknown additional fields are retained for forward-readable observations;
    required versioned graph fields must still satisfy this contract.
    """
    _check(stop_requested)
    if not isinstance(value, dict) or value.get("format") != FORMAT or type(value.get("version")) is not int or value["version"] != VERSION:
        raise ValueError("Unsupported compute-graph capture format/version.")
    _bounded_json(value)
    if not isinstance(value.get("provenance"), dict):
        raise ValueError("Graph capture provenance must be an object.")
    provenance = value["provenance"]
    identities = []
    if "helper" in provenance:
        identities.append(provenance["helper"])
    if "model_files" in provenance:
        if not isinstance(provenance["model_files"], list) or len(provenance["model_files"]) > 128:
            raise ValueError("Graph model identity records are invalid.")
        identities.extend(provenance["model_files"])
    if "runtime_files" in provenance:
        if not isinstance(provenance["runtime_files"], dict) or len(provenance["runtime_files"]) > 256:
            raise ValueError("Graph runtime identity records are invalid.")
        identities.extend(provenance["runtime_files"].values())
    for identity in identities:
        if not isinstance(identity, dict) or not isinstance(identity.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]):
            raise ValueError("Graph file identity SHA-256 is invalid.")
        _string(identity.get("path"), "file identity path", 32768)
        _integer(identity.get("size_bytes"), "file identity byte count")
    warnings = value.get("warnings")
    if not isinstance(warnings, list) or len(warnings) > 1000 or any(not isinstance(item, str) or len(item) > 16_000 for item in warnings):
        raise ValueError("Graph capture warnings are invalid.")
    graphs = value.get("graphs")
    if not isinstance(graphs, list) or not 1 <= len(graphs) <= 17:
        raise ValueError("Graph capture must contain 1..17 bounded phase graphs.")
    graph_ids, total_nodes, total_edges = set(), 0, 0
    for graph in graphs:
        _check(stop_requested)
        if not isinstance(graph, dict):
            raise ValueError("Each captured graph must be an object.")
        _string(graph.get("id"), "ID", 256)
        if graph["id"] in graph_ids:
            raise ValueError("Duplicate captured graph ID.")
        graph_ids.add(graph["id"])
        if graph.get("phase") not in ("prefill", "decode"):
            raise ValueError("Captured graph phase must be prefill or decode.")
        tokens = graph.get("input_token_ids", [])
        if not isinstance(tokens, list) or len(tokens) > 512:
            raise ValueError("Captured input tokens exceed their budget.")
        for token in tokens:
            _integer(token, "token ID", 0, 2**31 - 1)
        if "source_token_position" in graph:
            _integer(graph["source_token_position"], "source token position", 0, 1024)
        nodes = graph.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("A captured graph must contain operation/tensor nodes.")
        total_nodes += len(nodes)
        if total_nodes > MAX_NODES:
            raise ValueError("Graph capture exceeds its node budget.")
        by_id = {}
        for index, node in enumerate(nodes):
            if index % 256 == 0:
                _check(stop_requested)
            if not isinstance(node, dict):
                raise ValueError("A graph node must be an object.")
            _string(node.get("id"), "node ID", 256)
            if node["id"] in by_id:
                raise ValueError("Duplicate node ID inside a captured graph.")
            by_id[node["id"]] = node
            _string(node.get("name"), "node name", empty=True)
            _string(node.get("op"), "operation", 256)
            _string(node.get("dtype"), "tensor type", 256)
            shape, strides = node.get("shape"), node.get("strides")
            if not isinstance(shape, list) or not 1 <= len(shape) <= 4:
                raise ValueError("Graph tensor shape must contain 1..4 dimensions.")
            if not isinstance(strides, list) or len(strides) != len(shape):
                raise ValueError("Graph tensor strides must match its dimensions.")
            for dimension in shape:
                _integer(dimension, "tensor dimension", 0, 2**40)
            for stride in strides:
                _integer(stride, "tensor stride")
            if math.prod(shape) > 2**63 - 1:
                raise ValueError("Graph tensor element count exceeds its budget.")
            inputs = node.get("inputs")
            if not isinstance(inputs, list) or len(inputs) > 16:
                raise ValueError("Graph node inputs are invalid or exceed their budget.")
            for source in inputs:
                _string(source, "input reference", 256)
            if "input_slots" in node:
                slots = node["input_slots"]
                if not isinstance(slots, list) or len(slots) != len(inputs):
                    raise ValueError("Graph input slots must match the ordered input references.")
                for slot in slots:
                    _integer(slot, "input port slot", 0, 15)
                if len(set(slots)) != len(slots):
                    raise ValueError("Graph input port slots must be unique.")
            total_edges += len(inputs)
            if total_edges > MAX_EDGES:
                raise ValueError("Graph capture exceeds its edge budget.")
            if "bytes" in node:
                _integer(node["bytes"], "tensor storage span")
            if "op_code" in node:
                _integer(node["op_code"], "operation code", 0, 2**31 - 1)
            if "op_parameters_i32" in node:
                parameters = node["op_parameters_i32"]
                if not isinstance(parameters, list) or len(parameters) > 64:
                    raise ValueError("Graph operation parameters exceed their budget.")
                for parameter in parameters:
                    _integer(parameter, "operation parameter", -(2**31), 2**31 - 1)
            if "observed" in node and type(node["observed"]) is not bool:
                raise ValueError("Graph observed flag must be boolean.")
            if "view_offset_bytes" in node:
                _integer(node["view_offset_bytes"], "view offset")
        dependents = {key: [] for key in by_id}
        indegree = {key: 0 for key in by_id}
        for node in nodes:
            sources = list(node["inputs"])
            alias = node.get("view_source")
            if alias is not None:
                _string(alias, "view source", 256)
                sources.append(alias)
            for source in set(sources):
                if source not in by_id:
                    raise ValueError("Graph contains a missing input or view dependency.")
                if source == node["id"]:
                    raise ValueError("Graph contains a self dependency.")
                dependents[source].append(node["id"])
                indegree[node["id"]] += 1
        ready = deque(key for key, count in indegree.items() if count == 0)
        visited = 0
        while ready:
            if visited % 512 == 0:
                _check(stop_requested)
            visited += 1
            for target in dependents[ready.popleft()]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(nodes):
            raise ValueError("Captured tensor dependencies contain a cycle.")
    _check(stop_requested)
    return deepcopy(value)


def compare_graphs(capture: dict, left: int = 0, right: int = 1) -> dict:
    """Compare phase graphs by unambiguous tensor names, never allocation IDs.

    Naming is evidence for alignment within a recorded capture, not a universal
    identity across architectures or runtime revisions. Ambiguous names remain
    visible and are excluded from claimed structural matches.
    """
    capture = validate_graph_capture(capture)
    for index in (left, right):
        _integer(index, "comparison graph index", 0, len(capture["graphs"]) - 1)
    a, b = capture["graphs"][left], capture["graphs"][right]
    counts_a, counts_b = Counter(node["name"] for node in a["nodes"]), Counter(node["name"] for node in b["nodes"])
    nodes_a = {node["name"]: node for node in a["nodes"] if node["name"] and counts_a[node["name"]] == 1}
    nodes_b = {node["name"]: node for node in b["nodes"] if node["name"] and counts_b[node["name"]] == 1}
    id_names_a = {node["id"]: node["name"] for node in a["nodes"]}
    id_names_b = {node["id"]: node["name"] for node in b["nodes"]}
    ambiguous = {name for name in counts_a.keys() | counts_b.keys()
                 if not name or counts_a[name] > 1 or counts_b[name] > 1}
    rows, matched, changed_edges = [], 0, 0
    for name in sorted(counts_a.keys() | counts_b.keys()):
        na, nb = nodes_a.get(name), nodes_b.get(name)
        row = {"name": name or "(unnamed)", "left_op": na["op"] if na else None,
               "right_op": nb["op"] if nb else None, "left_shape": na["shape"] if na else None,
               "right_shape": nb["shape"] if nb else None, "left_dtype": na["dtype"] if na else None,
               "right_dtype": nb["dtype"] if nb else None}
        if name in ambiguous:
            row.update(change="Ambiguous identity", left_count=counts_a[name], right_count=counts_b[name])
            rows.append(row)
            continue
        if na is None or nb is None:
            row["change"] = "Added" if na is None else "Removed"
            rows.append(row)
            continue
        matched += 1
        changed = [field for field in ("op", "shape", "dtype", "strides", "op_parameters_i32", "bytes")
                   if na.get(field) != nb.get(field)]
        inputs_a = [id_names_a[source] for source in na["inputs"]]
        inputs_b = [id_names_b[source] for source in nb["inputs"]]
        row["left_inputs"], row["right_inputs"] = inputs_a, inputs_b
        if not any(name in ambiguous for name in inputs_a + inputs_b):
            if inputs_a != inputs_b or na.get("input_slots") != nb.get("input_slots"):
                changed.append("input dependencies")
                changed_edges += 1
        else:
            row["dependency_alignment"] = "Unavailable: an input name is ambiguous."
        if changed:
            row["change"] = ", ".join(changed)
            rows.append(row)
    warnings = ["Alignment uses unique exact tensor names within this capture; node IDs and allocation order are not matched."]
    if ambiguous:
        warnings.append(f"{len(ambiguous)} empty/duplicate tensor names could not be aligned unambiguously.")
    return {"summary": f"{a['id']} vs {b['id']}: {matched:,} uniquely named tensors aligned; "
                       f"{len(rows):,} differences/ambiguous identities; {changed_edges:,} matched nodes with changed inputs.",
            "rows": rows, "warnings": warnings, "left_graph": a["id"], "right_graph": b["id"]}


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate graph JSON field.")
        value[key] = item
    return value


def load_graph_capture(path: str | Path, *, stop_requested=None) -> dict:
    """Read a bounded UTF-8 capture without running or opening its source paths."""
    _check(stop_requested)
    path = Path(path)
    if path.stat().st_size > MAX_CAPTURE_BYTES:
        raise ValueError("Graph capture exceeds the 64 MiB file budget.")
    with path.open("rb") as stream:
        raw = stream.read(MAX_CAPTURE_BYTES + 1)
    if len(raw) > MAX_CAPTURE_BYTES:
        raise ValueError("Graph capture exceeds the 64 MiB file budget.")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite graph number.")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Capture is not valid bounded UTF-8 JSON.") from exc
    return validate_graph_capture(value, stop_requested=stop_requested)


def default_helper_path() -> Path:
    """Resolve the packaged/source helper path without probing any runtime."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "tools" / "llama_research" / "llama-graph-capture.exe"
    return Path(__file__).resolve().parents[2] / "tools" / "llama_research" / "build" / "llama-graph-capture.exe"


def _identity(path, stop_requested, deadline):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            _check(stop_requested, deadline)
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("A graph capture input changed while computing its identity.")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns}


def _run_process(arguments, work, *, stop_requested, deadline, output_path=None):
    """Own one hidden child, polling bounded files rather than unbounded pipes."""
    _check(stop_requested, deadline)
    stdout_path, stderr_path = work / "stdout.log", work / "stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                   cwd=str(work), shell=False)
        try:
            while True:
                _check(stop_requested, deadline)
                if max(stdout_path.stat().st_size, stderr_path.stat().st_size) > MAX_LOG_BYTES:
                    raise ValueError("Graph helper exceeded its log budget.")
                if output_path is not None and output_path.exists() and output_path.stat().st_size > MAX_CAPTURE_BYTES:
                    raise ValueError("Graph helper exceeded its output budget.")
                try:
                    code = process.wait(timeout=0.05)
                    break
                except subprocess.TimeoutExpired:
                    pass
            _check(stop_requested, deadline)
            if max(stdout_path.stat().st_size, stderr_path.stat().st_size) > MAX_LOG_BYTES:
                raise ValueError("Graph helper exceeded its log budget.")
            if output_path is not None and output_path.exists() and output_path.stat().st_size > MAX_CAPTURE_BYTES:
                raise ValueError("Graph helper exceeded its output budget.")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    with stdout_path.open("rb") as stream:
        stdout_value = stream.read(MAX_LOG_BYTES).decode("utf-8", errors="replace")
    with stderr_path.open("rb") as stream:
        stderr_value = stream.read(MAX_LOG_BYTES).decode("utf-8", errors="replace")
    if code:
        raise ValueError("Graph helper failed: " + (stderr_value or stdout_value or f"exit code {code}")[-4000:])
    return stdout_value + "\n" + stderr_value


def capture_graph(model_path: str | Path, *, helper_path: str | Path = "", runtime_dir: str | Path = "",
                  prompt: str, max_tokens: int = 1, max_seconds: float = 90,
                  stop_requested=None, on_progress=None) -> dict:
    """Capture actual evaluated CPU graph dependencies in a separate execution.

    The helper uses a new 1024-token context, raw prompt, greedy sampling and
    four CPU threads. These are research settings, unrelated to the live server.
    """
    _integer(max_tokens, "decode step count", 1, 16)
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 600:
        raise ValueError("Graph capture deadline must be 1..600 seconds.")
    _string(prompt, "prompt", 32768)
    if not prompt.strip() or len(prompt.encode("utf-8")) > 32768:
        raise ValueError("Capture prompt must contain 1..32768 UTF-8 bytes.")
    if stop_requested is not None and not callable(stop_requested) or on_progress is not None and not callable(on_progress):
        raise ValueError("Graph capture callbacks must be callable or None.")
    _check(stop_requested)
    if os.name != "nt":
        raise ValueError("The initial actual graph helper supports Windows x64 only. Recorded captures remain portable.")
    started, deadline = monotonic(), monotonic() + max_seconds
    helper = Path(helper_path).expanduser().resolve() if helper_path else default_helper_path()
    model = Path(model_path).expanduser().resolve()
    if not helper.is_file():
        raise ValueError("Graph capture helper is missing. Build tools/llama_research or select the packaged helper.")
    if not model.is_file():
        raise ValueError("Select an existing local single-file GGUF model for graph capture.")
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", model.name, re.IGNORECASE):
        raise ValueError("The first graph-capture release supports single-file GGUF models; split-shard capture is unsupported.")
    with model.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("The selected graph capture model is not GGUF.")
    if runtime_dir:
        runtime = Path(runtime_dir).expanduser().resolve()
    else:
        server = shutil.which("llama-server")
        if server is None:
            raise ValueError("Select the installed llama.cpp runtime directory; llama-server is not on PATH.")
        runtime = Path(server).resolve().parent
    if not runtime.is_dir() or not (runtime / "llama-server.exe").is_file():
        raise ValueError("Select a llama.cpp runtime directory containing llama-server.exe and its DLLs.")
    def progress(message):
        if on_progress is not None:
            on_progress({"message": message})
    progress("Checking the helper and exact installed runtime ABI…")
    files = {}
    for name in SUPPORTED_DLLS:
        path = runtime / name
        if not path.is_file():
            raise ValueError("The selected graph capture runtime is missing " + name + ".")
        files[name] = _identity(path, stop_requested, deadline)
        if files[name]["sha256"] != SUPPORTED_DLLS[name]:
            raise ValueError("Graph capture supports the verified b11193 / 4e7481175 runtime DLLs only. "
                             "This runtime has a different ABI/build identity; ordinary server inference is still available.")
    for path in sorted(runtime.glob("ggml-*.dll")):
        if path.name not in files:
            files[path.name] = _identity(path, stop_requested, deadline)
    files["llama-server.exe"] = _identity(runtime / "llama-server.exe", stop_requested, deadline)
    helper_identity = _identity(helper, stop_requested, deadline)
    progress("Fingerprinting the selected model without changing the server…")
    model_identity = _identity(model, stop_requested, deadline)
    with tempfile.TemporaryDirectory(prefix="mlcookbook-graph-") as temporary:
        work = Path(temporary)
        version = _run_process([str(runtime / "llama-server.exe"), "--version"], work,
                               stop_requested=stop_requested, deadline=deadline)
        if UPSTREAM_COMMIT not in version or "11193" not in version:
            raise ValueError("The graph helper requires llama.cpp b11193 / 4e7481175. The selected server build differs.")
        (work / "prompt.txt").write_text(prompt, encoding="utf-8")
        output = work / "capture.json"
        progress("Running a separate CPU capture: prompt processing followed by decode…")
        _run_process([str(helper), str(runtime), str(model), str(work / "prompt.txt"), str(output), str(max_tokens)],
                     work, stop_requested=stop_requested, deadline=deadline, output_path=output)
        result = load_graph_capture(output, stop_requested=stop_requested)
    _check(stop_requested, deadline)
    if len(result["graphs"]) != max_tokens + 1 or result["graphs"][0]["phase"] != "prefill" or any(
            graph["phase"] != "decode" for graph in result["graphs"][1:]):
        raise ValueError("The helper did not return the requested prefill/decode graph sequence.")
    if result["provenance"].get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Graph helper returned an incompatible source revision.")
    for identity in [helper_identity, model_identity, *files.values()]:
        after = Path(identity["path"]).stat()
        if (after.st_size, after.st_mtime_ns) != (identity["size_bytes"], identity["mtime_ns"]):
            raise ValueError("A model/helper/runtime file changed during graph capture.")
    result["provenance"].update({
        "captured_utc": datetime.now(UTC).isoformat(), "helper": helper_identity,
        "model_files": [model_identity], "runtime_files": files, "runtime_version_output": version.strip(),
        "prompt": prompt, "requested_decode_steps": max_tokens, "action_seconds": monotonic() - started,
        "deadline_seconds": max_seconds,
        "attribution": "Separate helper-loaded CPU execution; not the connected llama-server process.",
        "model_identity_scope": "Selected single-file GGUF loaded by this helper, including its tokenizer metadata; server binding unverified.",
        "storage_bytes_scope": "Each tensor's storage span from GGML; views and shared storage are not independent allocations.",
        "source_revision_url": "https://github.com/ggml-org/llama.cpp/tree/" + UPSTREAM_COMMIT,
    })
    progress("Actual prefill/decode operation dependencies captured and validated.")
    return validate_graph_capture(result, stop_requested=stop_requested)
