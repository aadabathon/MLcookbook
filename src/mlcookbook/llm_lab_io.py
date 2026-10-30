"""Atomic, bounded, data-only archives for Local LLM research sessions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path

FORMAT = "mlcookbook-llm-lab"
VERSION = 1
MAX_DOCUMENT = 64 * 1024 * 1024
MAX_ARCHIVE = 80 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, maximum=2**63 - 1, minimum=0):
    return type(value) is int and minimum <= value <= maximum


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _string(value, maximum=16000):
    return isinstance(value, str) and len(value) <= maximum


def _warnings(value):
    _require(isinstance(value, list) and len(value) <= 250000 and all(_string(item) for item in value),
             "Research warnings must be bounded text entries.")


def _state(value):
    _require(isinstance(value, dict), "Research settings must be an object.")
    limits = {"tab": (0, 2, True), "capture_tokens": (1, 16, True), "capture_seconds": (1, 600, False),
              "repetitions": (1, 100, True), "warmups": (0, 10, True), "benchmark_tokens": (1, 512, True),
              "benchmark_seconds": (1, 120, False), "graph_page": (0, 1000000, True)}
    strings = {"model_path": 4096, "helper_path": 4096, "runtime_dir": 4096, "capture_prompt": 16000,
               "benchmark_prompt": 16000, "tensor_search": 16000, "graph_search": 16000, "graph_selected": 4096}
    other = {"graph_phase", "graph_group", "cache_policy", "connection", "graph_trace", "graph_view",
             "tensor_selected", "trial_selected"}
    _require(set(value) <= limits.keys() | strings.keys() | other, "Research settings contain unknown keys.")
    for key, (low, high, integer) in limits.items():
        if key in value:
            number = value[key]
            _require(_integer(number, high, low) if integer else _number(number) and low <= number <= high,
                     f"Research setting {key} is outside its supported range.")
    for key, maximum in strings.items():
        if key in value:
            _require(_string(value[key], maximum), f"Invalid research setting: {key}.")
    if "cache_policy" in value:
        _require(value["cache_policy"] in ("uncontrolled", "disabled", "reuse"), "Unsupported prompt-cache policy.")
    if value.get("graph_phase") is not None:
        _require(_integer(value["graph_phase"], 1000000), "Invalid graph phase selection.")
    if value.get("graph_group") is not None:
        _require(_string(value["graph_group"], 4096), "Invalid graph group selection.")
    for key, maximum in (("tensor_selected", 1000000), ("trial_selected", 110)):
        if value.get(key) is not None:
            _require(_integer(value[key], maximum), f"Invalid research selection: {key}.")
    if "connection" in value:
        connection = value["connection"]
        _require(isinstance(connection, dict) and set(connection) == {"base_url", "model"}
                 and _string(connection["base_url"], 2048) and _string(connection["model"], 512),
                 "Invalid research server connection.")
    trace = value.get("graph_trace")
    if trace is not None:
        _require(isinstance(trace, list) and len(trace) <= 100000 and all(_string(item, 4096) for item in trace),
                 "Invalid graph trace selection.")
    if "graph_view" in value:
        view = value["graph_view"]
        _require(isinstance(view, dict) and set(view) == {"scale", "center"}, "Invalid graph viewport.")
        _require(_number(view["scale"]) and 0.001 <= view["scale"] <= 10, "Invalid graph viewport scale.")
        center = view["center"]
        _require(isinstance(center, list) and len(center) == 2 and
                 all(type(item) in (int, float) and math.isfinite(item) and abs(item) <= 1e9 for item in center),
                 "Invalid graph viewport center.")


def _inspection(value):
    required = {"path", "files", "metadata", "tensors", "summary", "warnings"}
    _require(isinstance(value, dict) and required <= value.keys(), "Invalid model inspection artifact.")
    _require(_string(value["path"], 4096) and isinstance(value["metadata"], dict) and isinstance(value["summary"], dict),
             "Invalid model inspection metadata.")
    files = value["files"]
    _require(isinstance(files, list) and 1 <= len(files) <= 64, "Invalid model inspection file inventory.")
    for file in files:
        _require(isinstance(file, dict) and _string(file.get("path"), 4096)
                 and isinstance(file.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", file["sha256"])
                 and _integer(file.get("size_bytes")), "Invalid inspected-file identity.")
    tensors = value["tensors"]
    _require(isinstance(tensors, list) and len(tensors) <= 250000, "Invalid or oversized tensor inventory.")
    for tensor in tensors:
        _require(isinstance(tensor, dict) and _string(tensor.get("name"), 4096) and bool(tensor["name"])
                 and _string(tensor.get("dtype"), 128) and _string(tensor.get("group"), 4096),
                 "Invalid tensor descriptor.")
        shape = tensor.get("shape")
        _require(isinstance(shape, list) and 1 <= len(shape) <= 4 and all(_integer(dim, 2**40, 1) for dim in shape)
                 and math.prod(shape) <= 2**63 - 1, "Invalid inspected tensor shape.")
        _require(_integer(tensor.get("elements")) and tensor["elements"] == math.prod(shape)
                 and _integer(tensor.get("offset"))
                 and (tensor.get("bytes") is None or _integer(tensor["bytes"])), "Invalid tensor storage accounting.")
    _require(type(value["summary"].get("tensor_count")) is int and value["summary"]["tensor_count"] == len(tensors)
             and type(value["summary"].get("file_count")) is int and value["summary"]["file_count"] == len(files),
             "Model inspection totals do not match its inventory.")
    _warnings(value["warnings"])


def _benchmark(value):
    required = {"method", "base_url", "model", "created_utc", "settings", "trials", "summary", "warnings"}
    _require(isinstance(value, dict) and required <= value.keys() and value["method"] == "benchmark_llamacpp",
             "Invalid performance artifact.")
    _require(_string(value["base_url"], 2048) and _string(value["model"], 512) and _string(value["created_utc"], 128)
             and isinstance(value["settings"], dict) and isinstance(value["summary"], dict), "Invalid benchmark provenance.")
    settings = value["settings"]
    _require(_integer(settings.get("repetitions"), 100, 1) and _integer(settings.get("warmups"), 10),
             "Invalid benchmark plan.")
    trials = value["trials"]
    _require(isinstance(trials, list) and len(trials) <= settings["repetitions"] + settings["warmups"],
             "Performance trial count exceeds its plan.")
    numbers = {"seconds", "action_seconds", "first_output_seconds", "first_answer_seconds", "first_reasoning_seconds",
               "prompt_seconds", "decode_seconds", "prompt_tokens_per_second", "decode_tokens_per_second",
               "end_to_end_tokens_per_second"}
    counts = {"input_tokens", "generated_tokens", "prompt_metric_tokens", "decode_metric_tokens"}
    for index, trial in enumerate(trials):
        _require(isinstance(trial, dict) and type(trial.get("index")) is int and trial["index"] == index + 1
                 and type(trial.get("warmup")) is bool and trial["warmup"] == (index < settings["warmups"])
                 and trial.get("status") in ("completed", "failed", "cancelled"), "Invalid performance trial outcome.")
        for key in numbers | counts:
            number = trial.get(key)
            _require(number is None or (_integer(number) if key in counts else _number(number)),
                     f"Invalid performance measurement: {key}.")
        for key in ("text", "thinking", "error"):
            _require(key not in trial or _string(trial[key], 256 * 1024), "Invalid benchmark output text.")
        arrivals = trial.get("chunk_arrivals", [])
        _require(isinstance(arrivals, list) and len(arrivals) <= 4096, "Invalid benchmark arrival trace.")
        previous = -1
        for arrival in arrivals:
            _require(isinstance(arrival, dict) and _number(arrival.get("seconds")) and arrival["seconds"] >= previous,
                     "Invalid benchmark arrival time.")
            previous = arrival["seconds"]
            for key in ("answer_characters", "reasoning_characters", "raw_characters"):
                _require(key not in arrival or _integer(arrival[key]), "Invalid benchmark arrival coverage.")
    aggregates = value["summary"].get("aggregates", {})
    _require(isinstance(aggregates, dict), "Invalid performance aggregates.")
    for metric in aggregates.values():
        _require(isinstance(metric, dict) and set(metric) == {"count", "mean", "median", "sd", "min", "max"}
                 and _integer(metric["count"], 100), "Invalid performance aggregate samples.")
        _require(all(number is None or _number(number) for name, number in metric.items() if name != "count"),
                 "Invalid performance aggregate value.")
    measured = [trial for trial in trials if not trial["warmup"]]
    summary = value["summary"]
    expected = {"planned_trials": settings["repetitions"] + settings["warmups"], "attempted_trials": len(trials),
                "planned_measured_trials": settings["repetitions"],
                "completed_measured_trials": sum(trial["status"] == "completed" for trial in measured),
                "failed_measured_trials": sum(trial["status"] == "failed" for trial in measured),
                "unattempted_trials": settings["repetitions"] + settings["warmups"] - len(trials)}
    _require(all(type(summary.get(key)) is int and summary[key] == number for key, number in expected.items()),
             "Performance summary does not match its trial outcomes.")
    _warnings(value["warnings"])


def _comparison(value, *, graph=False):
    _require(isinstance(value, dict) and {"summary", "rows", "warnings"} <= value.keys()
             and _string(value["summary"]), "Invalid research comparison artifact.")
    rows = value["rows"]
    _require(isinstance(rows, list) and len(rows) <= 500000, "Invalid comparison row inventory.")
    for row in rows:
        _require(isinstance(row, dict) and _string(row.get("name"), 4096) and _string(row.get("change"), 4096),
                 "Invalid comparison row.")
        for prefix in ("left", "right"):
            shape = row.get(prefix + "_shape")
            _require(shape is None or isinstance(shape, list) and 1 <= len(shape) <= 4 and
                     all(_integer(dim, 2**40, 0 if graph else 1) for dim in shape), "Invalid comparison tensor shape.")
            for suffix in ("dtype", "op"):
                item = row.get(prefix + "_" + suffix)
                _require(item is None or _string(item, 4096), "Invalid comparison tensor type/operation.")
            count = row.get(prefix + "_bytes")
            _require(count is None or _integer(count), "Invalid comparison storage size.")
            inputs = row.get(prefix + "_inputs")
            _require(inputs is None or isinstance(inputs, list) and len(inputs) <= 1024
                     and all(_string(item, 4096) for item in inputs), "Invalid comparison input references.")
    if graph:
        _require(_string(value.get("left_graph"), 4096) and _string(value.get("right_graph"), 4096),
                 "Invalid compared graph identities.")
    else:
        changes = value.get("metadata_changes", [])
        _require(isinstance(changes, list) and len(changes) <= 200000 and
                 all(isinstance(row, dict) and _string(row.get("key")) and {"left", "right"} <= row.keys()
                     for row in changes), "Invalid metadata comparison inventory.")
        for files in (value.get("left_files", []), value.get("right_files", [])):
            _require(isinstance(files, list) and len(files) <= 64, "Invalid compared file inventory.")
            for file in files:
                _require(isinstance(file, dict) and _string(file.get("path"), 4096)
                         and isinstance(file.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", file["sha256"])
                         and _integer(file.get("size_bytes")), "Invalid compared file identity.")
    _warnings(value["warnings"])


def validate_lab(value: dict) -> dict:
    """Validate a complete saved view and its artifacts without reading external data."""
    _document(value)
    keys = {"format", "version", "state", "inspection", "capture", "benchmark", "capabilities"}
    optional = {"model_comparison", "graph_comparison"}
    _require(keys <= value.keys() <= keys | optional and value["format"] == FORMAT and type(value["version"]) is int
             and value["version"] == VERSION, "Unsupported or incomplete LLM research lab document.")
    _state(value["state"])
    if value["inspection"] is not None:
        _inspection(value["inspection"])
    if value["capture"] is not None:
        from mlcookbook.llm_graph import validate_graph_capture

        validate_graph_capture(value["capture"])
    if value["benchmark"] is not None:
        _benchmark(value["benchmark"])
    capabilities = value["capabilities"]
    if capabilities is not None:
        required = {"base_url", "observed_utc", "connection", "capabilities", "warnings"}
        _require(isinstance(capabilities, dict) and required <= capabilities.keys()
                 and _string(capabilities["base_url"], 2048) and _string(capabilities["observed_utc"], 128)
                 and isinstance(capabilities["connection"], dict) and isinstance(capabilities["capabilities"], dict),
                 "Invalid server capability artifact.")
        for item in capabilities["capabilities"].values():
            _require(isinstance(item, dict) and item.get("status") in ("available", "unavailable", "not_tested")
                     and _string(item.get("reason")), "Invalid server capability observation.")
        _warnings(capabilities["warnings"])
    if value.get("model_comparison") is not None:
        _comparison(value["model_comparison"])
    if value.get("graph_comparison") is not None:
        _comparison(value["graph_comparison"], graph=True)
    return value


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Lab JSON contains duplicate keys.")
        value[key] = item
    return value


def _decode(raw):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite lab number.")))
    except (UnicodeError, RecursionError, ValueError) as exc:
        raise ValueError("The lab archive contains invalid bounded JSON.") from exc


def _document(value):
    if not isinstance(value, dict):
        raise ValueError("A lab document must be a JSON object.")
    try:
        raw = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("A lab document must contain finite JSON data only.") from exc
    if len(raw) > MAX_DOCUMENT:
        raise ValueError("The lab document exceeds the 64 MiB archive budget.")
    return raw


def save_lab(path: str | Path, document: dict) -> str:
    """Save one self-contained lab snapshot, leaving a prior save intact on failure."""
    validate_lab(document)
    raw = _document(document)
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.dumps({"format": FORMAT, "version": VERSION, "document_sha256": digest,
                           "document_bytes": len(raw)}, allow_nan=False).encode("utf-8")
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", manifest)
                archive.writestr("document.json", raw)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return digest


def load_lab(path: str | Path) -> dict:
    """Read checked JSON only. Never open source models, servers or executable code."""
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE:
        raise ValueError("The lab archive exceeds the 80 MiB file budget.")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if sorted(value.filename for value in members) != ["document.json", "manifest.json"]:
                raise ValueError("A lab archive must contain exactly its manifest and document.")
            for member in members:
                limit = MAX_DOCUMENT if member.filename == "document.json" else 8192
                if member.file_size > limit or member.flag_bits & 1:
                    raise ValueError("A lab archive member exceeds its budget or is encrypted.")
            manifest = _decode(archive.read("manifest.json"))
            if not isinstance(manifest, dict) or set(manifest) != {"format", "version", "document_sha256", "document_bytes"}:
                raise ValueError("The lab manifest is invalid.")
            if manifest["format"] != FORMAT or type(manifest["version"]) is not int or manifest["version"] != VERSION:
                raise ValueError("Unsupported lab archive format/version.")
            raw = archive.read("document.json")
            if type(manifest["document_bytes"]) is not int or manifest["document_bytes"] != len(raw):
                raise ValueError("The lab document byte count does not match.")
            if manifest["document_sha256"] != hashlib.sha256(raw).hexdigest():
                raise ValueError("The lab document checksum does not match.")
            document = _decode(raw)
            return validate_lab(document)
    except (zipfile.BadZipFile, RuntimeError, KeyError) as exc:
        raise ValueError("The selected file is not a valid LLM lab archive.") from exc


def export_lab(path: str | Path, document: dict) -> str:
    """Atomically publish the same validated, self-contained lab as plain JSON."""
    validate_lab(document)
    raw = _document(document)
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()
