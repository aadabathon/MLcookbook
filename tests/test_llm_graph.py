"""Portable graph integrity and owned-process contracts; no model download."""

import hashlib
import json
import os
import sys
import threading
from copy import deepcopy
from pathlib import Path
from time import monotonic

import pytest

import mlcookbook.llm_graph as graphs


def sample_capture():
    nodes = [
        {"id": "weight", "name": "blk.0.weight", "op": "NONE", "dtype": "q4_K",
         "shape": [256, 2, 1, 1], "strides": [144, 144, 288, 288], "bytes": 288, "inputs": []},
        {"id": "input", "name": "input", "op": "NONE", "dtype": "f32",
         "shape": [256, 4, 1, 1], "strides": [4, 1024, 4096, 4096], "bytes": 4096, "inputs": []},
        {"id": "output", "name": "blk.0.output", "op": "MUL_MAT", "dtype": "f32",
         "shape": [2, 4, 1, 1], "strides": [4, 8, 32, 32], "bytes": 32,
         "inputs": ["weight", "input"], "input_slots": [0, 1], "observed": True},
    ]
    decode = deepcopy(nodes)
    decode[1]["shape"][1], decode[2]["shape"][1] = 1, 1
    return {"format": graphs.FORMAT, "version": 1, "provenance": {"upstream_commit": graphs.UPSTREAM_COMMIT},
            "graphs": [{"id": "prefill-0", "phase": "prefill", "input_token_ids": [1, 2, 3, 4], "nodes": nodes},
                       {"id": "decode-0", "phase": "decode", "input_token_ids": [5], "nodes": decode}],
            "warnings": ["Separate CPU execution; not a connected server trace."]}


def test_validated_capture_roundtrip_retains_shapes_slots_and_independence(tmp_path):
    value = sample_capture()
    value["graphs"][0]["nodes"][1].update(shape=[0, 1, 1, 1], bytes=0)
    value["graphs"][0]["nodes"][2].update(view_source="input", view_offset_bytes=0)
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    restored = graphs.load_graph_capture(path)
    assert restored == value
    assert restored["graphs"][0]["nodes"][1]["shape"][0] == 0  # Real hybrid state graphs contain empty tensors.
    restored["graphs"][0]["nodes"][0]["shape"][0] = 1
    assert value["graphs"][0]["nodes"][0]["shape"][0] == 256


@pytest.mark.parametrize("change,match", [
    (lambda c: c.update(version=True), "format/version"),
    (lambda c: c.update(version=2), "format/version"),
    (lambda c: c.update(warnings="bad"), "warnings"),
    (lambda c: c["graphs"].append(deepcopy(c["graphs"][0])), "Duplicate captured graph"),
    (lambda c: c["graphs"][0].update(phase="unknown"), "phase"),
    (lambda c: c["graphs"][0]["nodes"].append(deepcopy(c["graphs"][0]["nodes"][0])), "Duplicate node"),
    (lambda c: c["graphs"][0]["nodes"][0].update(shape=[-1]), "strides|dimension"),
    (lambda c: c["graphs"][0]["nodes"][0].update(bytes=True), "storage span"),
    (lambda c: c["graphs"][0]["nodes"][0].update(inputs=["absent"]), "missing input"),
    (lambda c: c["graphs"][0]["nodes"][0].update(inputs=["weight"]), "self dependency"),
    (lambda c: c["graphs"][0]["nodes"][0].update(inputs=["output"]), "cycle"),
    (lambda c: c["graphs"][0]["nodes"][2].update(input_slots=[0, 0]), "unique"),
    (lambda c: c["graphs"][0]["nodes"][2].update(input_slots=[{}, 1]), "port slot"),
    (lambda c: c["provenance"].update(helper={"sha256": "incorrect"}), "SHA-256"),
])
def test_malformed_dependency_and_identity_contracts_are_rejected(change, match):
    value = sample_capture()
    change(value)
    with pytest.raises(ValueError, match=match):
        graphs.validate_graph_capture(value)


def test_import_budgets_invalid_json_and_cancellation(tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    for raw in ('{"format":1,"format":2}', '{"bad":NaN}', '{"bad":'):
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(ValueError, match="UTF-8 JSON"):
            graphs.load_graph_capture(path)
    path.write_bytes(b'\xff')
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        graphs.load_graph_capture(path)
    monkeypatch.setattr(graphs, "MAX_CAPTURE_BYTES", 16)
    path.write_bytes(b' ' * 17)
    with pytest.raises(ValueError, match="budget"):
        graphs.load_graph_capture(path)
    with pytest.raises(InterruptedError):
        graphs.validate_graph_capture(sample_capture(), stop_requested=lambda: True)


def test_node_budget_and_nonfinite_values_are_not_silently_accepted(monkeypatch):
    monkeypatch.setattr(graphs, "MAX_NODES", 2)
    with pytest.raises(ValueError, match="node budget"):
        graphs.validate_graph_capture(sample_capture())
    value = sample_capture()
    value["provenance"]["timing"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        graphs.validate_graph_capture(value)


def test_phase_diff_uses_names_and_reports_dependency_changes():
    value = sample_capture()
    # IDs are local to each phase, so different allocation ordering must not matter.
    for node in value["graphs"][1]["nodes"]:
        node["id"] = "decode-" + node["id"]
        node["inputs"] = ["decode-" + source for source in node["inputs"]]
    value["graphs"][1]["nodes"][2]["inputs"] = ["decode-input", "decode-weight"]
    result = graphs.compare_graphs(value)
    assert "3 uniquely named" in result["summary"]
    row = next(row for row in result["rows"] if row["name"] == "blk.0.output")
    assert "shape" in row["change"] and "input dependencies" in row["change"]
    assert row["left_shape"] == [2, 4, 1, 1] and row["right_shape"] == [2, 1, 1, 1]
    json.dumps(result, allow_nan=False)


def test_phase_diff_refuses_duplicate_name_alignment_and_reports_add_remove():
    value = sample_capture()
    value["graphs"][1]["nodes"][0]["name"] = "input"
    result = graphs.compare_graphs(value)
    assert any(row["change"] == "Ambiguous identity" for row in result["rows"])
    assert any("could not be aligned" in warning for warning in result["warnings"])
    assert any(row["name"] == "blk.0.weight" and row["change"] == "Removed" for row in result["rows"])
    with pytest.raises(ValueError, match="index"):
        graphs.compare_graphs(value, left=-1)


def test_owned_process_deadline_cancellation_and_log_budget(tmp_path, monkeypatch):
    real_popen, children = graphs.subprocess.Popen, []
    def tracked_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(graphs.subprocess, "Popen", tracked_popen)
    with pytest.raises(TimeoutError):
        graphs._run_process([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path,
                            stop_requested=None, deadline=monotonic() + .15)
    assert children[-1].poll() is not None
    cancelled = threading.Event()
    timer = threading.Timer(.15, cancelled.set)
    timer.start()
    try:
        with pytest.raises(InterruptedError):
            graphs._run_process([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path,
                                stop_requested=cancelled.is_set, deadline=monotonic() + 10)
    finally:
        timer.cancel()
    assert children[-1].poll() is not None
    monkeypatch.setattr(graphs, "MAX_LOG_BYTES", 100)
    with pytest.raises(ValueError, match="log budget"):
        graphs._run_process([sys.executable, "-c", "print('x'*1000)"], tmp_path,
                            stop_requested=None, deadline=monotonic() + 10)


def test_owned_process_failure_and_output_budget_are_actionable(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="specific failure"):
        graphs._run_process([sys.executable, "-c", "import sys; sys.stderr.write('specific failure');sys.exit(3)"],
                            tmp_path, stop_requested=None, deadline=monotonic() + 10)
    monkeypatch.setattr(graphs, "MAX_CAPTURE_BYTES", 10)
    output = tmp_path / "output.json"
    with pytest.raises(ValueError, match="output budget"):
        graphs._run_process([sys.executable, "-c", "from pathlib import Path;Path('output.json').write_text('x'*11)"],
                            tmp_path, stop_requested=None, deadline=monotonic() + 10, output_path=output)


@pytest.fixture
def capture_environment(tmp_path, monkeypatch):
    if os.name != "nt":
        pytest.skip("Initial capture orchestration is Windows-only; portable validation tested separately.")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    files = {name: b"known fake test DLL " + name.encode() for name in graphs.SUPPORTED_DLLS}
    for name, raw in files.items():
        (runtime / name).write_bytes(raw)
    (runtime / "llama-server.exe").write_bytes(b"version-check executable placeholder")
    helper = tmp_path / "capture.exe"
    helper.write_bytes(b"helper executable placeholder")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFfixture-model")
    monkeypatch.setattr(graphs, "SUPPORTED_DLLS", {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()})
    def run(arguments, work, **kwargs):
        if arguments[-1] == "--version":
            return "version: 0.5.0-dev (build 11193, commit 4e7481175)"
        Path(arguments[4]).write_text(json.dumps(sample_capture()), encoding="utf-8")
        return ""
    monkeypatch.setattr(graphs, "_run_process", run)
    return runtime, helper, model


def test_capture_records_helper_loaded_identity_and_exact_configuration(capture_environment):
    runtime, helper, model = capture_environment
    events = []
    result = graphs.capture_graph(model, helper_path=helper, runtime_dir=runtime, prompt="test", on_progress=events.append)
    provenance = result["provenance"]
    assert provenance["model_files"][0]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert provenance["helper"]["path"] == str(helper)
    assert "not the connected" in provenance["attribution"]
    assert provenance["prompt"] == "test" and provenance["requested_decode_steps"] == 1
    assert len(events) >= 3
    json.dumps(result, allow_nan=False)


def test_incompatible_runtime_is_rejected_before_helper_process(capture_environment, monkeypatch):
    runtime, helper, model = capture_environment
    (runtime / "llama.dll").write_bytes(b"different ABI")
    monkeypatch.setattr(graphs, "_run_process", lambda *a, **k: pytest.fail("ABI rejection must precede execution"))
    with pytest.raises(ValueError, match="different ABI"):
        graphs.capture_graph(model, helper_path=helper, runtime_dir=runtime, prompt="test")


def test_capture_rejects_changed_model_or_incomplete_phase_sequence(capture_environment, monkeypatch):
    runtime, helper, model = capture_environment
    def incomplete(arguments, work, **kwargs):
        if arguments[-1] == "--version":
            return "build 11193 commit 4e7481175"
        value = sample_capture()
        value["graphs"] = value["graphs"][:1]
        Path(arguments[4]).write_text(json.dumps(value), encoding="utf-8")
        return ""
    monkeypatch.setattr(graphs, "_run_process", incomplete)
    with pytest.raises(ValueError, match="sequence"):
        graphs.capture_graph(model, helper_path=helper, runtime_dir=runtime, prompt="test")
    def mutated(arguments, work, **kwargs):
        result = incomplete(arguments, work, **kwargs)
        if arguments[-1] != "--version":
            Path(arguments[4]).write_text(json.dumps(sample_capture()), encoding="utf-8")
            stat = model.stat()
            os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return result
    monkeypatch.setattr(graphs, "_run_process", mutated)
    with pytest.raises(ValueError, match="changed during"):
        graphs.capture_graph(model, helper_path=helper, runtime_dir=runtime, prompt="test")


@pytest.mark.parametrize("kwargs", [{"max_tokens": True}, {"max_tokens": 0}, {"max_tokens": 17},
                                    {"max_seconds": float("inf")}, {"prompt": " "}])
def test_capture_limits_are_explicit(kwargs):
    with pytest.raises(ValueError):
        graphs.capture_graph("unused.gguf", prompt=kwargs.pop("prompt", "hello"), **kwargs)


def test_default_helper_path_has_packaged_and_source_locations(monkeypatch):
    assert graphs.default_helper_path().name == "llama-graph-capture.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", "C:/example/bundle", raising=False)
    assert graphs.default_helper_path() == Path("C:/example/bundle/tools/llama_research/llama-graph-capture.exe")
