"""Read-only descriptor, capability and measurement contracts."""

import hashlib
import json
import struct
import threading
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from mlcookbook.llm_lab import benchmark_llamacpp, compare_gguf, inspect_gguf, probe_llamacpp


def _string(value, endian="<"):
    raw = value.encode("utf-8")
    return struct.pack(endian + "Q", len(raw)) + raw


def _file(path, *, dtype=0, shape=(2, 3), metadata=None, offset=0, endian="<", data_size=None):
    metadata = {"general.architecture": "llama", **(metadata or {})}
    raw = b"GGUF" + struct.pack(endian + "IQQ", 3, 1, len(metadata))
    for key, value in metadata.items():
        raw += _string(key, endian)
        if isinstance(value, str):
            raw += struct.pack(endian + "I", 8) + _string(value, endian)
        elif isinstance(value, list):
            raw += struct.pack(endian + "IIQ", 9, 4, len(value))
            raw += struct.pack(endian + "I" * len(value), *value)
        else:
            raw += struct.pack(endian + "II", 4, value)
    raw += _string("blk.0.attn_q.weight", endian) + struct.pack(endian + "I", len(shape))
    raw += struct.pack(endian + "Q" * len(shape), *shape) + struct.pack(endian + "IQ", dtype, offset)
    raw += b"\0" * (-len(raw) % metadata.get("general.alignment", 32))
    raw += b"\0" * (data_size if data_size is not None else 24 + offset)
    path.write_bytes(raw)
    return path


@pytest.mark.parametrize("endian", ["<", ">"])
def test_inspection_shapes_bytes_and_identity_without_loading_weights(tmp_path, endian):
    path = _file(tmp_path / "tiny.gguf", endian=endian)
    result = inspect_gguf(path)
    assert result.summary["architecture"] == "llama"
    assert result.summary["tensor_count"] == 1 and result.summary["elements"] == 6
    assert result.summary["storage_bytes"] == 24
    assert result.tensors[0]["shape"] == [2, 3]
    assert result.tensors[0]["dtype"] == "F32" and result.tensors[0]["group"] == "Layer 0"
    assert result.files[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.files[0]["byte_order"] == ("little" if endian == "<" else "big")
    assert any("not a captured" in value for value in result.warnings)
    json.dumps(result.to_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        result.path = "changed"


def test_quantized_storage_uses_blocks_and_unknown_layout_is_explicit(tmp_path):
    path = _file(tmp_path / "quant.gguf", dtype=12, shape=(256, 2), data_size=288)
    result = inspect_gguf(path)
    assert result.tensors[0]["bytes"] == 288
    assert result.tensors[0]["block_elements"] == 256
    unknown = inspect_gguf(_file(tmp_path / "future.gguf", dtype=999))
    assert unknown.tensors[0]["bytes"] is None
    assert unknown.summary["unknown_storage_tensors"] == 1
    assert unknown.summary["dtype_bytes"]["Unknown (999)"] is None
    assert any("unsupported" in item for item in unknown.warnings)


def test_exact_aliases_count_logical_descriptors_and_storage_separately(tmp_path):
    metadata = _string("general.architecture") + struct.pack("<I", 8) + _string("llama")
    raw = b"GGUF" + struct.pack("<IQQ", 3, 2, 1) + metadata
    for name in ("token_embd.weight", "output.weight"):
        raw += _string(name) + struct.pack("<IQQIQ", 2, 2, 3, 0, 0)
    raw += b"\0" * (-len(raw) % 32) + b"\0" * 24
    path = tmp_path / "aliases.gguf"
    path.write_bytes(raw)
    result = inspect_gguf(path)
    assert result.summary["elements"] == 12
    assert result.summary["storage_bytes"] == 24
    assert result.summary["dtype_bytes"]["F32"] == 24
    assert result.summary["dtype_counts"]["F32"] == 2
    assert any("share an exact storage range" in warning for warning in result.warnings)


@pytest.mark.parametrize("options,match", [
    ({"dtype": 12, "shape": (255, 2), "data_size": 1024}, "quantization block"),
    ({"offset": 1}, "not aligned"),
    ({"shape": (0, 2)}, "shape is invalid"),
    ({"shape": (2, 3), "data_size": 1}, "outside the file"),
    ({"metadata": {"general.alignment": 3}}, "power of two"),
])
def test_invalid_tensor_descriptors_fail_before_any_result(tmp_path, options, match):
    with pytest.raises(ValueError, match=match):
        inspect_gguf(_file(tmp_path / "bad.gguf", **options))


def test_truncated_header_unknown_version_and_bounded_strings(tmp_path):
    path = tmp_path / "bad.gguf"
    for raw, match in ((b"GGUF\x03", "truncated"),
                       (b"GGUF" + struct.pack("<I", 99), "versions"),
                       (b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 20 * 1024 * 1024), "string exceeds")):
        path.write_bytes(raw)
        with pytest.raises(ValueError, match=match):
            inspect_gguf(path)


def test_large_metadata_arrays_keep_exact_counts_and_small_previews(tmp_path):
    result = inspect_gguf(_file(tmp_path / "vocab.gguf", metadata={"tokenizer.ggml.token_type": list(range(100))}))
    tokens = result.metadata["tokenizer.ggml.token_type"]
    assert tokens["count"] == 100 and tokens["omitted"] == 68 and len(tokens["preview"]) == 32


def test_cancel_and_file_mutation_preserve_no_inspection(tmp_path):
    path = _file(tmp_path / "change.gguf")
    with pytest.raises(InterruptedError):
        inspect_gguf(path, stop_requested=lambda: True)
    def mutate(event):
        if event["stage"] == "hash":
            with path.open("ab") as file:
                file.write(b"change")
    with pytest.raises(ValueError, match="changed during"):
        inspect_gguf(path, on_progress=mutate)


def test_explicit_shard_selection_never_follows_other_paths(tmp_path):
    first = _file(tmp_path / "one.gguf", metadata={"split.count": 2, "split.no": 0})
    result = inspect_gguf(first)
    assert result.summary["file_count"] == 1
    assert any("Incomplete" in item for item in result.warnings)
    second = _file(tmp_path / "two.gguf", metadata={"split.count": 2, "split.no": 1})
    complete = inspect_gguf([first, second])
    assert complete.summary["file_count"] == 2
    assert not any("Incomplete" in item for item in complete.warnings)
    with pytest.raises(ValueError, match="distinct"):
        inspect_gguf([first, first])


def test_descriptor_comparison_is_offline_and_does_not_claim_equal_weights(tmp_path):
    first = inspect_gguf(_file(tmp_path / "one.gguf"))
    second = inspect_gguf(_file(tmp_path / "two.gguf", metadata={"general.name": "second"}))
    for path in tmp_path.glob("*.gguf"):
        path.unlink()
    result = compare_gguf(first, second)
    assert result["counts"]["unchanged"] == 1 and result["rows"] == []
    assert result["metadata_changes"] == [{"key": "general.name", "left": None, "right": "second",
                                            "left_present": False, "right_present": True}]
    assert result["left_files"][0]["sha256"] != result["right_files"][0]["sha256"]
    assert any("numerical weights" in warning for warning in result["warnings"])
    json.dumps(result, allow_nan=False)


def test_descriptor_comparison_reports_shape_add_remove_and_ambiguity(tmp_path):
    from copy import deepcopy

    left = inspect_gguf(_file(tmp_path / "one.gguf")).to_dict()
    right = inspect_gguf(_file(tmp_path / "two.gguf", shape=(3, 2))).to_dict()
    changed = compare_gguf(left, right)
    assert changed["rows"][0]["change"] == "changed"
    assert changed["rows"][0]["left_shape"] == [2, 3] and changed["rows"][0]["right_shape"] == [3, 2]
    right["tensors"][0]["name"] = "output.weight"
    names = compare_gguf(left, right)
    assert names["counts"]["added"] == names["counts"]["removed"] == 1
    ambiguous = deepcopy(left)
    ambiguous["tensors"].append(deepcopy(ambiguous["tensors"][0]))
    ambiguous["summary"]["tensor_count"] = 2
    result = compare_gguf(left, ambiguous)
    assert result["counts"]["ambiguous"] == 1
    assert result["rows"][0]["right_shape"] is None
    assert any("list order" in warning for warning in result["warnings"])


def test_capability_discovery_never_generates_or_opens_server_paths(llamacpp_server):
    result = probe_llamacpp(llamacpp_server.base_url)
    assert result.connection["models"][0]["name"] == "test-gguf"
    assert result.capabilities["slots"]["status"] == "unavailable"
    assert result.capabilities["chat_generation"]["status"] == "not_tested"
    assert result.capabilities["compute_graph"]["status"] == "unavailable"
    assert not any(path in ("/completion", "/v1/chat/completions") for path, _ in llamacpp_server.requests)
    json.dumps(result.to_dict(), allow_nan=False)


def test_capability_probe_requires_a_valid_body_not_just_http_200(llamacpp_server, monkeypatch):
    import io
    from contextlib import contextmanager

    @contextmanager
    def response(*args):
        yield io.BytesIO(b"<html>not a capability</html>")

    monkeypatch.setattr("mlcookbook.llm_lab._response", response)
    result = probe_llamacpp(llamacpp_server.base_url)
    assert all(result.capabilities[name]["status"] == "unavailable"
               for name in ("slots", "metrics", "tokenize", "apply_template"))


def test_benchmark_retains_warmups_and_separates_missing_measurements(llamacpp_server):
    result = benchmark_llamacpp(base_url=llamacpp_server.base_url, model="test-gguf", prompt="hello", repetitions=2, warmups=1)
    assert len(result.trials) == 3 and result.trials[0]["warmup"]
    assert all(trial["status"] == "completed" for trial in result.trials)
    assert result.summary["completed_measured_trials"] == 2
    trial = result.trials[1]
    assert trial["prompt_seconds"] == 0.05 and trial["decode_seconds"] == 0.2
    assert trial["decode_tokens_per_second"] is None  # no matching backend predicted_n in the fixture
    if trial["seconds"] > 0:
        assert trial["end_to_end_tokens_per_second"] == pytest.approx(2 / trial["seconds"])
    else:
        assert trial["end_to_end_tokens_per_second"] is None
    assert trial["first_answer_seconds"] <= trial["action_seconds"]
    assert trial["chunk_arrivals"][0]["reasoning_characters"] > 0
    assert result.summary["aggregates"]["seconds"]["count"] == 2
    assert result.summary["aggregates"]["decode_tokens_per_second"]["count"] == 0
    json.dumps(result.to_dict(), allow_nan=False)


def test_benchmark_cancellation_returns_completed_and_cancelled_trials(llamacpp_server):
    stopped = threading.Event()
    def progress(event):
        if event.get("outcome", {}).get("index") == 1:
            stopped.set()
    result = benchmark_llamacpp(base_url=llamacpp_server.base_url, model="test-gguf", prompt="hello", repetitions=4,
                               warmups=0, stop_requested=stopped.is_set, on_progress=progress)
    assert [trial["status"] for trial in result.trials] == ["completed", "cancelled"]
    assert result.summary["completed_measured_trials"] == 1
    assert result.summary["unattempted_trials"] == 2 and result.summary["cancelled"]
    assert result.summary["aggregates"]["seconds"]["count"] == 1


def test_benchmark_failures_are_counted_and_missing_usage_not_zero(llamacpp_server):
    llamacpp_server.state["usage"] = False
    result = benchmark_llamacpp(base_url=llamacpp_server.base_url, model="test-gguf", prompt="hello", warmups=0, repetitions=1)
    assert result.trials[0]["generated_tokens"] is None
    assert result.trials[0]["end_to_end_tokens_per_second"] is None
    llamacpp_server.state["status"] = 500
    failed = benchmark_llamacpp(base_url=llamacpp_server.base_url, model="test-gguf", prompt="hello", warmups=0, repetitions=2)
    assert failed.summary["failed_measured_trials"] == 2
    assert failed.summary["aggregates"]["seconds"]["count"] == 0


def test_trial_deadline_includes_discovery_and_setup(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("mlcookbook.llm_lab.monotonic", lambda: clock[0])

    def delayed(**params):
        clock[0] += 2
        params["stop_requested"]()
        pytest.fail("An expired whole-trial deadline must stop setup.")

    monkeypatch.setattr("mlcookbook.llamacpp.generate_llamacpp", delayed)
    result = benchmark_llamacpp(base_url="http://127.0.0.1:1", model="tiny", prompt="hello", repetitions=1,
                               warmups=0, max_seconds=1)
    assert result.trials[0]["status"] == "failed"
    assert "time limit" in result.trials[0]["error"]
    assert result.summary["completed_measured_trials"] == 0


@pytest.fixture
def native_server():
    state = {"applied": True, "terminal": True, "wrong_model": False, "after_terminal": False}
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def respond(self, raw, content_type="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            values = {"/health": {"status": "ok"}, "/v1/models": {"data": [{"id": "tiny"}]},
                      "/props": {"default_generation_settings": {"n_ctx": 2048}, "build_info": "test-native"}}
            self.respond(json.dumps(values[self.path]).encode())
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if self.path == "/apply-template":
                self.respond(json.dumps({"prompt": "<user>hello<assistant>"}).encode())
                return
            frames = [{"content": "answer", "stop": False}]
            if state["terminal"]:
                final = {"content": "", "stop": True, "stop_type": "eos", "model": "other" if state["wrong_model"] else "tiny",
                         "timings": {"prompt_n": 4, "prompt_ms": 20, "predicted_n": 2, "predicted_ms": 100},
                         "tokens_cached": 3 if body["cache_prompt"] else 0}
                if state["applied"]:
                    final["generation_settings"] = {"cache_prompt": body["cache_prompt"]}
                frames.append(final)
                if state["after_terminal"]:
                    frames.append({"content": "late", "stop": False})
            self.respond(b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames), "text/event-stream")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    yield SimpleNamespace(base_url=f"http://127.0.0.1:{server.server_port}", requests=requests, state=state)
    server.shutdown()
    server.server_close()
    thread.join(2)


@pytest.mark.parametrize("policy", ["disabled", "reuse"])
def test_native_benchmark_controls_cache_and_matches_count_duration(native_server, policy):
    result = benchmark_llamacpp(base_url=native_server.base_url, model="tiny", prompt="hello", repetitions=2, warmups=0,
                               cache_policy=policy)
    trial = result.trials[0]
    assert trial["status"] == "completed" and trial["cache_verified"]
    assert trial["rendered_prompt"] == "<user>hello<assistant>"
    assert trial["decode_tokens_per_second"] == 20
    assert trial["prompt_tokens_per_second"] == 200
    assert trial["first_output_seconds"] is not None
    assert trial["first_answer_seconds"] is None  # raw native text can include thinking markup
    assert trial["first_reasoning_seconds"] is None
    assert trial["chunk_arrivals"][0]["raw_characters"] > 0
    assert next(body for path, body in native_server.requests if path == "/completion")["cache_prompt"] == (policy == "reuse")
    assert result.summary["aggregates"]["decode_tokens_per_second"]["median"] == 20


def test_native_missing_applied_policy_is_unverified_and_incomplete_stream_fails(native_server):
    native_server.state["applied"] = False
    result = benchmark_llamacpp(base_url=native_server.base_url, model="tiny", prompt="hello", repetitions=1, warmups=0, cache_policy="reuse")
    assert not result.trials[0]["cache_verified"]
    native_server.state["terminal"] = False
    failed = benchmark_llamacpp(base_url=native_server.base_url, model="tiny", prompt="hello", repetitions=1, warmups=0, cache_policy="disabled")
    assert failed.trials[0]["status"] == "failed"
    assert "terminal" in failed.trials[0]["error"]


@pytest.mark.parametrize("change,match", [({"wrong_model": True}, "different model"),
                                         ({"after_terminal": True}, "after its terminal")])
def test_native_identity_changes_and_post_terminal_content_fail(native_server, change, match):
    native_server.state.update(change)
    result = benchmark_llamacpp(base_url=native_server.base_url, model="tiny", prompt="hello", repetitions=1,
                               warmups=0, cache_policy="disabled")
    assert result.trials[0]["status"] == "failed"
    assert match in result.trials[0]["error"]


@pytest.mark.parametrize("params", [{"repetitions": True}, {"warmups": 11}, {"max_new_tokens": 0}, {"temperature": float("nan")},
                                    {"cache_policy": "pretend-cold"}, {"prompt": ""}])
def test_benchmark_rejects_invalid_plans_before_network(params):
    with pytest.raises(ValueError):
        benchmark_llamacpp(**{"base_url": "http://127.0.0.1:1", "model": "tiny", "prompt": "hello", **params})
