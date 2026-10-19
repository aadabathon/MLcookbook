"""llama.cpp HTTP/SSE contracts, without model/runtime installations."""

import hashlib
import json
import subprocess
import sys
import threading
import time
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from mlcookbook import Dataset
from mlcookbook.llamacpp import generate_llamacpp, llamacpp_models, local_llamacpp_url


@pytest.mark.parametrize("chunked", [False, True])
def test_llamacpp_fragmented_multiline_sse_usage_and_provenance(llamacpp_server, chunked):
    server = llamacpp_server
    server.state["chunked"] = chunked
    events = []
    connection = llamacpp_models(server.base_url)
    assert connection.status == "ready" and connection.version == "b-test"
    assert connection.models[0].metadata["meta"]["n_ctx"] == 2048
    json.dumps(connection.to_dict(), allow_nan=False)
    result = generate_llamacpp(model="test-gguf", prompt="hello", base_url=server.base_url, on_progress=events.append)
    output = result.outputs[0]
    assert output.text == "Hi 🌍" and output.thinking == "A thought"
    assert output.input_tokens == 3 and output.generated_tokens == 2
    assert output.finish_reason == "stop" and output.timings["predicted_ms"] == 200
    assert output.applied_settings == {"temperature": 0}
    assert result.local_file is None  # Never read the model_path supplied by the server.
    assert result.versions == {"llama.cpp": "b-test"}
    assert events[-1]["text"] == output.text
    request = next(body for path, body in server.requests if path == "/v1/chat/completions")
    assert request == {"model": "test-gguf", "messages": [{"role": "user", "content": "hello"}],
                       "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 128, "temperature": 0, "seed": 0}
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["outputs"][0]["text"] == "Hi 🌍"
    with pytest.raises(FrozenInstanceError):
        result.backend = "ollama"


def test_llamacpp_explicit_file_hash_and_rows(llamacpp_server, tmp_path):
    model = tmp_path / "selected.gguf"
    model.write_bytes(b"GGUFtest-data-for-fingerprinting-only")
    data = Dataset(pd.DataFrame({"text": ["one", None, "", "two", "three"]}, index=[5] * 5))
    result = generate_llamacpp(data, model="test-gguf", prompt="Summarize", text_column="text", max_rows=2,
                               base_url=llamacpp_server.base_url, gguf_path=model, temperature=0.7, top_p=0.8)
    assert [output.row_position for output in result.outputs] == [0, 3]
    assert result.outputs[1].submitted_prompt == "Summarize\n\ntwo"
    assert result.local_file["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert result.settings["requested_options"]["top_p"] == 0.8
    assert "unverified" in result.local_file["scope"]
    assert any("Skipped 2" in warning for warning in result.warnings)
    assert any("first 2" in warning for warning in result.warnings)


def test_llamacpp_loading_and_changed_model_never_generate(llamacpp_server):
    server = llamacpp_server
    server.state["loading"] = True
    result = llamacpp_models(server.base_url)
    assert result.status == "loading" and result.models == []
    with pytest.raises(ValueError, match="still loading"):
        generate_llamacpp(model="test-gguf", prompt="hello", base_url=server.base_url)
    server.state["loading"] = False
    with pytest.raises(ValueError, match="model changed"):
        generate_llamacpp(model="missing", prompt="hello", base_url=server.base_url)
    assert all(body is None for path, body in server.requests)


@pytest.mark.parametrize("change,match", [
    ({"models": []}, "single-model"),
    ({"models": [{"id": "one"}, {"id": "two"}]}, "single-model"),
    ({"models": [{"id": "one", "status": {"value": "cached"}}]}, "not loaded"),
    ({"props": {}}, "properties"),
    ({"models": [{"id": "one", "architecture": {"output_modalities": ["embeddings"]}}]}, "text output"),
])
def test_llamacpp_rejects_router_and_non_generation_servers(llamacpp_server, change, match):
    llamacpp_server.state.update(change)
    with pytest.raises(ValueError, match=match):
        llamacpp_models(llamacpp_server.base_url)


@pytest.mark.parametrize("stream,match", [
    (b'data: {bad}\n\n', "invalid UTF-8 JSON"),
    (b'data: {"choices":[]}\n\n', "without a completed"),
    (b'data: [DONE]\n\n', "without a completed"),
    (b'data: {"choices":[]}', "inside an SSE event"),
    (b'data: {"error":{"message":"out of memory"}}\n\n', "out of memory"),
    (b'data: {"model":"other","choices":[]}\n\n', "different model"),
    (b'data: {"choices":[{"index":0,"delta":{"content":0}}]}\n\n', "must be a string"),
    (b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{}]}}]}\n\n', "tool calls"),
    (b'data: {"choices":[{"index":0,"finish_reason":"tool_calls"}]}\n\n', "stop reason"),
    (b'data: ' + b'x' * (256 * 1024) + b'\n\n', "size limit"),
], ids=["invalid-json", "incomplete", "done-only", "unfinished-event", "backend-error", "wrong-model",
         "wrong-text-type", "tool-call", "tool-finish", "oversized-event"])
def test_llamacpp_rejects_incomplete_or_invalid_streams(llamacpp_server, stream, match):
    llamacpp_server.state["raw_stream"] = stream
    with pytest.raises(ValueError, match=match):
        generate_llamacpp(model="test-gguf", prompt="hello", base_url=llamacpp_server.base_url)


def test_llamacpp_missing_usage_and_version_stay_unavailable(llamacpp_server):
    llamacpp_server.state["usage"] = False
    llamacpp_server.state["props"].pop("build_info")
    result = generate_llamacpp(model="test-gguf", prompt="hello", base_url=llamacpp_server.base_url)
    assert result.outputs[0].generated_tokens is None and result.versions["llama.cpp"] is None
    assert any("missing usage" in warning for warning in result.warnings)


@pytest.mark.parametrize("status", [400, 401, 500])
def test_llamacpp_http_errors_preserve_message(llamacpp_server, status):
    llamacpp_server.state["status"] = status
    with pytest.raises(ValueError, match=f"HTTP {status}: request rejected"):
        generate_llamacpp(model="test-gguf", prompt="hello", base_url=llamacpp_server.base_url)


@pytest.mark.parametrize("after_preview", [False, True])
def test_llamacpp_cancel_idle_read_before_or_during_stream(llamacpp_server, after_preview):
    server = llamacpp_server
    server.state["hold_after_preview" if after_preview else "hold"] = True
    stopped = threading.Event()
    timer = threading.Thread(target=lambda: (server.started.wait(3), stopped.set()) if not after_preview else None, daemon=True)
    timer.start()
    with pytest.raises(InterruptedError, match="cancelled"):
        generate_llamacpp(model="test-gguf", prompt="hello", base_url=server.base_url, stop_requested=stopped.is_set,
                           on_progress=lambda event: stopped.set(), max_seconds=5)
    timer.join(timeout=1)


def test_llamacpp_deadline_includes_idle_load(llamacpp_server):
    llamacpp_server.state["hold"] = True
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="time limit"):
        generate_llamacpp(model="test-gguf", prompt="hello", base_url=llamacpp_server.base_url, max_seconds=1)
    assert time.monotonic() - started < 2.5


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "http://10.0.0.1:8080", "http://user:pw@localhost:8080",
                                  "http://127.0.0.1/other", "http://localhost:0", "http://localhost?key=secret"])
def test_llamacpp_loopback_only(url):
    with pytest.raises(ValueError, match="loopback"):
        local_llamacpp_url(url)


def test_llamacpp_normalizes_v1_prefix():
    assert local_llamacpp_url("http://localhost:8080/v1/") == "http://127.0.0.1:8080"
    assert local_llamacpp_url("http://[::1]:8080") == "http://[::1]:8080"


def test_llamacpp_generates_without_torch_transformers_or_ollama_import(llamacpp_server):
    script = "import sys; from mlcookbook.llamacpp import generate_llamacpp; " \
             "generate_llamacpp(model='test-gguf',prompt='hello',base_url=sys.argv[1]); " \
             "assert not {'torch','transformers','mlcookbook.ollama'} & sys.modules.keys()"
    process = subprocess.run([sys.executable, "-c", script, llamacpp_server.base_url], capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
