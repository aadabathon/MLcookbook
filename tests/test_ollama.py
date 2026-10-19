"""Ollama contract tests use HTTP and never require installed runtimes/models."""

import json
import threading
import time
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from mlcookbook import Dataset
from mlcookbook.ollama import describe_ollama_model, generate_ollama, local_ollama_url, ollama_models


def test_ollama_discovery_filters_cloud_and_describes_model(ollama_server):
    ollama_server.state["cloud"] = True
    connection = ollama_models(ollama_server.base_url)
    assert connection.version == "test-0.40"
    assert [model.name for model in connection.models] == ["test:tiny"]
    assert "cloud" in connection.warnings[0]
    model = describe_ollama_model(ollama_server.base_url, "test:tiny", connection=connection)
    assert model.digest == "a" * 64
    assert model.metadata["model_info"]["llama.context_length"] == 4096
    json.dumps(connection.to_dict(), allow_nan=False)


@pytest.mark.parametrize("chunked", [False, True])
def test_ollama_streams_fragmented_unicode_and_records_usage(ollama_server, chunked):
    ollama_server.state["chunked"] = chunked
    events = []
    result = generate_ollama(model="test:tiny", prompt="hello", base_url=ollama_server.base_url, on_progress=events.append)
    output = result.outputs[0]
    assert output.text == "Hi 🌍" and output.input_tokens == 3 and output.generated_tokens == 2
    assert output.generation_seconds == 1 and output.load_seconds == 0.5 and output.server_seconds == 2
    assert output.messages == [{"role": "user", "content": "hello"}]
    assert output.finish_reason == "stop" and result.model.digest == "a" * 64
    assert result.versions == {"ollama": "test-0.40"}
    assert events[-1]["text"] == "Hi 🌍"
    request = next(body for path, body in ollama_server.requests if path == "/api/chat")
    assert request["think"] is False and request["stream"] is True
    assert request["options"] == {"num_predict": 128, "num_ctx": 4096, "temperature": 0, "seed": 0}
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["outputs"][0]["text"] == "Hi 🌍"
    with pytest.raises(FrozenInstanceError):
        result.backend = "pytorch"


def test_ollama_raw_prompt_and_sample_options(ollama_server):
    result = generate_ollama(model="test:tiny", prompt="hello", base_url=ollama_server.base_url,
                             prompt_format="plain", temperature=0.8, top_p=0.7, thinking="default")
    request = next(body for path, body in ollama_server.requests if path == "/api/generate")
    assert request["raw"] is True and request["options"]["top_p"] == 0.7
    assert "think" not in request and result.outputs[0].messages == []


def test_ollama_batch_uses_original_positions(ollama_server):
    data = Dataset(pd.DataFrame({"text": ["one", None, "", "two", "three"]}, index=[5] * 5))
    result = generate_ollama(data, model="test:tiny", prompt="answer", text_column="text", max_rows=2,
                             base_url=ollama_server.base_url)
    assert [output.row_position for output in result.outputs] == [0, 3]
    assert result.outputs[1].submitted_prompt == "answer\n\ntwo"
    assert any("Skipped 2" in warning for warning in result.warnings)
    assert any("first 2" in warning for warning in result.warnings)


@pytest.mark.parametrize("stream,match", [
    (b'{"message":{"content":"partial"},"done":false}\n', "without a completed"),
    (b'{bad}\n', "invalid UTF-8 JSON"),
    (b'{"error":"out of memory"}\n', "out of memory"),
    (b'[]\n', "not an object"),
    (b'{"message":{"content":1},"done":true}\n', "must be a string"),
    (b'{"model":"other","done":true}\n', "different model"),
])
def test_ollama_rejects_broken_streams(ollama_server, stream, match):
    ollama_server.state["raw_stream"] = stream
    with pytest.raises(ValueError, match=match):
        generate_ollama(model="test:tiny", prompt="hello", base_url=ollama_server.base_url)


def test_ollama_missing_usage_stays_unavailable(ollama_server):
    ollama_server.state["usage"] = False
    result = generate_ollama(model="test:tiny", prompt="hello", base_url=ollama_server.base_url)
    assert result.outputs[0].generated_tokens is None and result.outputs[0].load_seconds is None
    assert any("unavailable" in warning for warning in result.warnings)


def test_ollama_server_errors_are_clear(ollama_server):
    ollama_server.state["status"] = 500
    with pytest.raises(ValueError, match="HTTP 500: model failed"):
        generate_ollama(model="test:tiny", prompt="hello", base_url=ollama_server.base_url)


@pytest.mark.parametrize("url", ["https://ollama.com", "http://192.168.1.1:11434", "http://user:pw@localhost:11434",
                                  "http://localhost:0", "http://127.0.0.1/redirect", "http://127.0.0.1?token=secret"])
def test_ollama_urls_are_loopback_only(url):
    with pytest.raises(ValueError, match="loopback"):
        local_ollama_url(url)


def test_ollama_normalizes_local_url():
    assert local_ollama_url("http://localhost:11434/api/") == "http://127.0.0.1:11434"
    assert local_ollama_url("http://[::1]:11434") == "http://[::1]:11434"


@pytest.mark.parametrize("remote", [False, True])
def test_ollama_does_not_pull_missing_or_cloud_models(ollama_server, remote):
    ollama_server.state["show_remote"] = remote
    with pytest.raises(ValueError, match="cloud runtime" if remote else "not installed"):
        generate_ollama(model="test:tiny" if remote else "missing", prompt="hello", base_url=ollama_server.base_url)
    assert not any(path in ("/api/chat", "/api/generate", "/api/pull") for path, _ in ollama_server.requests)


def test_ollama_cancel_interrupts_loading_without_stream_data(ollama_server):
    ollama_server.state["hold"] = True
    stopped = threading.Event()
    timer = threading.Thread(target=lambda: (ollama_server.started.wait(3), stopped.set()), daemon=True)
    timer.start()
    started = time.monotonic()
    with pytest.raises(InterruptedError, match="cancelled"):
        generate_ollama(model="test:tiny", prompt="hello", base_url=ollama_server.base_url,
                        stop_requested=stopped.is_set, max_seconds=5)
    assert time.monotonic() - started < 2
    timer.join(timeout=1)


def test_ollama_deadline_includes_model_loading(ollama_server):
    ollama_server.state["hold"] = True
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="time limit"):
        generate_ollama(model="test:tiny", prompt="hello", base_url=ollama_server.base_url, max_seconds=1)
    assert time.monotonic() - started < 2.5


def test_ollama_rejects_unsupported_thinking_options(ollama_server):
    ollama_server.state["thinking_values"] = ["low", "high"]
    with pytest.raises(ValueError, match="Model default"):
        generate_ollama(model="test:tiny", prompt="hello", base_url=ollama_server.base_url, thinking="off")


def test_ollama_import_and_http_do_not_require_pytorch(ollama_server):
    import subprocess
    import sys

    script = "import sys; from mlcookbook.ollama import generate_ollama; " \
             "generate_ollama(model='test:tiny',prompt='hello',base_url=sys.argv[1]); " \
             "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", script, ollama_server.base_url],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
