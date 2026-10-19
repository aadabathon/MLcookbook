import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ollama_server():
    """An actual loopback HTTP server with controllable Ollama protocol responses."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from types import SimpleNamespace

    state = {"cloud": False, "show_remote": False, "thinking_values": [False, True],
             "status": 200, "raw_stream": None, "hold": False, "usage": True,
             "hold_after_preview": False, "chunked": False}
    requests, started, release = [], threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            requests.append((self.path, None))
            if self.path == "/api/version":
                self.respond({"version": "test-0.40"})
            elif self.path == "/api/tags":
                model = {"name": "test:tiny", "digest": "a" * 64, "size": 1234,
                         "details": {"family": "llama", "parameter_size": "1M", "quantization_level": "Q4_K_M"}}
                models = [model]
                if state["cloud"]:
                    models += [{"name": "remote:cloud", "remote_host": "https://ollama.com"}]
                self.respond({"models": models})
            else:
                self.respond({"error": "not found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if self.path == "/api/show":
                self.respond({"capabilities": ["completion", "thinking"],
                              "thinking": {"values": state["thinking_values"]},
                              "model_info": {"general.architecture": "llama", "general.parameter_count": 1000000,
                                             "llama.context_length": 4096},
                              "remote_host": "https://ollama.com" if state["show_remote"] else None})
                return
            if self.path not in ("/api/chat", "/api/generate"):
                self.respond({"error": "unexpected endpoint"}, 404)
                return
            started.set()
            if state["hold"]:
                release.wait(5)
            if state["status"] != 200:
                self.respond({"error": "model failed"}, state["status"])
                return
            final = {"done": True, "model": "test:tiny", "done_reason": "stop"}
            if state["usage"]:
                final.update(prompt_eval_count=3, eval_count=2, total_duration=2000000000,
                             eval_duration=1000000000, load_duration=500000000)
            content = {"message": {"content": "Hi 🌍"}} if self.path == "/api/chat" else {"response": "Hi 🌍"}
            raw = state["raw_stream"]
            if raw is None:
                raw = (json.dumps(dict(content, done=False), ensure_ascii=False) + "\n" + json.dumps(final) + "\n").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            if state["chunked"]:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                # Deliberately split UTF-8 code points and JSON across writes.
                for position in range(0, len(raw), 3):
                    chunk = raw[position:position + 3]
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n" if state["chunked"] else chunk)
                    self.wfile.flush()
                    if state["hold_after_preview"] and b"\n" in chunk:
                        release.wait(5)
                if state["chunked"]:
                    self.wfile.write(b"0\r\n\r\n")
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    yield SimpleNamespace(base_url=f"http://127.0.0.1:{server.server_port}", state=state, requests=requests,
                          started=started, release=release)
    release.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.fixture
def llamacpp_server():
    """A real HTTP/SSE transport with no llama.cpp installation or model dependency."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from types import SimpleNamespace

    state = {"loading": False, "status": 200, "raw_stream": None, "hold": False,
             "hold_after_preview": False, "usage": True, "chunked": True,
             "models": [{"id": "test-gguf", "meta": {"n_ctx": 2048, "ftype": "Q4_K", "n_params": 135000000}}],
             "props": {"default_generation_settings": {"n_ctx": 2048}, "total_slots": 1,
                       "model_path": "C:/untrusted/not-selected.gguf", "build_info": "b-test"}}
    requests, started, release = [], threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value, status=200):
            raw = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            requests.append((self.path, None))
            if self.path == "/health":
                self.respond({"error": {"message": "Loading model"}} if state["loading"] else {"status": "ok"},
                             503 if state["loading"] else 200)
            elif self.path == "/v1/models":
                self.respond({"data": state["models"]})
            elif self.path == "/props":
                self.respond(state["props"])
            else:
                self.respond({"error": {"message": "not found"}}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if self.path != "/v1/chat/completions":
                self.respond({"error": {"message": "unexpected endpoint"}}, 404)
                return
            started.set()
            if state["hold"]:
                release.wait(5)
            if state["status"] != 200:
                self.respond({"error": {"message": "request rejected"}}, state["status"])
                return
            first = {"id": "completion-1", "model": "test-gguf", "choices": [
                {"index": 0, "delta": {"content": "Hi 🌍", "reasoning_content": "A thought"}, "finish_reason": None}]}
            final = {"id": "completion-1", "model": "test-gguf", "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}],
                "timings": {"prompt_ms": 50, "predicted_ms": 200, "predicted_per_second": 10},
                "generation_settings": {"temperature": 0}}
            usage = {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
            # SSE comments, CRLF, and a JSON object spanning multiple data lines.
            data = json.dumps(first, ensure_ascii=False).replace(', "model"', ',\r\ndata: "model"', 1)
            frames = [(': keepalive\r\nevent: message\r\ndata: ' + data + '\r\n\r\n').encode("utf-8"),
                      ("data: " + json.dumps(final) + "\n\n").encode()]
            if state["usage"]:
                frames.append(("data: " + json.dumps(usage) + "\n\n").encode())
            frames.append(b"data: [DONE]\n\n")
            if state["raw_stream"] is not None:
                frames = [state["raw_stream"]]
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if state["chunked"]:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for index, frame in enumerate(frames):
                    stride = 8192 if len(frame) > 100000 else 3
                    for offset in range(0, len(frame), stride):
                        chunk = frame[offset:offset + stride]
                        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n" if state["chunked"] else chunk)
                        self.wfile.flush()
                    if index == 0 and state["hold_after_preview"]:
                        release.wait(5)
                if state["chunked"]:
                    self.wfile.write(b"0\r\n\r\n")
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    yield SimpleNamespace(base_url=f"http://127.0.0.1:{server.server_port}", state=state, requests=requests,
                          started=started, release=release)
    release.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.fixture(scope="session")
def tiny_llm(tmp_path_factory):
    """A real, random, tiny local checkpoint: tests never download model weights."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    directory = tmp_path_factory.mktemp("tiny-llm")
    vocabulary = {word: i for i, word in enumerate([
        "[PAD]", "[UNK]", "[EOS]", "hello", "world", "answer", "question", "explain", "data", "training",
        "one", "two", "three", "four", "five", "user", "assistant", "why", "is", "a",
    ])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    tokenizer.save_pretrained(directory)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_positions=64,
                                         n_layer=1, n_head=2, n_embd=16, eos_token_id=2, pad_token_id=0))
        model.save_pretrained(directory, safe_serialization=True)
    return directory


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """Small frame covering every VarType, with missing values and a constant."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "income": rng.normal(50_000, 10_000, n),
            "city": rng.choice(["Oslo", "Bergen", "Trondheim"], n),
            "is_member": rng.choice([True, False], n),
            "signup": pd.date_range("2024-01-01", periods=n, freq="D"),
            "notes": [f"free text entry number {i}" for i in range(n)],
            "constant": ["same"] * n,
        }
    )
    df.loc[:4, "age"] = np.nan
    return df


@pytest.fixture
def linear_df() -> pd.DataFrame:
    """Frame with a known linear relationship: y = 3*x1 - 2*x2 + noise."""
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    group = rng.choice(["a", "b"], n)
    y = 3 * x1 - 2 * x2 + rng.normal(0, 0.1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "group": group, "y": y})


@pytest.fixture
def logistic_df() -> pd.DataFrame:
    """Frame with a known logistic relationship: P(y=yes) = sigmoid(2*x1 - 1*x2)."""
    rng = np.random.default_rng(1)
    n = 400
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    group = rng.choice(["a", "b"], n)
    p = 1 / (1 + np.exp(-(2 * x1 - x2)))
    y = np.where(rng.random(n) < p, "yes", "no")
    return pd.DataFrame({"x1": x1, "x2": x2, "group": group, "y": y})
