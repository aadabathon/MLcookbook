"""Bounded text generation through an existing, single-model llama.cpp server."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter

from mlcookbook._local_http import LocalHTTPError, _check, _json, _object, _response, local_url
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.llm import _inputs, _integer, _number

DEFAULT_URL = "http://127.0.0.1:8080"
MAX_EVENT = 256 * 1024


@dataclass(frozen=True)
class LlamaCppModel(Result):
    name: str
    metadata: dict


@dataclass(frozen=True)
class LlamaCppConnection(Result):
    base_url: str
    status: str
    models: list[LlamaCppModel]
    properties: dict
    version: str | None
    warnings: list[str]


@dataclass(frozen=True)
class LlamaCppOutput(Result):
    row_position: int | None
    input_text: str
    submitted_prompt: str
    messages: list[dict[str, str]]
    text: str
    thinking: str
    input_tokens: int | None
    generated_tokens: int | None
    seconds: float
    finish_reason: str
    usage: dict
    timings: dict
    applied_settings: dict


@dataclass(frozen=True)
class LlamaCppGenerationResult(Result):
    method: str
    backend: str
    base_url: str
    model: LlamaCppModel
    server_properties: dict
    local_file: dict | None
    prompt: str
    text_column: str | None
    settings: dict
    versions: dict
    outputs: list[LlamaCppOutput]
    warnings: list[str]


def local_llamacpp_url(value: str) -> str:
    return local_url(value, api_path="/v1")


def llamacpp_models(base_url: str = DEFAULT_URL, *, stop_requested: Callable[[], bool] | None = None) -> LlamaCppConnection:
    """Inspect health, loaded model and read-only properties; never load/download a model."""
    base_url = local_llamacpp_url(base_url)
    try:
        health = _json(base_url, "/health", stop_requested=stop_requested)
    except LocalHTTPError as exc:
        if exc.status == 503:
            return LlamaCppConnection(base_url, "loading", [], {}, None,
                                      ["llama.cpp is loading its model. Refresh when loading finishes."])
        raise
    if health.get("status") != "ok":
        raise ValueError("llama.cpp did not report a ready health status.")
    entries = _json(base_url, "/v1/models", stop_requested=stop_requested).get("data")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("Connect directly to a single-model llama-server. Router mode and model loading are not supported.")
    entry = entries[0]
    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
        raise ValueError("llama.cpp returned an invalid model identity.")
    state = entry.get("status", {})
    if isinstance(state, dict) and state.get("value") not in (None, "loaded", "sleeping"):
        raise ValueError("The listed model is not loaded. Load it outside ML Cookbook, then refresh.")
    try:
        properties = _json(base_url, "/props", stop_requested=stop_requested)
    except LocalHTTPError as exc:
        if exc.status == 404:
            raise ValueError("llama.cpp /props is unavailable. Connect directly to llama-server, rather than a generic compatible API.") from exc
        raise
    if not isinstance(properties.get("default_generation_settings"), dict):
        raise ValueError("The server does not expose llama.cpp generation properties. Router mode is unsupported.")
    architecture = entry.get("architecture", {})
    modalities = architecture.get("output_modalities", []) if isinstance(architecture, dict) else []
    if modalities and "text" not in modalities:
        raise ValueError("The loaded llama.cpp model does not advertise text output.")
    # No filesystem access follows server-provided paths. A local file is an explicit opt-in.
    kept = {key: properties[key] for key in ("default_generation_settings", "total_slots", "model_path",
                                           "chat_template", "chat_template_caps", "modalities", "is_sleeping",
                                           "build_info", "model_ftype", "model_alias") if key in properties}
    version = properties.get("build_info")
    if isinstance(version, dict):
        version = " · ".join(str(version[key]) for key in ("version", "build", "commit") if key in version) or None
    warnings = ["Server model references are unverified identities; the API does not provide a full model fingerprint."]
    if not isinstance(version, str):
        version = None
        warnings.append("The server did not expose a runtime version; it is unavailable.")
    status = "sleeping" if properties.get("is_sleeping") is True else "ready"
    return LlamaCppConnection(base_url, status, [LlamaCppModel(entry["id"], entry)], kept, version, warnings)


def _file_fingerprint(path, stop_requested):
    if path in (None, ""):
        return None
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("The selected local GGUF file does not exist.")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        before = path.stat()
        if file.read(4) != b"GGUF":
            raise ValueError("The selected local file is not GGUF.")
        file.seek(0)
        while block := file.read(8 * 1024 * 1024):
            _check(stop_requested)
            digest.update(block)
        after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("The selected GGUF file changed while computing its fingerprint.")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": after.st_size,
            "scope": "Selected file only; binding to server-loaded weights and additional shards is unverified."}


def _sse(response):
    """Parse SSE events, including comments, CRLF, multiline data and fragmented UTF-8."""
    lines, size = [], 0
    while True:
        raw = response.readline(MAX_EVENT + 1)
        if not raw:
            if lines:
                raise ValueError("llama.cpp stream ended inside an SSE event.")
            return
        size += len(raw)
        if size > MAX_EVENT:
            raise ValueError("llama.cpp SSE event exceeds the size limit.")
        line = raw.rstrip(b"\r\n")
        if not line:
            if lines:
                yield b"\n".join(lines)
            lines, size = [], 0
        elif line.startswith(b"data:"):
            value = line[5:]
            lines.append(value[1:] if value.startswith(b" ") else value)


def _count(value):
    return value if type(value) is int and value >= 0 else None


def timing_seconds(value):
    """Convert a reported millisecond duration, retaining unavailable values."""
    return value / 1000 if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def generate_llamacpp(
    dataset: Dataset | None = None, *, model: str, prompt: str, base_url: str = DEFAULT_URL,
    text_column: str | None = None, max_rows: int = 1, max_new_tokens: int = 128,
    temperature: float = 0.0, top_p: float = 0.9, random_state: int = 0,
    max_seconds: float = 90.0, gguf_path: str | Path | None = None,
    response_schema: dict | None = None,
    stop_requested: Callable[[], bool] | None = None, on_progress: Callable[[dict], None] | None = None,
) -> LlamaCppGenerationResult:
    """Stream one user-message completion per input through llama.cpp's SSE API.

    Controls are request settings. Context, GPU offload and model loading belong
    to the external server. No custom templates, tools, downloads or process control.
    """
    base_url = local_llamacpp_url(base_url)
    _integer(max_rows, "max_rows", 1, 10)
    _integer(max_new_tokens, "max_new_tokens", 1, 512)
    _integer(random_state, "random_state", 0, 2147483647)
    _number(temperature, "temperature", 0.0, 2.0)
    _number(top_p, "top_p", 0.01, 1.0)
    _number(max_seconds, "max_seconds", 1.0, 120.0)
    if not isinstance(model, str) or not model.strip() or len(model) > 512:
        raise ValueError("Choose the loaded llama.cpp model.")
    if any(callback is not None and not callable(callback) for callback in (stop_requested, on_progress)):
        raise ValueError("Generation callbacks must be callable or None.")
    rows, warnings = _inputs(dataset, prompt, text_column, max_rows)
    if response_schema is not None and not isinstance(response_schema, dict):
        raise ValueError("response_schema must be a JSON schema object or None.")
    connection = llamacpp_models(base_url, stop_requested=stop_requested)
    if not connection.models:
        raise ValueError("llama.cpp is still loading its model. Refresh when loading finishes.")
    selected = connection.models[0]
    if selected.name != model:
        raise ValueError("The loaded llama.cpp model changed. Refresh and choose its current identity.")
    local_file = _file_fingerprint(gguf_path, stop_requested)
    options = {"max_tokens": max_new_tokens, "temperature": temperature, "seed": random_state}
    if temperature > 0:
        options["top_p"] = top_p
    outputs = []
    for index, (position, text) in enumerate(rows):
        _check(stop_requested)
        combined = prompt.strip() + "\n\n" + text if text_column and prompt.strip() else text
        messages = [{"role": "user", "content": combined}]
        body = {"model": model, "messages": messages, "stream": True,
                "stream_options": {"include_usage": True}, **options}
        if response_schema is not None:
            body["response_format"] = {"type": "json_object", "schema": response_schema}
        started = monotonic()
        measured_started = perf_counter()
        answer, thinking, usage, timings, applied = "", "", {}, {}, {}
        finish_reason, done, completion_id = None, False, None
        with _response(base_url, "/v1/chat/completions", body, max_seconds, stop_requested) as response:
            for payload in _sse(response):
                _check(stop_requested, started + max_seconds)
                if payload == b"[DONE]":
                    done = True
                    break
                event = _object(payload)
                if event.get("model") not in (None, model):
                    raise ValueError("llama.cpp returned a different model identity than requested.")
                if event.get("id") is not None:
                    if completion_id is not None and event["id"] != completion_id:
                        raise ValueError("llama.cpp changed completion identity inside the stream.")
                    completion_id = event["id"]
                for key, target in (("usage", usage), ("timings", timings), ("generation_settings", applied)):
                    value = event.get(key)
                    if value is not None:
                        if not isinstance(value, dict):
                            raise ValueError(f"llama.cpp returned invalid {key} metadata.")
                        target.update(value)
                choices = event.get("choices")
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ValueError("llama.cpp returned invalid completion choices.")
                if not choices:
                    continue  # Usage-only terminal chunk.
                choice = choices[0]
                if not isinstance(choice, dict) or choice.get("index") != 0:
                    raise ValueError("llama.cpp returned an unexpected completion choice.")
                delta = choice.get("delta", {})
                if not isinstance(delta, dict) or delta.get("tool_calls") or delta.get("function_call"):
                    raise ValueError("llama.cpp returned unsupported completion content or tool calls.")
                content, thought = delta.get("content"), delta.get("reasoning_content")
                content = "" if content is None else content
                thought = "" if thought is None else thought
                if not isinstance(content, str) or not isinstance(thought, str):
                    raise ValueError("llama.cpp stream text must be a string.")
                if finish_reason is not None and (content or thought):
                    raise ValueError("llama.cpp sent text after the completion finished.")
                answer += content
                thinking += thought
                if len(answer) + len(thinking) > MAX_EVENT:
                    raise ValueError("llama.cpp generated text exceeds the response size limit.")
                if on_progress is not None and (content or thought):
                    on_progress({"output_index": index + 1, "row_position": position, "text": answer, "thinking": thinking})
                reason = choice.get("finish_reason")
                if reason is not None:
                    if reason not in ("stop", "length"):
                        raise ValueError(f"Unsupported llama.cpp stop reason: {reason!r}.")
                    finish_reason = reason
        if not done or finish_reason is None:
            raise ValueError("llama.cpp stream ended without a completed response. Partial text was not committed.")
        outputs.append(LlamaCppOutput(position, text, combined, messages, answer, thinking,
                                      _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens")),
                                      perf_counter() - measured_started, finish_reason, usage, timings, applied))
    warnings += connection.warnings + [
        "llama.cpp manages chat formatting, tokenization, context and device settings. Exact rendered prompts and input truncation are unverified.",
        "Generation settings are requests; only explicitly returned applied settings are recorded. Seeds do not guarantee identical output across runtimes or hardware.",
        "Generated text may contain factual errors. No labeled evaluation was performed.",
        "Saved outputs open without a server. Reruns require the external llama-server and its model.",
    ]
    if local_file:
        warnings.append(local_file["scope"])
    if any(output.generated_tokens is None or output.input_tokens is None for output in outputs):
        warnings.append("The server did not report every token count; missing usage metrics are unavailable.")
    if any(not output.text for output in outputs):
        warnings.append("At least one completion returned no visible answer; inspect reasoning and the stop reason.")
    return LlamaCppGenerationResult("generate_llamacpp", "llamacpp", base_url, selected, connection.properties,
                                     local_file, prompt, text_column, {"requested_options": options, "max_rows": max_rows,
                                     "max_seconds": max_seconds,
                                     **({"response_schema": response_schema} if response_schema is not None else {})},
                                     {"llama.cpp": connection.version}, outputs, warnings)
