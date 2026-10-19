"""Local Ollama discovery and bounded text generation using standard-library HTTP.

No runtime installation, model pulls, PyTorch imports, or server ownership.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from mlcookbook._local_http import _check, _json, _object, _response, local_url
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.llm import _inputs, _integer, _number

DEFAULT_URL = "http://127.0.0.1:11434"
MAX_LINE = 256 * 1024


@dataclass(frozen=True)
class OllamaModel(Result):
    name: str
    digest: str | None
    size_bytes: int | None
    details: dict
    metadata: dict


@dataclass(frozen=True)
class OllamaConnection(Result):
    base_url: str
    version: str | None
    models: list[OllamaModel]
    warnings: list[str]


@dataclass(frozen=True)
class OllamaOutput(Result):
    row_position: int | None
    input_text: str
    submitted_prompt: str
    messages: list[dict[str, str]]
    text: str
    thinking: str
    input_tokens: int | None
    generated_tokens: int | None
    seconds: float
    server_seconds: float | None
    generation_seconds: float | None
    load_seconds: float | None
    finish_reason: str | None


@dataclass(frozen=True)
class OllamaGenerationResult(Result):
    method: str
    backend: str
    base_url: str
    model: OllamaModel
    prompt: str
    text_column: str | None
    prompt_format: str
    settings: dict
    versions: dict[str, str | None]
    outputs: list[OllamaOutput]
    warnings: list[str]


def local_ollama_url(value: str) -> str:
    """Accept only explicit loopback HTTP endpoints; no proxies or redirects."""
    return local_url(value, api_path="/api")


def _remote(value):
    return bool(value.get("remote_host") or value.get("remote_model")
                or str(value.get("name", "")).endswith(":cloud"))


def ollama_models(base_url: str = DEFAULT_URL, *, stop_requested: Callable[[], bool] | None = None) -> OllamaConnection:
    """Test the connection and list installed local models without loading weights."""
    base_url = local_ollama_url(base_url)
    version = _json(base_url, "/api/version", stop_requested=stop_requested).get("version")
    data = _json(base_url, "/api/tags", stop_requested=stop_requested).get("models")
    if not isinstance(data, list):
        raise ValueError("Ollama model listing is missing its models array.")
    models, warnings = [], []
    for value in data:
        if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not value["name"]:
            raise ValueError("Ollama returned an invalid model entry.")
        if _remote(value):
            warnings.append(f"Excluded cloud model {value['name']!r}; this connector runs installed local models.")
            continue
        models.append(OllamaModel(value["name"], value.get("digest"), _count(value.get("size")),
                                  value.get("details") if isinstance(value.get("details"), dict) else {}, {}))
    if not models:
        warnings.append("No installed local models were found. Add a model using Ollama, then refresh.")
    return OllamaConnection(base_url, version if isinstance(version, str) else None,
                            sorted(models, key=lambda model: model.name), warnings)


def describe_ollama_model(base_url: str, model: str, *, connection: OllamaConnection | None = None,
                          stop_requested: Callable[[], bool] | None = None) -> OllamaModel:
    """Inspect an installed model; never pull a missing model or execute a cloud reference."""
    base_url = local_ollama_url(base_url)
    connection = connection or ollama_models(base_url, stop_requested=stop_requested)
    if connection.base_url != base_url or not isinstance(model, str):
        raise ValueError("The selected model does not match this Ollama connection.")
    selected = next((entry for entry in connection.models if entry.name == model), None)
    if selected is None:
        raise ValueError(f"Local model {model!r} is not installed on this Ollama server. Refresh the model list.")
    metadata = _json(base_url, "/api/show", {"model": model, "verbose": False}, stop_requested=stop_requested)
    if _remote(metadata):
        raise ValueError("This model refers to a cloud runtime. Choose an installed local model.")
    capabilities = metadata.get("capabilities", [])
    if capabilities and "completion" not in capabilities:
        raise ValueError("This Ollama model does not support text completion.")
    details = metadata.get("details") if isinstance(metadata.get("details"), dict) else selected.details
    # Keep useful inference metadata; licenses and verbose vocabulary arrays are not run state.
    kept = {key: metadata[key] for key in ("capabilities", "thinking", "model_info", "parameters", "template",
                                          "renderer", "parser", "template_source", "modified_at") if key in metadata}
    return OllamaModel(selected.name, selected.digest, selected.size_bytes, details, kept)


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _seconds(value):
    return value / 1_000_000_000 if _count(value) is not None else None


def generate_ollama(
    dataset: Dataset | None = None,
    *,
    model: str,
    prompt: str,
    base_url: str = DEFAULT_URL,
    text_column: str | None = None,
    max_rows: int = 1,
    max_new_tokens: int = 128,
    context_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.9,
    random_state: int = 0,
    prompt_format: str = "auto",
    thinking: str = "off",
    max_seconds: float = 30.0,
    response_schema: dict | None = None,
    stop_requested: Callable[[], bool] | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> OllamaGenerationResult:
    """Stream a bounded local request; callbacks receive transient per-output previews.

    ``plain`` uses raw generation; ``auto``/``chat`` let Ollama render its model's
    chat template. Context allocation is requested from the server; this client
    cannot verify its input truncation or inspect the exact rendered chat prompt.
    The request deadline includes server model loading. Closing an interrupted
    request does not stop the Ollama service. Callbacks must be thread-safe.
    """
    base_url = local_ollama_url(base_url)
    _integer(max_rows, "max_rows", 1, 10)
    _integer(max_new_tokens, "max_new_tokens", 1, 512)
    _integer(context_tokens, "context_tokens", 256, 32768)
    _integer(random_state, "random_state", 0, 2147483647)
    _number(temperature, "temperature", 0.0, 2.0)
    _number(top_p, "top_p", 0.01, 1.0)
    _number(max_seconds, "max_seconds", 1.0, 120.0)
    if not isinstance(model, str) or not model.strip() or len(model) > 512:
        raise ValueError("Choose an installed Ollama model.")
    if prompt_format not in ("auto", "chat", "plain") or thinking not in ("default", "off", "on"):
        raise ValueError("Choose a supported prompt format and thinking option.")
    if any(callback is not None and not callable(callback) for callback in (stop_requested, on_progress)):
        raise ValueError("Generation callbacks must be callable or None.")
    if max_new_tokens >= context_tokens:
        raise ValueError("The output token limit must be smaller than the requested context budget.")
    rows, warnings = _inputs(dataset, prompt, text_column, max_rows)
    if response_schema is not None and not isinstance(response_schema, dict):
        raise ValueError("response_schema must be a JSON schema object or None.")
    connection = ollama_models(base_url, stop_requested=stop_requested)
    selected = describe_ollama_model(base_url, model, connection=connection, stop_requested=stop_requested)
    warnings += connection.warnings
    options = {"num_predict": max_new_tokens, "num_ctx": context_tokens, "temperature": temperature, "seed": random_state}
    if temperature > 0:
        options["top_p"] = top_p
    think = None
    if thinking != "default":
        thinking_metadata = selected.metadata.get("thinking", {})
        supported = thinking_metadata.get("values", []) if isinstance(thinking_metadata, dict) else []
        capable = "thinking" in selected.metadata.get("capabilities", [])
        if capable:
            think = thinking == "on"
            if supported and not any(type(value) is bool and value == think for value in supported):
                raise ValueError("This model does not support the selected boolean thinking option. Choose Model default.")
        elif thinking == "on":
            raise ValueError("This model does not advertise thinking support. Choose Off or Model default.")
    outputs = []
    for index, (position, text) in enumerate(rows):
        _check(stop_requested)
        combined = prompt.strip() + "\n\n" + text if text_column and prompt.strip() else text
        messages = [{"role": "user", "content": combined}] if prompt_format != "plain" else []
        body = {"model": model, "stream": True, "options": options}
        body.update({"messages": messages} if messages else {"prompt": combined, "raw": True})
        if think is not None:
            body["think"] = think
        if response_schema is not None:
            body["format"] = response_schema
        started = monotonic()
        text_parts, thinking_parts, final = [], [], None
        with _response(base_url, "/api/chat" if messages else "/api/generate", body, max_seconds, stop_requested) as response:
            while True:
                _check(stop_requested)
                raw = response.readline(MAX_LINE + 1)
                if not raw:
                    break
                if len(raw) > MAX_LINE:
                    raise ValueError("Ollama stream event exceeds the size limit.")
                if not raw.strip():
                    continue
                event = _object(raw)
                message = event.get("message", {}) if messages else event
                if not isinstance(message, dict):
                    raise ValueError("Ollama stream contains an invalid message.")
                content = message.get("content" if messages else "response", "")
                thought = message.get("thinking", "")
                if not isinstance(content, str) or not isinstance(thought, str):
                    raise ValueError("Ollama stream text must be a string.")
                text_parts.append(content)
                thinking_parts.append(thought)
                current, reasoning = "".join(text_parts), "".join(thinking_parts)
                if len(current) + len(reasoning) > MAX_LINE:
                    raise ValueError("Ollama generated text exceeds the response size limit.")
                if on_progress is not None and (content or thought):
                    on_progress({"output_index": index + 1, "row_position": position, "text": current, "thinking": reasoning})
                if event.get("done") is True:
                    final = event
                    break
            if final is None:
                _check(stop_requested, started + max_seconds)
                raise ValueError("Ollama stream ended without a completed response. Partial text was not committed.")
        if final.get("model") not in (None, model):
            raise ValueError("Ollama returned a different model identity than requested.")
        outputs.append(OllamaOutput(position, text, combined, messages, "".join(text_parts), "".join(thinking_parts),
                                     _count(final.get("prompt_eval_count")), _count(final.get("eval_count")),
                                     monotonic() - started, _seconds(final.get("total_duration")),
                                     _seconds(final.get("eval_duration")), _seconds(final.get("load_duration")),
                                     final.get("done_reason") if isinstance(final.get("done_reason"), str) else None))
    warnings += ["Ollama manages tokenization and context. Input truncation and the exact rendered chat prompt cannot be verified by this client.",
                 "Generation options are requests to the server. Applied device/precision are unavailable; seeds do not guarantee identical results across hardware or runtime versions.",
                 "Generated text may contain factual errors. No labeled evaluation was performed.",
                 "Saved experiments retain outputs and model references; reruns need the local Ollama server and matching model."]
    if any(output.generated_tokens is None or output.input_tokens is None for output in outputs):
        warnings.append("The server did not report every token count; missing usage metrics are unavailable.")
    return OllamaGenerationResult("generate_ollama", "ollama", base_url, selected, prompt, text_column,
                                   "chat" if prompt_format != "plain" else "plain",
                                   {"requested_options": options, "thinking": thinking, "max_rows": max_rows,
                                    "max_seconds": max_seconds, "max_new_tokens": max_new_tokens,
                                    **({"response_schema": response_schema} if response_schema is not None else {})},
                                   {"ollama": connection.version}, outputs, warnings)
