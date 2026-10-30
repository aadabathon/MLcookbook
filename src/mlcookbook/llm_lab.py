"""Read-only model inventory and bounded llama.cpp research measurements.

This submodule never starts a server or decodes model weights. Storage layouts
follow upstream gguf-py ``GGML_QUANT_SIZES``; unknown layouts stay unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import struct
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, perf_counter

from mlcookbook._local_http import _check, _json, _response
from mlcookbook.core.results import Result

MAX_HEADER = 128 * 1024 * 1024
MAX_TENSORS = 250_000
MAX_METADATA = 100_000
MAX_ARRAY = 2_000_000
MAX_STRING = 16 * 1024 * 1024

# Format descriptors, not quantization implementations. (name, elements/block, bytes/block)
# https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py
_STORAGE = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
}
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B",
            10: "Q", 11: "q", 12: "d"}


@dataclass(frozen=True)
class GGUFInspection(Result):
    path: str
    files: list[dict]
    metadata: dict
    tensors: list[dict]
    summary: dict
    warnings: list[str]


@dataclass(frozen=True)
class LlamaCppCapabilities(Result):
    base_url: str
    observed_utc: str
    connection: dict
    capabilities: dict
    warnings: list[str]


@dataclass(frozen=True)
class LlamaCppBenchmark(Result):
    method: str
    base_url: str
    model: str
    created_utc: str
    settings: dict
    trials: list[dict]
    summary: dict
    warnings: list[str]


class _GGUFReader:
    def __init__(self, file, size, stop_requested):
        self.file, self.size, self.stop_requested = file, size, stop_requested
        self.endian = "<"

    def read(self, count):
        _check(self.stop_requested)
        if count < 0 or self.file.tell() + count > min(self.size, MAX_HEADER):
            raise ValueError("GGUF header is truncated or exceeds the 128 MiB inspection budget.")
        value = self.file.read(count)
        if len(value) != count:
            raise ValueError("GGUF header is truncated.")
        return value

    def unpack(self, kind):
        return struct.unpack(self.endian + kind, self.read(struct.calcsize(kind)))[0]

    def string(self):
        count = self.unpack("Q")
        if count > MAX_STRING:
            raise ValueError("GGUF string exceeds the 16 MiB inspection limit.")
        try:
            return self.read(count).decode("utf-8")
        except UnicodeError as exc:
            raise ValueError("GGUF contains an invalid UTF-8 string.") from exc

    def value(self, kind, depth=0):
        if kind in _SCALARS:
            value = self.unpack(_SCALARS[kind])
            if kind == 7:
                if value not in (0, 1):
                    raise ValueError("GGUF boolean value is invalid.")
                return bool(value)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("GGUF metadata contains a nonfinite number.")
            return value
        if kind == 8:
            return self.string()
        if kind != 9 or depth > 2:
            raise ValueError("GGUF metadata uses an unsupported value type or array nesting.")
        element_type, count = self.unpack("I"), self.unpack("Q")
        if count > MAX_ARRAY:
            raise ValueError("GGUF metadata array exceeds the two-million-element inspection limit.")
        if element_type not in (*_SCALARS, 8, 9):
            raise ValueError("GGUF metadata array has an unknown element type.")
        if count <= 64:
            return [self.value(element_type, depth + 1) for _ in range(count)]
        preview = []
        for index in range(count):
            value = self.value(element_type, depth + 1)
            if index < 32:
                preview.append(value[:1024] if isinstance(value, str) else value)
        return {"count": count, "element_type": element_type, "preview": preview,
                "omitted": count - len(preview), "scope": "Metadata array preview; all elements were validated."}


def _group(name):
    layer = re.match(r"(?:blk|layers|layer)\.(\d+)\.", name)
    if layer:
        return "Layer " + layer[1]
    if name.startswith(("token_embd", "tok_embeddings", "position_embd")):
        return "Embeddings"
    if name.startswith(("output", "lm_head")):
        return "Output"
    return "Other"


def _stat_identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _inspect_file(path, stop_requested, on_progress):
    path = Path(path).expanduser().resolve()
    warnings = []
    with path.open("rb") as file:
        before = os.fstat(file.fileno())
        reader = _GGUFReader(file, before.st_size, stop_requested)
        if reader.read(4) != b"GGUF":
            raise ValueError("The selected file is not GGUF.")
        raw_version = reader.read(4)
        version = struct.unpack("<I", raw_version)[0]
        if version not in (2, 3):
            if struct.unpack(">I", raw_version)[0] in (2, 3):
                reader.endian = ">"
                version = struct.unpack(">I", raw_version)[0]
            else:
                raise ValueError("GGUF inspection supports format versions 2 and 3.")
        tensor_count, metadata_count = reader.unpack("Q"), reader.unpack("Q")
        if tensor_count > MAX_TENSORS or metadata_count > MAX_METADATA:
            raise ValueError("GGUF tensor or metadata count exceeds the inspection budget.")
        metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            if not key or key in metadata:
                raise ValueError("GGUF metadata keys must be nonempty and unique.")
            metadata[key] = reader.value(reader.unpack("I"))
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or not 1 <= alignment <= 65536 or alignment & (alignment - 1):
            raise ValueError("GGUF alignment must be a power of two between 1 and 65536.")
        tensors, names = [], set()
        for index in range(tensor_count):
            name, dimensions = reader.string(), reader.unpack("I")
            if not name or name in names:
                raise ValueError("GGUF tensor names must be nonempty and unique within a file.")
            names.add(name)
            if not 1 <= dimensions <= 4:
                raise ValueError("GGUF tensor dimensions must be between one and four.")
            shape = [reader.unpack("Q") for _ in range(dimensions)]
            if any(not 1 <= value <= 2**40 for value in shape) or math.prod(shape) > 2**63 - 1:
                raise ValueError("GGUF tensor shape is invalid or exceeds the inspection limit.")
            type_id, offset = reader.unpack("I"), reader.unpack("Q")
            if offset % alignment:
                raise ValueError("GGUF tensor offset is not aligned.")
            dtype, block, block_bytes = _STORAGE.get(type_id, (f"Unknown ({type_id})", None, None))
            elements, stored = math.prod(shape), None
            if block is not None:
                if shape[0] % block:
                    raise ValueError(f"GGUF tensor {name!r} has a row incompatible with its quantization block.")
                stored = elements // block * block_bytes
            else:
                warnings.append(f"Storage type {type_id} is unsupported; bytes for {name!r} are unavailable.")
            tensors.append({"name": name, "shape": shape, "dtype": dtype, "type_id": type_id,
                            "elements": elements, "bytes": stored, "offset": offset, "group": _group(name),
                            "source_path": str(path), "block_elements": block, "block_bytes": block_bytes})
            if on_progress and index % 256 == 0:
                on_progress({"stage": "descriptors", "completed": index, "total": tensor_count})
        descriptor_end = file.tell()
        data_offset = (descriptor_end + alignment - 1) // alignment * alignment
        if tensor_count and data_offset > before.st_size:
            raise ValueError("GGUF tensor data section is missing.")
        ranges = []
        for tensor in tensors:
            absolute = data_offset + tensor["offset"]
            tensor["file_offset"] = absolute
            if absolute >= before.st_size or tensor["bytes"] is not None and absolute + tensor["bytes"] > before.st_size:
                raise ValueError(f"GGUF tensor {tensor['name']!r} extends outside the file.")
            if tensor["bytes"] is not None:
                ranges.append((absolute, absolute + tensor["bytes"]))
        ordered = sorted(set(ranges))
        if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("GGUF tensor storage ranges overlap inconsistently.")
        if len(ranges) != len(ordered):
            warnings.append("Some tensor descriptors share an exact storage range; logical counts include each descriptor.")
        file.seek(0)
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            _check(stop_requested)
            chunk = file.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise ValueError("The GGUF file changed during inspection.")
            remaining -= len(chunk)
            digest.update(chunk)
            if on_progress:
                on_progress({"stage": "hash", "completed": file.tell(), "total": before.st_size})
        if _stat_identity(before) != _stat_identity(os.fstat(file.fileno())) or _stat_identity(before) != _stat_identity(path.stat()):
            raise ValueError("The GGUF file changed during inspection.")
    info = {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": before.st_size,
            "version": version, "byte_order": "little" if reader.endian == "<" else "big",
            "alignment": alignment, "data_offset": data_offset, "header_bytes": descriptor_end, "metadata": metadata,
            "identity_scope": "Selected file only; binding to any server-loaded model is unverified."}
    return info, tensors, warnings


def inspect_gguf(path: str | Path | Sequence[str | Path], *, stop_requested: Callable | None = None,
                 on_progress: Callable | None = None) -> GGUFInspection:
    """Index explicitly selected GGUF files/shards, then hash without decoding weights."""
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    if not 1 <= len(paths) <= 64:
        raise ValueError("Select between one and 64 GGUF files.")
    resolved = [Path(value).expanduser().resolve() for value in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Each selected GGUF shard must be distinct.")
    files, tensors, warnings = [], [], []
    for selected in resolved:
        info, entries, notes = _inspect_file(selected, stop_requested, on_progress)
        files.append(info)
        tensors.extend(entries)
        warnings.extend(notes)
        if len(tensors) > MAX_TENSORS or sum(item["header_bytes"] for item in files) > MAX_HEADER:
            raise ValueError("The selected files exceed the combined tensor/header inspection budget.")
    metadata = files[0]["metadata"]
    split_count = metadata.get("split.count", 1)
    if type(split_count) is not int or not 1 <= split_count <= 65536:
        raise ValueError("GGUF split.count is invalid.")
    if split_count > 1:
        indices = [value["metadata"].get("split.no") for value in files]
        compatible = (all(value["metadata"].get("split.count") == split_count for value in files)
                      and all(type(value) is int for value in indices) and len(set(indices)) == len(indices)
                      and set(indices) == set(range(split_count)))
        if not compatible:
            warnings.append(f"Incomplete or incompatible shard selection: declared {split_count}, selected {len(files)}. No other files were opened.")
    elif len(files) > 1:
        warnings.append("Multiple files were selected without a shared split declaration; totals describe the selection, not one verified model.")
    dtype_counts = Counter(tensor["dtype"] for tensor in tensors)
    dtype_bytes = {dtype: (None if any(t["bytes"] is None for t in tensors if t["dtype"] == dtype) else
                          sum(value[2] for value in {(t["source_path"], t["file_offset"], t["bytes"])
                                                    for t in tensors if t["dtype"] == dtype})) for dtype in dtype_counts}
    unique = {(t["source_path"], t["file_offset"], t["bytes"]) for t in tensors if t["bytes"] is not None}
    known_bytes = sum(value[2] for value in unique)
    warnings.append("This is a stored-tensor inventory, not a captured computation graph or a runtime RAM/VRAM measurement.")
    warnings.append("Selected file hashes do not establish which weights an existing server loaded.")
    if any(isinstance(value, dict) and "omitted" in value for value in metadata.values()):
        warnings.append("Large metadata arrays are retained as validated bounded previews with exact counts.")
    summary = {"architecture": metadata.get("general.architecture", "Unknown"), "tensor_count": len(tensors),
               "file_count": len(files), "elements": sum(t["elements"] for t in tensors),
               "storage_bytes": known_bytes, "unknown_storage_tensors": sum(t["bytes"] is None for t in tensors),
               "file_bytes": sum(value["size_bytes"] for value in files), "dtype_counts": dict(dtype_counts),
               "dtype_bytes": dtype_bytes, "layer_count": len({t["group"] for t in tensors if t["group"].startswith("Layer ")}),
               "dtype_bytes_scope": "Unique known ranges per storage type; ranges shared across types appear in each type.",
               "storage_scope": "Unique known tensor byte ranges in selected files; excludes padding and metadata."}
    return GGUFInspection(str(resolved[0]), files, metadata, tensors, summary, list(dict.fromkeys(warnings)))


def compare_gguf(left: dict | GGUFInspection, right: dict | GGUFInspection) -> dict:
    """Compare saved descriptor inventories by unique exact tensor names.

    This never opens source files or compares numerical weights. Unchanged
    descriptors establish shape/storage agreement only, including when a file
    hash changed. Duplicate names across a selected set are left ambiguous.
    """
    from mlcookbook.llm_lab_io import _document, _inspection

    left = left.to_dict() if isinstance(left, GGUFInspection) else left
    right = right.to_dict() if isinstance(right, GGUFInspection) else right
    for value in (left, right):
        _document(value)
        _inspection(value)
        if not all(isinstance(key, str) for key in value["metadata"]):
            raise ValueError("Inspection metadata names must be strings.")
    inventories = []
    for value in (left, right):
        by_name = {}
        for tensor in value["tensors"]:
            by_name.setdefault(tensor["name"], []).append(tensor)
        inventories.append(by_name)
    first, second = inventories
    rows, counts = [], {"added": 0, "removed": 0, "changed": 0, "unchanged": 0, "ambiguous": 0}
    for name in sorted(first.keys() | second.keys()):
        before, after = first.get(name, []), second.get(name, [])
        if len(before) > 1 or len(after) > 1:
            change = "ambiguous"
        elif not before:
            change = "added"
        elif not after:
            change = "removed"
        elif all(before[0][field] == after[0][field] for field in ("shape", "dtype", "bytes")):
            change = "unchanged"
        else:
            change = "changed"
        counts[change] += 1
        if change == "unchanged":
            continue
        row = {"name": name, "change": change}
        for prefix, values in (("left", before), ("right", after)):
            for field in ("shape", "dtype", "bytes"):
                row[prefix + "_" + field] = values[0][field] if len(values) == 1 else None
        rows.append(row)
    metadata_changes = [{"key": key, "left": left["metadata"].get(key), "right": right["metadata"].get(key),
                         "left_present": key in left["metadata"], "right_present": key in right["metadata"]}
                        for key in sorted(left["metadata"].keys() | right["metadata"].keys())
                        if key not in left["metadata"] or key not in right["metadata"]
                        or left["metadata"][key] != right["metadata"][key]]
    warnings = ["Tensor matching uses unique exact names only; it does not establish equivalent architectures or numerical weights.",
                "Comparisons describe explicitly inspected files, not the weights loaded by a connected server."]
    if counts["ambiguous"]:
        warnings.append("Duplicate tensor names in a selected file set are ambiguous and were not paired by list order.")
    if any(isinstance(item, dict) and "omitted" in item for value in (left, right) for item in value["metadata"].values()):
        warnings.append("Large metadata arrays are compared by their saved previews; equality does not cover omitted elements.")
    def files(value):
        return [{key: item[key] for key in ("path", "sha256", "size_bytes")} for item in value["files"]]

    return {"summary": (f"{counts['changed']} changed, {counts['added']} added, {counts['removed']} removed, "
                        f"{counts['unchanged']} unchanged descriptor names; {counts['ambiguous']} ambiguous. "
                        f"{len(metadata_changes)} changed metadata entries."),
            "rows": rows, "metadata_changes": metadata_changes, "counts": counts,
            "left_files": files(left), "right_files": files(right), "warnings": warnings}


def probe_llamacpp(base_url: str, *, stop_requested: Callable | None = None) -> LlamaCppCapabilities:
    """Explicitly probe bounded read-only endpoints; never run inference."""
    from mlcookbook.llamacpp import llamacpp_models, local_llamacpp_url

    base_url = local_llamacpp_url(base_url)
    connection = llamacpp_models(base_url, stop_requested=stop_requested).to_dict()
    capabilities = {"chat_generation": {"status": "not_tested", "reason": "Connection discovery does not run inference."},
                    "compute_graph": {"status": "unavailable", "reason": "Standard server discovery does not expose an internal graph; use an explicit research capture."},
                    "probabilities": {"status": "not_tested", "reason": "Requires an explicit generation request."},
                    "embeddings": {"status": "not_tested", "reason": "Requires an embedding-compatible model and explicit request."}}
    warnings = list(connection["warnings"])
    for name, endpoint, body in (("slots", "/slots", None), ("metrics", "/metrics", None),
                                  ("tokenize", "/tokenize", {"content": "", "with_pieces": True}),
                                  ("apply_template", "/apply-template", {"messages": [{"role": "user", "content": "inspection probe"}]})):
        _check(stop_requested)
        try:
            with _response(base_url, endpoint, body, 5.0, stop_requested) as response:
                payload = response.read(2 * 1024 * 1024 + 1)
                if len(payload) > 2 * 1024 * 1024:
                    raise ValueError("Probe response exceeds the 2 MiB budget.")
            if name == "metrics":
                text = payload.decode("utf-8")
                if not any(line.startswith(("# HELP ", "# TYPE ", "llamacpp:")) for line in text.splitlines()):
                    raise ValueError("The endpoint did not return recognizable server metrics.")
            else:
                value = json.loads(payload)
                valid = (isinstance(value, list) and all(isinstance(row, dict) for row in value) if name == "slots" else
                         isinstance(value, dict) and isinstance(value.get("tokens"), list) if name == "tokenize" else
                         isinstance(value, dict) and isinstance(value.get("prompt"), str))
                if not valid:
                    raise ValueError("The endpoint did not return the expected capability response.")
            capabilities[name] = {"status": "available", "reason": "Endpoint answered a bounded read-only probe.", "endpoint": endpoint}
        except InterruptedError:
            raise
        except (ValueError, UnicodeError, ConnectionError, TimeoutError) as exc:
            capabilities[name] = {"status": "unavailable", "reason": str(exc)[:1000], "endpoint": endpoint}
    return LlamaCppCapabilities(base_url, datetime.now(UTC).isoformat(), connection, capabilities, warnings)


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _ratio(count, seconds):
    return count / seconds if count is not None and seconds is not None and seconds > 0 else None


def _native_trial(base_url, model, prompt, max_new_tokens, temperature, max_seconds, cache_policy,
                  random_state, stop_requested, on_progress):
    from mlcookbook._local_http import _object
    from mlcookbook.llamacpp import _sse, llamacpp_models

    connection = llamacpp_models(base_url, stop_requested=stop_requested)
    if not connection.models or connection.models[0].name != model:
        raise ValueError("The connected llama.cpp model changed or is not ready.")
    messages = [{"role": "user", "content": prompt}]
    rendered = _json(base_url, "/apply-template", {"messages": messages}, seconds=max_seconds, stop_requested=stop_requested).get("prompt")
    if not isinstance(rendered, str) or len(rendered) > 256 * 1024:
        raise ValueError("llama.cpp did not return a bounded rendered prompt.")
    body = {"prompt": rendered, "n_predict": max_new_tokens, "temperature": temperature,
            "seed": random_state, "stream": True, "cache_prompt": cache_policy == "reuse"}
    started, content, reasoning, timings, applied, final = monotonic(), "", "", {}, {}, None
    measured_started = perf_counter()
    with _response(base_url, "/completion", body, max_seconds, stop_requested) as response:
        for payload in _sse(response):
            _check(stop_requested, started + max_seconds)
            if payload == b"[DONE]":
                break
            event = _object(payload)
            if event.get("model") not in (None, model):
                raise ValueError("The native completion returned a different model identity.")
            text = event.get("content", "")
            if not isinstance(text, str):
                raise ValueError("Native completion content must be text.")
            if final is not None:
                raise ValueError("Native completion sent data after its terminal event.")
            content += text
            if len(content) > 256 * 1024:
                raise ValueError("Native benchmark output exceeds the 256 KiB budget.")
            if on_progress and text:
                # Native completion text may contain thinking markup. It is not
                # parsed answer/reasoning data as on the chat-completions path.
                on_progress({"raw_text": content})
            if isinstance(event.get("timings"), dict):
                timings.update(event["timings"])
            if isinstance(event.get("generation_settings"), dict):
                applied.update(event["generation_settings"])
            if event.get("stop") is True:
                final = event
    if final is None:
        raise ValueError("Native benchmark stream ended without a terminal stop event.")
    if "cache_prompt" in applied and type(applied["cache_prompt"]) is not bool:
        raise ValueError("Native benchmark returned an invalid applied cache policy.")
    if "cache_prompt" in applied and applied["cache_prompt"] != body["cache_prompt"]:
        raise ValueError("Native benchmark did not apply the requested cache policy.")
    seconds = perf_counter() - measured_started
    return {"seconds": seconds, "text": content, "thinking": reasoning,
            "input_tokens": _count(final.get("tokens_evaluated")),
            "input_tokens_scope": "Native tokens_evaluated as reported; processed prompt_n is a separate metric.",
            "generated_tokens": _count(timings.get("predicted_n", final.get("tokens_predicted"))),
            "finish_reason": final.get("stop_type", "stop"), "timings": timings, "applied_settings": applied,
            "usage": {key: final[key] for key in ("tokens_cached", "tokens_evaluated", "tokens_predicted", "truncated") if key in final},
            "rendered_prompt": rendered, "messages": messages, "versions": {"llama.cpp": connection.version},
            "model_identity": connection.models[0].to_dict(), "server_properties": connection.properties,
            "content_scope": "Raw native completion; answer and reasoning are not separated.",
            "cache_verified": "cache_prompt" in applied}


def benchmark_llamacpp(*, base_url: str, model: str, prompt: str, repetitions: int = 3, warmups: int = 1,
                       max_new_tokens: int = 64, temperature: float = 0.0, cache_policy: str = "uncontrolled",
                       max_seconds: float = 90.0, random_state: int = 0, stop_requested: Callable | None = None,
                       on_progress: Callable | None = None) -> LlamaCppBenchmark:
    """Serial natural-generation trials with raw outcomes and separate timing scopes.

    ``disabled`` and ``reuse`` explicitly request native cache_prompt values;
    support/application is recorded separately from the request. Warmups remain
    visible but are excluded from measured aggregates. Cancellation returns the
    completed and interrupted trials, rather than discarding their evidence.
    """
    from mlcookbook.llamacpp import generate_llamacpp, local_llamacpp_url

    base_url = local_llamacpp_url(base_url)
    for value, label, low, high in ((repetitions, "repetitions", 1, 100), (warmups, "warmups", 0, 10),
                                  (max_new_tokens, "max_new_tokens", 1, 512), (random_state, "random_state", 0, 2147483647)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{label} must be an integer between {low} and {high}.")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 65536:
        raise ValueError("Enter a nonempty benchmark prompt of at most 65536 characters.")
    if not isinstance(model, str) or not model.strip() or len(model) > 512:
        raise ValueError("Choose a loaded llama.cpp model.")
    if _number(temperature) is None or temperature > 2 or _number(max_seconds) is None or not 1 <= max_seconds <= 120:
        raise ValueError("Temperature must be 0–2 and per-trial time limit must be 1–120 seconds.")
    if cache_policy not in ("uncontrolled", "disabled", "reuse"):
        raise ValueError("cache_policy must be uncontrolled, disabled or reuse.")
    if any(callback is not None and not callable(callback) for callback in (stop_requested, on_progress)):
        raise ValueError("Benchmark callbacks must be callable or None.")
    settings = {"prompt": prompt, "repetitions": repetitions, "warmups": warmups, "max_new_tokens": max_new_tokens,
                "temperature": temperature, "random_state": random_state, "max_seconds": max_seconds,
                "cache_policy": cache_policy, "concurrency": 1, "trial_order": "warmups then measured; serial",
                "workload": "natural generation; actual output counts and early stops may differ",
                "first_output_scope": "whole trial, including connection/template setup; client-observed chunk arrival",
                "first_answer_scope": "Parsed chat answer only; unavailable for raw native completion text",
                "seconds_scope": "generation request through terminal stream", "action_seconds_scope": "whole trial including setup",
                "throughput_scope": "reported combined generation tokens; reasoning may be included",
                "client_clock": "time.perf_counter; monotonic deadlines use time.monotonic separately",
                "cache_verification_scope": "Only explicitly reported application of cache_prompt; not proof of prefix reuse or a cold cache."}
    trials, warnings, stopped = [], [], False
    if cache_policy == "uncontrolled":
        warnings.append("Prompt-cache behavior is uncontrolled on the chat API; warmups do not establish an uncached or cold-storage baseline.")
    else:
        warnings.append("Cache policy is a native request. Applied settings and tokens_cached establish only their reported scope, not filesystem-cache state.")
    action_started = perf_counter()
    for index in range(warmups + repetitions):
        trial_start = perf_counter()
        trial = {"index": index + 1, "warmup": index < warmups, "status": "pending",
                 "first_output_seconds": None, "first_answer_seconds": None, "first_reasoning_seconds": None,
                 "chunk_arrivals": []}
        deadline = monotonic() + max_seconds

        def trial_stop(deadline=deadline):
            _check(stop_requested, deadline)
            return False

        def progress(event, trial=trial, trial_start=trial_start, index=index):
            arrived = perf_counter() - trial_start
            if any(event.get(key) for key in ("text", "thinking", "raw_text")) and trial["first_output_seconds"] is None:
                trial["first_output_seconds"] = arrived
            if event.get("text") and trial["first_answer_seconds"] is None:
                trial["first_answer_seconds"] = arrived
            if event.get("thinking") and trial["first_reasoning_seconds"] is None:
                trial["first_reasoning_seconds"] = arrived
            if len(trial["chunk_arrivals"]) < 4096:
                trial["chunk_arrivals"].append({"seconds": arrived, "answer_characters": len(event.get("text", "")),
                                                 "reasoning_characters": len(event.get("thinking", "")),
                                                 "raw_characters": len(event.get("raw_text", ""))})
            if on_progress:
                on_progress({"trial": index + 1, "total": warmups + repetitions, "event": event})
        try:
            _check(trial_stop)
            if cache_policy == "uncontrolled":
                result = generate_llamacpp(base_url=base_url, model=model, prompt=prompt, max_new_tokens=max_new_tokens,
                                          temperature=temperature, random_state=random_state, max_seconds=max_seconds,
                                          stop_requested=trial_stop, on_progress=progress)
                value = result.outputs[0].to_dict() if hasattr(result.outputs[0], "to_dict") else result.to_dict()["outputs"][0]
                value["versions"] = result.versions
                value["cache_verified"] = False
                value["model_identity"] = result.model.to_dict()
                value["server_properties"] = result.server_properties
                value["content_scope"] = "Chat answer and separately reported reasoning content."
            else:
                value = _native_trial(base_url, model, prompt, max_new_tokens, temperature, max_seconds, cache_policy,
                                      random_state, trial_stop, progress)
            trial.update(value)
            timings = value["timings"]
            prompt_seconds = _number(timings.get("prompt_ms"))
            decode_seconds = _number(timings.get("predicted_ms"))
            prompt_seconds = None if prompt_seconds is None else prompt_seconds / 1000
            decode_seconds = None if decode_seconds is None else decode_seconds / 1000
            prompt_count = _count(timings.get("prompt_n"))
            decode_count = _count(timings.get("predicted_n"))
            trial.update({"status": "completed", "prompt_seconds": prompt_seconds, "decode_seconds": decode_seconds,
                          "prompt_tokens_per_second": _ratio(prompt_count, prompt_seconds),
                          "decode_tokens_per_second": _ratio(decode_count, decode_seconds),
                          "end_to_end_tokens_per_second": _ratio(_count(value["generated_tokens"]), _number(value["seconds"])),
                          "prompt_metric_tokens": prompt_count, "decode_metric_tokens": decode_count})
            if decode_seconds is not None and decode_count is None:
                warnings.append("A decode duration lacked a matching backend decoded-token count; decode throughput is unavailable.")
            if cache_policy != "uncontrolled" and not value["cache_verified"]:
                warnings.append("At least one native trial did not report whether the requested cache policy was applied.")
        except InterruptedError as exc:
            trial.update({"status": "cancelled", "error": str(exc)[:1000]})
            stopped = True
        except (ValueError, ConnectionError, TimeoutError, OSError) as exc:
            trial.update({"status": "failed", "error": str(exc)[:1000]})
        trial["action_seconds"] = perf_counter() - trial_start
        trials.append(trial)
        if on_progress:
            on_progress({"trial": index + 1, "total": warmups + repetitions, "outcome": trial})
        if stopped:
            break
    measured = [trial for trial in trials if not trial["warmup"]]
    completed = [trial for trial in measured if trial["status"] == "completed"]
    aggregates = {}
    for key in ("seconds", "action_seconds", "first_output_seconds", "first_answer_seconds", "first_reasoning_seconds", "prompt_tokens_per_second",
                "decode_tokens_per_second", "end_to_end_tokens_per_second"):
        values = [trial[key] for trial in completed if _number(trial.get(key)) is not None]
        aggregates[key] = {"count": len(values), "mean": statistics.fmean(values) if values else None,
                           "median": statistics.median(values) if values else None,
                           "sd": statistics.stdev(values) if len(values) > 1 else None,
                           "min": min(values) if values else None, "max": max(values) if values else None}
    summary = {"planned_trials": repetitions + warmups, "attempted_trials": len(trials),
               "planned_measured_trials": repetitions, "completed_measured_trials": len(completed),
               "failed_measured_trials": sum(t["status"] == "failed" for t in measured),
               "cancelled": stopped, "unattempted_trials": repetitions + warmups - len(trials),
               "action_seconds": perf_counter() - action_started, "aggregates": aggregates}
    warnings += ["First output times are client-observed chunk arrivals, not device or individual-token timings.",
                 "Native/server timings and client wall times have different scopes. Missing measurements remain unavailable.",
                 "These natural-generation trials do not establish equal computational work or labeled output quality."]
    if stopped:
        warnings.append("Benchmark cancelled; completed trials are retained and unfinished work is counted explicitly.")
    return LlamaCppBenchmark("benchmark_llamacpp", base_url, model, datetime.now(UTC).isoformat(), settings,
                             trials, summary, list(dict.fromkeys(warnings)))
