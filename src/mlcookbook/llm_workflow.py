"""Repeatable text-to-columns runs, independent of Qt and local runtime ownership.

Templates use literal {{column name}} bindings. Output fields have explicit
statistical roles; parsing never repairs JSON or converts strings into numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import uuid4

import pandas as pd

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.fingerprint import dataset_fingerprint
from mlcookbook.core.results import Result

_BINDING = re.compile(r"\{\{([^{}]+)\}\}")
_TYPES = {"text": "string", "categorical": "string", "numeric": "number", "boolean": "boolean"}


@dataclass(frozen=True)
class LLMWorkflowResult(Result):
    method: str
    version: int
    run_id: str
    created_utc: str
    dataset_fingerprint: str
    config_fingerprint: str
    config: dict
    model_identity: dict | None
    rows: list[dict]
    parent_run_id: str | None
    warnings: list[str]


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def parse_fields(spec: str) -> list[dict]:
    """Parse ``name: role`` or ``name: categorical: label | label`` lines."""
    if not isinstance(spec, str) or len(spec) > 8000:
        raise ValueError("Field definitions must be text of at most 8000 characters.")
    fields = []
    for line in spec.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(":", 2)]
        if len(parts) < 2 or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", parts[0]):
            raise ValueError("Define each field as name: role, using a short Python-style field name.")
        name, role = parts[:2]
        if role not in _TYPES:
            raise ValueError(f"Unknown field role {role!r}. Choose text, categorical, numeric or boolean.")
        labels = [label.strip() for label in parts[2].split("|")] if len(parts) == 3 else []
        if labels and (role != "categorical" or any(not label or len(label) > 128 for label in labels)
                       or len(labels) != len(set(labels)) or len(labels) > 50):
            raise ValueError("Labels must be unique, nonblank categorical values (at most 50).")
        fields.append({"name": name, "role": role, "labels": labels})
    if not 1 <= len(fields) <= 8 or len({field['name'] for field in fields}) != len(fields):
        raise ValueError("Define between one and eight fields with unique names.")
    return fields


def output_schema(fields: list[dict]) -> dict:
    properties = {}
    for field in fields:
        entry = {"type": [_TYPES[field["role"]], "null"]}
        if field["labels"]:
            entry["enum"] = [*field["labels"], None]
        properties[field["name"]] = entry
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}.")
        result[key] = value
    return result


def validate_output(text: str, fields: list[dict]) -> tuple[dict | None, list[str]]:
    """Strict flat JSON only. Null represents an unknown value, never an imputation."""
    try:
        values = json.loads(text, object_pairs_hook=_strict_object,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {value}.")))
        if not isinstance(values, dict) or set(values) != {field["name"] for field in fields}:
            raise ValueError("JSON must be an object containing exactly the declared fields.")
        errors = []
        for field in fields:
            value, role = values[field["name"]], field["role"]
            valid = value is None or (
                isinstance(value, str) if role in ("text", "categorical") else
                type(value) is bool if role == "boolean" else
                type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e308)
            if value is not None and field["labels"] and value not in field["labels"]:
                valid = False
            if not valid:
                errors.append(f"{field['name']}: expected {role}" + (f" in {field['labels']}" if field["labels"] else "") + ".")
        return (None, errors) if errors else (values, [])
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        return None, [str(exc)]


def prepare_workflow(dataset: Dataset, *, template: str, bindings: list[str], field_spec: str,
                     references: str = "") -> dict:
    """Validate bindings before inference. Reference columns never enter prompts."""
    fields = parse_fields(field_spec)
    if not isinstance(template, str) or not template.strip() or len(template) > 16000:
        raise ValueError("Write a prompt template of at most 16000 characters.")
    columns = list(dict.fromkeys(_BINDING.findall(template)))
    if "{{" in _BINDING.sub("", template) or "}}" in _BINDING.sub("", template):
        raise ValueError("Use {{column name}} for each template binding.")
    if not columns or set(columns) != set(bindings) or len(bindings) != len(set(bindings)):
        raise ValueError("Select exactly the input columns named in the template's {{bindings}}.")
    for column in columns:
        if column not in dataset.df or dataset.schema().var_type(column).value not in ("text", "categorical"):
            raise ValueError(f"Binding {column!r} needs an existing text or categorical column.")
        if any(not isinstance(value, str) for value in dataset.df[column].dropna()):
            raise ValueError(f"Binding {column!r} contains non-string values. No silent conversion is performed.")
    if not isinstance(references, str) or len(references) > 8000:
        raise ValueError("Reference bindings must be text of at most 8000 characters.")
    reference_map = {}
    for line in references.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split("=", 1)]
        if len(parts) != 2 or parts[0] not in {field["name"] for field in fields} or parts[1] not in dataset.df:
            raise ValueError("Reference bindings use output_field=source_column, one per line.")
        if parts[0] in reference_map or parts[1] in columns:
            raise ValueError("Reference columns must be unique per field and excluded from prompt inputs.")
        reference_map[parts[0]] = parts[1]
    return {"template": template, "bindings": columns, "fields": fields, "references": reference_map}


def _identity(result):
    model = result["model"]
    if result["backend"] == "llamacpp":
        # /v1/models "created" can be the response time; status is transient.
        # Preserve the complete response per row, but compare stable attributes.
        model = {"name": model["name"], "metadata": {key: value for key, value in model["metadata"].items()
                                                   if key not in ("created", "status")}}
    return {"backend": result["backend"], "base_url": result["base_url"], "model": model,
            "local_file": result.get("local_file"), "versions": result["versions"],
            "server_configuration": {key: value for key, value in result.get("server_properties", {}).items()
                                      if key not in ("is_sleeping", "total_slots")}}


def run_workflow(dataset: Dataset, *, backend: str, model: str, base_url: str, template: str,
                 bindings: list[str], field_spec: str, references: str = "", max_rows: int = 1,
                 settings: dict | None = None, previous: dict | None = None,
                 positions: list[int] | None = None, stop_requested: Callable[[], bool] | None = None,
                 on_progress: Callable[[dict], None] | None = None) -> LLMWorkflowResult:
    """Run at most ten source positions; preserve completed rows on cancellation.

    An explicit retry copies previous outcomes only when source/config match and
    the newly observed model identity matches. Each new run remains a separate
    artifact. A complete backend response is retained even if JSON validation fails.
    """
    if backend not in ("ollama", "llamacpp"):
        raise ValueError("Structured data workflows require Ollama or llama.cpp.")
    if type(max_rows) is not int or not 1 <= max_rows <= 10:
        raise ValueError("Choose a batch size from 1 to 10 source rows.")
    allowed = {"max_new_tokens", "temperature", "top_p", "random_state", "max_seconds"}
    allowed |= {"context_tokens", "thinking", "prompt_format"} if backend == "ollama" else {"gguf_path"}
    if settings is not None and (not isinstance(settings, dict) or not set(settings) <= allowed):
        raise ValueError("Workflow settings contain unsupported or reserved generation controls.")
    if any(callback is not None and not callable(callback) for callback in (stop_requested, on_progress)):
        raise ValueError("Workflow callbacks must be callable or None.")
    prepared = prepare_workflow(dataset, template=template, bindings=bindings, field_spec=field_spec, references=references)
    if any(field["name"] in dataset.df for field in prepared["fields"]):
        raise ValueError("Generated field names must not overwrite source columns. Choose new names.")
    config = {**prepared, "backend": backend, "model": model, "base_url": base_url,
              "settings": dict(settings or {}), "max_rows": max_rows}
    fingerprint, config_hash = dataset_fingerprint(dataset), _hash(config)
    selected = list(range(min(max_rows, len(dataset.df))))
    if not selected:
        raise ValueError("The source dataset has no rows.")
    identity, retained = None, {}
    if previous is not None:
        check_source(dataset, previous)
        if previous["config_fingerprint"] != config_hash:
            raise ValueError("Settings changed. Start a new full run before retrying rows.")
        identity = previous["model_identity"]
        if identity is None or (backend == "ollama" and not identity["model"].get("digest")):
            raise ValueError("A retry requires a recorded model identity; start a new full run.")
        retained = {row["position"]: row for row in previous["rows"]}
        if positions is None or not positions or len(set(positions)) != len(positions) or any(
                type(position) is not int or position not in selected for position in positions):
            raise ValueError("Select existing source row positions to retry.")
    elif positions is not None:
        raise ValueError("Selected-row retries require a previous run.")
    schema, rows, warnings = output_schema(prepared["fields"]), [], []
    stopped = False
    for number, position in enumerate(selected):
        def progress(event, *, position=position, number=number):
            if on_progress:
                on_progress({**event, "row_position": position, "output_index": number + 1, "total_rows": len(selected)})
        if previous is not None and position not in positions:
            rows.append(retained[position])
            progress({"text": retained[position]["raw_text"], "thinking": "", "status": "retained"})
            continue
        source = {column: dataset.df[column].iloc[position] for column in prepared["bindings"]}
        row = {"position": position, "inputs": source, "prompt": None, "status": "skipped",
               "raw_text": "", "values": None, "errors": [], "generation": None, "seconds": 0.0}
        if any(pd.isna(value) or not value.strip() for value in source.values()):
            row["inputs"] = {key: None if pd.isna(value) else value for key, value in source.items()}
            row["errors"] = ["Missing or blank input binding."]
            rows.append(row)
            progress({"text": "", "thinking": "", "status": "skipped"})
            continue
        prompt = _BINDING.sub(lambda match, source=source: source[match[1]], template)
        prompt += "\n\nReturn only a JSON object matching this schema. Use null for unknown facts.\n" + json.dumps(schema)
        if len(prompt) > 16000:
            row.update(status="error", errors=["Rendered prompt exceeds 16000 characters; no request sent."])
            rows.append(row)
            progress({"text": "", "thinking": "", "status": "error"})
            continue
        row["prompt"] = prompt
        stopped = stopped or bool(stop_requested and stop_requested())
        if stopped:
            row.update(status="cancelled", errors=["Cancelled before a complete response."])
            rows.append(row)
            progress({"text": "", "thinking": "", "status": "cancelled"})
            continue
        started = monotonic()
        try:
            if backend == "ollama":
                from mlcookbook.ollama import generate_ollama

                result = generate_ollama(model=model, base_url=base_url, prompt=prompt, response_schema=schema,
                                         stop_requested=stop_requested, on_progress=progress, **config["settings"]).to_dict()
            else:
                from mlcookbook.llamacpp import generate_llamacpp

                result = generate_llamacpp(model=model, base_url=base_url, prompt=prompt, response_schema=schema,
                                           stop_requested=stop_requested, on_progress=progress, **config["settings"]).to_dict()
            observed = _identity(result)
            if identity is not None and observed != identity:
                raise RuntimeError("Model/runtime identity changed. Start a new full run; previous results were preserved.")
            identity = observed
            output = result["outputs"][0]
            values, errors = validate_output(output["text"], prepared["fields"])
            if output["finish_reason"] in ("length", "max_new_tokens", "time_limit"):
                values, errors = None, [*errors, "Generation stopped at a limit; review a complete response before deriving columns."]
            row.update(status="invalid" if errors else "valid", raw_text=output["text"], values=values,
                       errors=errors, generation=result)
            warnings.extend(warning.replace("No labeled evaluation was performed.",
                                             "Reference agreement is calculated separately when reference columns are configured.")
                            for warning in result["warnings"])
        except InterruptedError:
            stopped = True
            row.update(status="cancelled", errors=["Cancelled before a complete response. Partial stream was not accepted."])
        except (ValueError, OSError, TimeoutError) as exc:
            row.update(status="error", errors=[str(exc)])
        row["seconds"] = monotonic() - started
        rows.append(row)
        progress({"text": row["raw_text"], "thinking": "", "status": row["status"]})
    warnings += ["Schema validation checks shape/types/labels, not factual correctness. Null means unknown.",
                 "Rows are identified by zero-based source position plus a full typed dataset fingerprint.",
                 "Generated predictors can leak labels. Reference columns are excluded from prompt bindings; review the text itself before evaluation."]
    if stopped:
        warnings.append("This batch was cancelled. Completed rows were retained; unfinished rows are explicitly marked cancelled.")
    return LLMWorkflowResult("llm_data_workflow", 1, uuid4().hex, datetime.now(UTC).isoformat(), fingerprint,
                             config_hash, config, identity, rows, previous["run_id"] if previous else None,
                             list(dict.fromkeys(warnings)))


def check_source(dataset: Dataset, run: dict):
    """Validate persisted data before positional joins or retries."""
    if run.get("version") != 1 or run.get("method") != "llm_data_workflow":
        raise ValueError("Unsupported LLM workflow result.")
    if dataset_fingerprint(dataset) != run.get("dataset_fingerprint"):
        raise ValueError("Source dataset fingerprint changed; generated rows cannot be joined safely.")
    config = run["config"]
    if _hash(config) != run.get("config_fingerprint"):
        raise ValueError("Workflow configuration fingerprint does not match.")
    expected = list(range(min(config["max_rows"], len(dataset.df))))
    if [row["position"] for row in run["rows"]] != expected:
        raise ValueError("Workflow row identities are missing, duplicated or out of order.")
    for row in run["rows"]:
        if row["status"] not in ("valid", "invalid", "skipped", "error", "cancelled"):
            raise ValueError("Unknown workflow row status.")
        expected_inputs = {column: dataset.df[column].iloc[row["position"]] for column in config["bindings"]}
        expected_inputs = {key: None if pd.isna(value) else value for key, value in expected_inputs.items()}
        if expected_inputs != row["inputs"]:
            raise ValueError("Saved input bindings do not match their source row identity.")
        if row["status"] == "valid":
            parsed, errors = validate_output(row["raw_text"], config["fields"])
            if errors or parsed != row["values"]:
                raise ValueError("Saved validated values do not match their raw output.")


def derive_dataset(dataset: Dataset, run: dict) -> tuple[Dataset, dict]:
    """Explicit acceptance; copy source and append nullable columns by position."""
    check_source(dataset, run)
    frame = dataset.df.copy()
    roles = {column.name: column.var_type.value for column in dataset.schema().columns}
    for field in run["config"]["fields"]:
        name, role = field["name"], field["role"]
        if name in frame:
            raise ValueError(f"Generated field {name!r} would overwrite a source column.")
        values = [None] * len(frame)
        for row in run["rows"]:
            if row["status"] == "valid":
                values[row["position"]] = row["values"][name]
        dtype = {"numeric": "Float64", "boolean": "boolean", "text": "string", "categorical": "string"}[role]
        frame[name] = pd.array(values, dtype=dtype)  # no index-label alignment
        roles[name] = role
    lineage = {"kind": "llm_derived", "source_fingerprint": run["dataset_fingerprint"],
               "run_id": run["run_id"], "config_fingerprint": run["config_fingerprint"],
               "fields": run["config"]["fields"], "row_positions": [row["position"] for row in run["rows"] if row["status"] == "valid"],
               "workflow": run}
    return Dataset(frame, overrides=roles), lineage


def run_metrics(dataset: Dataset, run: dict) -> dict:
    check_source(dataset, run)
    rows = run["rows"]
    outputs = [row["generation"]["outputs"][0] for row in rows if row["generation"]]
    tokens = [output["generated_tokens"] for output in outputs]
    wall = sum(row["seconds"] for row in rows)
    metrics = {"rows": len(rows), "valid": sum(row["status"] == "valid" for row in rows),
               "validation_rate": sum(row["status"] == "valid" for row in rows) / len(rows),
               "wall_seconds": wall, "output_characters": sum(len(row["raw_text"]) for row in rows),
               "tokens_per_second": sum(tokens) / wall if wall and tokens and all(token is not None for token in tokens) else None}
    # Backend decode/load durations retain their own semantics, never replace wall time.
    for key in ("load_seconds", "generation_seconds"):
        values = [output.get(key) for output in outputs]
        if run["config"]["backend"] == "llamacpp" and key == "generation_seconds":
            values = [output.get("timings", {}).get("predicted_ms") for output in outputs]
            values = [value / 1000 if isinstance(value, (int, float)) and value >= 0 else None for value in values]
        metrics[key] = sum(values) if values and all(value is not None for value in values) else None
    evaluation = {}
    for field, column in run["config"]["references"].items():
        pairs = [(row["values"][field], dataset.df[column].iloc[row["position"]]) for row in rows
                 if row["status"] == "valid" and row["values"][field] is not None
                 and not pd.isna(dataset.df[column].iloc[row["position"]])]
        # Compare like scalar types; string '1' is not numeric 1.
        from mlcookbook.core.results import to_jsonable

        role = next(value["role"] for value in run["config"]["fields"] if value["name"] == field)
        def agrees(pred, reference, role=role):
            reference = to_jsonable(reference)
            compatible = (type(reference) in (int, float) if role == "numeric" else
                          type(reference) is bool if role == "boolean" else isinstance(reference, str))
            return compatible and pred == reference
        matches = [agrees(pred, ref) for pred, ref in pairs]
        evaluation[field] = {"reference_column": column, "evaluated_rows": len(pairs),
                             "agreement": sum(bool(match) for match in matches) / len(pairs) if pairs else None}
    metrics["reference_agreement"] = evaluation
    return metrics


def compare_runs(dataset: Dataset, runs: list[dict]) -> list[dict]:
    if len(runs) < 2:
        raise ValueError("Keep at least two runs to compare.")
    first = runs[0]
    for run in runs:
        check_source(dataset, run)
        if [row["position"] for row in run["rows"]] != [row["position"] for row in first["rows"]]:
            raise ValueError("Compare runs on the same source row positions and dataset snapshot.")
        if run["config"]["fields"] != first["config"]["fields"] or run["config"]["references"] != first["config"]["references"]:
            raise ValueError("Compare runs with the same output fields and reference bindings.")
    return [{"run_id": run["run_id"], **run_metrics(dataset, run)} for run in runs]


def export_workflow(path: str | Path, dataset: Dataset, run: dict):
    """One explicit JSON export: full source rows, outcomes, provenance and metrics."""
    check_source(dataset, run)
    from mlcookbook.core.results import to_jsonable

    payload = {"format": "mlcookbook-llm-workflow-export", "version": 1, "run": run,
               "metrics": run_metrics(dataset, run), "source_columns": list(dataset.df.columns),
               "source_rows": to_jsonable(dataset.df.astype(object).where(dataset.df.notna(), None).values.tolist())}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
