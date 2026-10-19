"""Text workflow contracts: source identity, strict validation, retries and derivation."""

import json
from copy import deepcopy

import pandas as pd
import pytest

from mlcookbook import Dataset
from mlcookbook.llm_workflow import (
    check_source,
    compare_runs,
    dataset_fingerprint,
    derive_dataset,
    export_workflow,
    output_schema,
    parse_fields,
    prepare_workflow,
    run_metrics,
    run_workflow,
    validate_output,
)

FIELDS = "sentiment: categorical: positive | negative\nscore: numeric\nurgent: boolean"
ANSWER = '{"sentiment":"positive","score":2.5,"urgent":false}'


def ndjson(text=ANSWER):
    return (json.dumps({"message": {"content": text}, "done": False}) + "\n" + json.dumps({
        "done": True, "model": "test:tiny", "done_reason": "stop", "prompt_eval_count": 10,
        "eval_count": 12, "total_duration": 1000000000, "eval_duration": 500000000, "load_duration": 100000000}) + "\n").encode()


def sse(text=ANSWER):
    return ("data: " + json.dumps({"model": "test-gguf", "choices": [{"index": 0,
        "delta": {"content": text}, "finish_reason": "stop"}], "usage": {"completion_tokens": 12, "prompt_tokens": 10}})
            + "\n\ndata: [DONE]\n\n").encode()


@pytest.fixture
def source():
    return Dataset(pd.DataFrame({"text": ["great", None, "terrible", "", "last"],
                                 "reference": ["positive", "negative", "negative", "negative", "positive"],
                                 "trusted_target": pd.array([1, 2, 3, 4, 5], dtype="Int64")}, index=[7, 7, 2, 2, 7]),
                   overrides={"text": "text"})


def execute(server, source, **kwargs):
    return run_workflow(source, backend="ollama", model="test:tiny", base_url=server.base_url,
                         template="Read {{text}}", bindings=["text"], field_spec=FIELDS,
                         references="sentiment=reference", max_rows=4, **kwargs)


def test_structured_ollama_rows_roles_lineage_and_serialization(ollama_server, source, tmp_path):
    ollama_server.state["raw_stream"] = ndjson()
    events = []
    result = execute(ollama_server, source, on_progress=events.append)
    run = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert [row["status"] for row in run["rows"]] == ["valid", "skipped", "valid", "skipped"]
    sent = [body for path, body in ollama_server.requests if path == "/api/chat"]
    assert len(sent) == 2 and sent[0]["format"] == output_schema(parse_fields(FIELDS))
    assert "great" in sent[0]["messages"][0]["content"]
    assert "trusted_target" not in sent[0]["messages"][0]["content"]
    assert "reference" not in sent[0]["messages"][0]["content"]
    assert events[-1]["row_position"] == 3 and events[-1]["total_rows"] == 4 and events[-1]["status"] == "skipped"
    derived, lineage = derive_dataset(source, run)
    pd.testing.assert_frame_equal(derived.df[source.df.columns], source.df)
    assert derived.df.index.tolist() == [7, 7, 2, 2, 7]
    assert derived.df.score.iloc[[0, 2]].tolist() == [2.5, 2.5]
    assert derived.df.score.iloc[[1, 3, 4]].isna().all()
    assert derived.schema().var_type("sentiment").value == "categorical"
    assert str(derived.df.score.dtype) == "Float64" and str(derived.df.urgent.dtype) == "boolean"
    assert lineage["source_fingerprint"] == dataset_fingerprint(source)
    assert "score" not in source.df
    metrics = run_metrics(source, run)
    assert metrics["validation_rate"] == .5
    assert metrics["reference_agreement"]["sentiment"] == {"reference_column": "reference", "evaluated_rows": 2, "agreement": .5}
    assert metrics["generation_seconds"] == 1 and metrics["load_seconds"] == .2
    assert not any("No labeled evaluation was performed" in warning for warning in run["warnings"])
    path = tmp_path / "run.json"
    export_workflow(path, source, run)
    exported = json.loads(path.read_text(encoding="utf-8"))
    assert exported["source_rows"][1] == [None, "negative", 2]
    assert exported["run"] == run


def test_structured_llamacpp_request_contract(llamacpp_server, source):
    llamacpp_server.state["raw_stream"] = sse()
    run = run_workflow(source, backend="llamacpp", model="test-gguf", base_url=llamacpp_server.base_url,
                       template="Read {{text}}", bindings=["text"], field_spec=FIELDS).to_dict()
    assert run["rows"][0]["status"] == "valid"
    body = next(body for path, body in llamacpp_server.requests if path == "/v1/chat/completions")
    assert body["response_format"] == {"type": "json_object", "schema": output_schema(parse_fields(FIELDS))}
    assert run_metrics(source, run)["generation_seconds"] is None


def test_llamacpp_response_timestamp_is_not_model_identity(llamacpp_server, source):
    llamacpp_server.state["raw_stream"] = sse()
    llamacpp_server.state["models"][0]["created"] = 1
    kwargs = dict(backend="llamacpp", model="test-gguf", base_url=llamacpp_server.base_url,
                  template="{{text}}", bindings=["text"], field_spec=FIELDS)
    first = run_workflow(source, **kwargs).to_dict()
    llamacpp_server.state["models"][0]["created"] = 2
    retry = run_workflow(source, previous=first, positions=[0], **kwargs)
    assert retry.model_identity == first["model_identity"]
    llamacpp_server.state["models"][0]["meta"]["n_params"] += 1
    with pytest.raises(RuntimeError, match="identity changed"):
        run_workflow(source, previous=first, positions=[0], **kwargs)


def test_llamacpp_changed_server_template_prevents_retry_merge(llamacpp_server, source):
    llamacpp_server.state["raw_stream"] = sse()
    kwargs = dict(backend="llamacpp", model="test-gguf", base_url=llamacpp_server.base_url,
                  template="{{text}}", bindings=["text"], field_spec=FIELDS)
    first = run_workflow(source, **kwargs).to_dict()
    llamacpp_server.state["props"]["chat_template"] = "new server template"
    with pytest.raises(RuntimeError, match="identity changed"):
        run_workflow(source, previous=first, positions=[0], **kwargs)


@pytest.mark.parametrize("text", ["```json\n" + ANSWER + "\n```", ANSWER + " trailing", "[]",
    '{"sentiment":"positive","score":"2.5","urgent":false}',
    '{"sentiment":"other","score":2,"urgent":false}',
    '{"sentiment":"positive","score":true,"urgent":false}',
    '{"sentiment":"positive","score":NaN,"urgent":false}',
    '{"sentiment":"positive","score":1e999,"urgent":false}',
    '{"sentiment":"positive","score":2,"score":3,"urgent":false}',
    '{"sentiment":"positive","score":2}',
    '{"sentiment":"positive","score":2,"urgent":false,"extra":1}'])
def test_strict_json_without_repair_or_coercion(text):
    values, errors = validate_output(text, parse_fields(FIELDS))
    assert values is None and errors


def test_null_is_known_missing_not_numeric_coercion():
    values, errors = validate_output('{"sentiment":null,"score":null,"urgent":null}', parse_fields(FIELDS))
    assert not errors and values == {"sentiment": None, "score": None, "urgent": None}


def test_limit_stopped_output_is_not_accepted_even_if_json_parses(ollama_server, source):
    ollama_server.state["raw_stream"] = ndjson().replace(b'"done_reason": "stop"', b'"done_reason": "length"')
    run = execute(ollama_server, source).to_dict()
    assert run["rows"][0]["status"] == "invalid" and run["rows"][0]["raw_text"] == ANSWER
    assert run["rows"][0]["values"] is None
    derived, _ = derive_dataset(source, run)
    assert derived.df.score.isna().all()


@pytest.mark.parametrize("spec", ["", "a: integer", "a: text\na: numeric", "bad name: text", "a: numeric: yes | no",
                                       "a: categorical: yes | yes", "a: categorical: yes | "])
def test_bad_field_definitions(spec):
    with pytest.raises(ValueError):
        parse_fields(spec)


@pytest.mark.parametrize("template,bindings,references", [
    ("{{missing}}", ["missing"], ""), ("{{trusted_target}}", ["trusted_target"], ""),
    ("{{text}}", [], ""), ("{{text}} {{reference}}", ["text", "reference"], "sentiment=reference"),
    ("{{text", ["text"], ""), ("{{text}}", ["text"], "absent=reference")])
def test_invalid_or_leaking_bindings(source, template, bindings, references):
    with pytest.raises(ValueError):
        prepare_workflow(source, template=template, bindings=bindings, field_spec=FIELDS, references=references)


def test_multiple_explicit_bindings_do_not_recursively_expand():
    data = Dataset(pd.DataFrame({"a": ["{{b}}"], "b": ["second"]}))
    prepared = prepare_workflow(data, template="{{a}} / {{b}}", bindings=["a", "b"], field_spec="summary: text")
    assert prepared["bindings"] == ["a", "b"]
    coded = Dataset(pd.DataFrame({"a": [1, 2]}), overrides={"a": "categorical"})
    with pytest.raises(ValueError, match="non-string"):
        prepare_workflow(coded, template="{{a}}", bindings=["a"], field_spec="summary: text")


def test_failed_rows_retry_preserves_other_rows_and_old_run(ollama_server, source):
    ollama_server.state["raw_stream"] = ndjson("bad json")
    old = execute(ollama_server, source).to_dict()
    assert old["rows"][0]["raw_text"] == "bad json" and old["rows"][0]["status"] == "invalid"
    ollama_server.state["raw_stream"] = ndjson()
    retried = execute(ollama_server, source, previous=old, positions=[2]).to_dict()
    assert retried["parent_run_id"] == old["run_id"]
    assert retried["rows"][0] == old["rows"][0] and retried["rows"][2]["status"] == "valid"
    assert old["rows"][2]["status"] == "invalid"
    assert len(compare_runs(source, [old, retried])) == 2
    with pytest.raises(ValueError, match="Settings changed"):
        execute(ollama_server, source, previous=old, positions=[2], settings={"temperature": .5})
    with pytest.raises(ValueError, match="Select existing"):
        execute(ollama_server, source, previous=old, positions=[100])


def test_cancel_retains_completed_rows(ollama_server, source):
    ollama_server.state["raw_stream"] = ndjson()
    events = []
    result = execute(ollama_server, source, stop_requested=lambda: any(event.get("status") == "valid" for event in events),
                     on_progress=events.append)
    assert [row["status"] for row in result.rows] == ["valid", "skipped", "cancelled", "skipped"]
    assert result.rows[0]["values"]["score"] == 2.5


def test_backend_errors_recorded_per_row(ollama_server, source):
    ollama_server.state["status"] = 500
    result = execute(ollama_server, source)
    assert [row["status"] for row in result.rows] == ["error", "skipped", "error", "skipped"]
    assert result.model_identity is None


def test_source_and_saved_value_tampering_rejected(ollama_server, source):
    ollama_server.state["raw_stream"] = ndjson()
    run = execute(ollama_server, source).to_dict()
    changed = Dataset(source.df.iloc[::-1], overrides={"text": "text"})
    with pytest.raises(ValueError, match="fingerprint"):
        derive_dataset(changed, run)
    for change in (lambda value: value["rows"].reverse(),
                   lambda value: value["rows"][0]["values"].update(score=100),
                   lambda value: value["rows"][0]["inputs"].update(text="different")):
        bad = deepcopy(run)
        change(bad)
        with pytest.raises(ValueError):
            check_source(source, bad)


def test_comparison_requires_same_rows_and_schema(ollama_server, source):
    ollama_server.state["raw_stream"] = ndjson()
    run = execute(ollama_server, source).to_dict()
    shorter = run_workflow(source, backend="ollama", model="test:tiny", base_url=ollama_server.base_url,
                            template="{{text}}", bindings=["text"], field_spec=FIELDS).to_dict()
    with pytest.raises(ValueError, match="same source row"):
        compare_runs(source, [run, shorter])


def test_missing_usage_is_unavailable(ollama_server, source):
    ollama_server.state["raw_stream"] = ndjson().replace(b'"eval_count": 12, ', b'')
    run = execute(ollama_server, source).to_dict()
    assert run_metrics(source, run)["tokens_per_second"] is None
