"""Research files are checked data, independent of the existing experiment format."""

import hashlib
import json
import zipfile

import pytest

from mlcookbook.llm_lab_io import export_lab, load_lab, save_lab, validate_lab


def document(**updates):
    return {"format": "mlcookbook-llm-lab", "version": 1, "state": {"tab": 0},
            "inspection": None, "capture": None, "benchmark": None, "capabilities": None, **updates}


def test_lab_roundtrip_is_source_free_and_checked(tmp_path):
    path = tmp_path / "research.mlcl"
    value = document(inspection={"path": "missing.gguf", "files": [{"path": "missing.gguf", "sha256": "a" * 64,
                         "size_bytes": 100}], "metadata": {"λ": "value"},
                         "tensors": [{"name": "λ", "shape": [6], "elements": 6, "dtype": "F32", "bytes": 24,
                                      "offset": 0, "group": "Other"}],
                         "summary": {"tensor_count": 1, "file_count": 1}, "warnings": []})
    digest = save_lab(path, value)
    assert load_lab(path) == value
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == ["document.json", "manifest.json"]
        assert hashlib.sha256(archive.read("document.json")).hexdigest() == digest


def test_atomic_failure_preserves_previous_archive(tmp_path, monkeypatch):
    path = tmp_path / "research.mlcl"
    save_lab(path, document())
    before = path.read_bytes()
    def fail(*args):
        raise OSError("publish failed")
    monkeypatch.setattr("mlcookbook.llm_lab_io.os.replace", fail)
    with pytest.raises(OSError, match="publish failed"):
        save_lab(path, document(state={"tab": 1}))
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m.update(version=2), "format/version"),
    (lambda m: m.update(version=True), "format/version"),
    (lambda m: m.update(document_sha256="0" * 64), "checksum"),
    (lambda m: m.update(document_bytes=99), "byte count"),
])
def test_lab_manifest_validation(tmp_path, mutate, match):
    path = tmp_path / "bad.mlcl"
    raw = b'{"view":"inspect"}'
    manifest = {"format": "mlcookbook-llm-lab", "version": 1, "document_sha256": hashlib.sha256(raw).hexdigest(), "document_bytes": len(raw)}
    mutate(manifest)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("document.json", raw)
    with pytest.raises(ValueError, match=match):
        load_lab(path)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'[]'])
def test_lab_rejects_duplicate_nonfinite_or_nonobject_json(tmp_path, raw):
    path = tmp_path / "bad.mlcl"
    manifest = {"format": "mlcookbook-llm-lab", "version": 1, "document_sha256": hashlib.sha256(raw).hexdigest(), "document_bytes": len(raw)}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("document.json", raw)
    with pytest.raises(ValueError):
        load_lab(path)


def test_unknown_members_and_invalid_values_cannot_replace_good_save(tmp_path):
    path = tmp_path / "lab.mlcl"
    save_lab(path, document())
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="finite JSON"):
        save_lab(path, document(state={"value": float("inf")}))
    assert path.read_bytes() == previous
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("unexpected.py", "do not execute")
    with pytest.raises(ValueError, match="exactly"):
        load_lab(path)


def test_export_is_actual_json_with_the_same_validated_snapshot(tmp_path):
    path = tmp_path / "lab.json"
    value = document(state={"tab": 2, "connection": {"base_url": "unrun invalid edit", "model": "model"},
                            "graph_view": {"scale": 1.5, "center": [-5.0, 9.5]}, "graph_trace": None})
    digest = export_lab(path, value)
    assert json.loads(path.read_text(encoding="utf-8")) == value
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert not zipfile.is_zipfile(path)


@pytest.mark.parametrize("state", [
    {"capture_tokens": True}, {"capture_tokens": 17}, {"repetitions": 0}, {"graph_phase": -1},
    {"cache_policy": "cold"}, {"graph_trace": [1]}, {"graph_view": {"scale": 0, "center": [0, 0]}},
    {"graph_view": {"scale": 1, "center": [0, 1e10]}}, {"connection": {"base_url": 1, "model": "x"}},
    {"unknown": "unsupported"}, {"trial_selected": 111},
])
def test_unrepresentable_settings_are_rejected_before_overwriting(tmp_path, state):
    path = tmp_path / "lab.mlcl"
    save_lab(path, document())
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        save_lab(path, document(state=state))
    assert path.read_bytes() == previous


@pytest.mark.parametrize("value", [document(version=True), document(version=2), document(format="other"),
                                   document(inspection={"tensors": [None]}),
                                   document(benchmark={"trials": "wrong"}),
                                   document(capabilities={"capabilities": []})])
def test_invalid_artifacts_and_schema_do_not_pass_validation(value):
    with pytest.raises(ValueError):
        validate_lab(value)


def test_real_benchmark_roundtrip_and_inconsistent_summary_rejected(llamacpp_server, tmp_path):
    from copy import deepcopy

    from mlcookbook.llm_lab import benchmark_llamacpp, probe_llamacpp

    result = benchmark_llamacpp(base_url=llamacpp_server.base_url, model="test-gguf", prompt="hello",
                               repetitions=1, warmups=0).to_dict()
    value = document(benchmark=result, capabilities=probe_llamacpp(llamacpp_server.base_url).to_dict())
    path = tmp_path / "trial.mlcl"
    save_lab(path, value)
    assert load_lab(path) == value
    broken = deepcopy(value)
    broken["benchmark"]["summary"]["completed_measured_trials"] += 1
    with pytest.raises(ValueError, match="summary"):
        validate_lab(broken)


def test_comparison_artifacts_roundtrip_without_opening_a_dialog_or_source(tmp_path):
    model_comparison = {"summary": "One type change", "rows": [{"name": "weight", "change": "changed",
                          "left_shape": [256, 2], "right_shape": [256, 2], "left_dtype": "F32", "right_dtype": "Q4_K",
                          "left_bytes": 2048, "right_bytes": 288}],
                        "metadata_changes": [{"key": "name", "left": "a", "right": "b"}], "warnings": []}
    graph_comparison = {"summary": "A shape changes", "left_graph": "prefill", "right_graph": "decode",
                        "rows": [{"name": "hybrid-state", "change": "shape", "left_shape": [0, 1], "right_shape": [1, 1]}],
                        "warnings": ["Exact names only"]}
    value = document(model_comparison=model_comparison, graph_comparison=graph_comparison)
    path = tmp_path / "comparisons.mlcl"
    save_lab(path, value)
    assert load_lab(path) == value
    value["model_comparison"]["rows"][0]["left_bytes"] = -1
    with pytest.raises(ValueError, match="storage"):
        validate_lab(value)


def test_optional_comparison_members_are_backward_compatible_and_strict(tmp_path):
    for value in (document(), document(model_comparison=None, graph_comparison=None)):
        path = tmp_path / "legacy.mlcl"
        save_lab(path, value)
        assert load_lab(path) == value
    with pytest.raises(ValueError, match="comparison"):
        validate_lab(document(graph_comparison={"summary": "bad", "rows": None, "warnings": []}))
