"""LLM integration uses local safetensors, real tokenization and real decoding."""

import json
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from mlcookbook import Dataset, generate_text
from mlcookbook.llm import _model_files


def test_llm_local_generation_serializes_and_is_frozen(tiny_llm, monkeypatch):
    import huggingface_hub

    def no_network(*args, **kwargs):
        pytest.fail("Local generation must not download anything")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)
    result = generate_text(model_path=tiny_llm, prompt="hello world", max_new_tokens=4, prompt_format="plain")
    assert result.model_type == "gpt2" and result.device == "cpu"
    assert result.parameter_count > 0 and result.context_window == 64
    assert result.outputs[0].input_tokens == 2
    assert 1 <= result.outputs[0].generated_tokens <= 4
    assert result.outputs[0].row_position is None
    assert result.outputs[0].formatted_prompt == "hello world"
    assert result.outputs[0].finish_reason in ("eos", "token_limit")
    assert len(result.model_sha256) == 64 and result.versions["torch"]
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["outputs"][0]["input_text"] == "hello world"
    with pytest.raises(FrozenInstanceError):
        result.device = "cuda"


def test_llm_sampling_repeats_without_changing_callers_rng(tiny_llm):
    import torch

    before = torch.random.get_rng_state().clone()
    kwargs = dict(model_path=tiny_llm, prompt="hello", temperature=0.8, random_state=6,
                  max_new_tokens=6, prompt_format="plain")
    first = generate_text(**kwargs)
    assert torch.equal(before, torch.random.get_rng_state())
    second = generate_text(**kwargs)
    assert first.outputs[0].text == second.outputs[0].text
    assert first.outputs[0].generated_tokens == second.outputs[0].generated_tokens


def test_llm_chat_template_is_applied_once(tiny_llm):
    result = generate_text(model_path=tiny_llm, prompt="hello", max_new_tokens=2)
    assert result.prompt_format == "chat"
    assert result.outputs[0].formatted_prompt == "user hello assistant "
    assert result.outputs[0].input_tokens == 3


def test_llm_batch_keeps_positions_skips_missing_and_bounds_rows(tiny_llm):
    frame = pd.DataFrame({"text": pd.Categorical(["hello", None, " ", "world", "five"])}, index=[9, 9, 9, 2, 2])
    result = generate_text(Dataset(frame), model_path=tiny_llm, prompt="answer", text_column="text",
                           max_rows=2, max_new_tokens=2, prompt_format="plain")
    assert [output.row_position for output in result.outputs] == [0, 3]
    assert result.outputs[1].formatted_prompt == "answer\n\nworld"
    assert any("2 missing or blank" in warning for warning in result.warnings)
    assert any("first 2" in warning for warning in result.warnings)


@pytest.mark.parametrize("kwargs,match", [
    ({"prompt": ""}, "Write a prompt"),
    ({"temperature": float("nan")}, "temperature"),
    ({"top_p": 0}, "top_p"),
    ({"max_new_tokens": 0}, "max_new_tokens"),
    ({"max_rows": True}, "max_rows"),
    ({"random_state": -1}, "random_state"),
    ({"max_seconds": 0}, "max_seconds"),
    ({"device": "tpu"}, "device"),
    ({"prompt_format": "invalid"}, "prompt_format"),
])
def test_llm_invalid_settings_fail_before_model_load(kwargs, match):
    with pytest.raises(ValueError, match=match):
        generate_text(**dict({"model_path": "", "prompt": "hello"}, **kwargs))


def test_llm_wrong_roles_and_nontext_are_rejected():
    numeric = pd.DataFrame({"text": [1, 2, 3]})
    with pytest.raises(ValueError, match="role"):
        generate_text(Dataset(numeric), model_path="", prompt="", text_column="text")
    with pytest.raises(ValueError, match="non-text"):
        generate_text(Dataset(numeric, overrides={"text": "categorical"}), model_path="", prompt="", text_column="text")
    with pytest.raises(ValueError, match="non-empty"):
        generate_text(Dataset(pd.DataFrame({"text": [None]}), overrides={"text": "text"}),
                      model_path="", prompt="", text_column="text")


@pytest.mark.parametrize("kwargs", [{"max_input_tokens": 1}, {"max_new_tokens": 64}])
def test_llm_context_limits_are_explicit(tiny_llm, kwargs):
    with pytest.raises(ValueError, match="no text was truncated"):
        generate_text(model_path=tiny_llm, prompt="hello world", **kwargs)


def test_llm_bad_model_and_unsupported_checkpoint_are_actionable(tmp_path):
    with pytest.raises(ValueError, match="GGUF"):
        generate_text(model_path=tmp_path / "model.gguf", prompt="hello")
    (tmp_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Supported model types"):
        generate_text(model_path=tmp_path, prompt="hello")
    (tmp_path / "config.json").write_text('{"model_type": "gpt2"}', encoding="utf-8")
    with pytest.raises(ValueError, match="safetensors"):
        generate_text(model_path=tmp_path, prompt="hello")


def test_llm_fingerprint_detects_changed_weights(tiny_llm, tmp_path):
    import shutil

    copied = tmp_path / "copy"
    shutil.copytree(tiny_llm, copied)
    before = _model_files(copied)[2]
    with (copied / "model.safetensors").open("ab") as file:
        file.write(b"changed")
    assert _model_files(copied)[2] != before


def test_llm_cuda_unavailable_is_actionable(tiny_llm, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        generate_text(model_path=tiny_llm, prompt="hello", device="cuda")


def test_llm_cancel_during_decoding_restores_rng(tiny_llm):
    import torch

    calls = []

    def cancel():
        calls.append(True)
        return len(calls) >= 4

    before = torch.random.get_rng_state().clone()
    with pytest.raises(InterruptedError, match="cancelled"):
        generate_text(model_path=tiny_llm, prompt="hello", max_new_tokens=10, stop_requested=cancel)
    assert len(calls) >= 4
    assert torch.equal(before, torch.random.get_rng_state())


def test_llm_core_import_does_not_initialize_optional_runtimes():
    import subprocess
    import sys

    completed = subprocess.run([sys.executable, "-c", "import sys; import mlcookbook; "
                                "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
