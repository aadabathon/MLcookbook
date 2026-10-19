"""Bounded, offline causal language-model generation with PyTorch.

Weights stay outside results and experiment files. Imports are lazy so the
classical recipes do not require the optional ``llm`` extra or initialize torch.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType

MODEL_TYPES = ("gpt2", "llama", "qwen2", "qwen3")


@dataclass(frozen=True)
class GeneratedText(Result):
    row_position: int | None
    input_text: str
    formatted_prompt: str
    text: str
    input_tokens: int
    generated_tokens: int
    seconds: float
    finish_reason: str


@dataclass(frozen=True)
class TextGenerationResult(Result):
    method: str
    model_path: str
    model_type: str
    model_sha256: str
    parameter_count: int
    context_window: int
    device: str
    dtype: str
    prompt: str
    text_column: str | None
    prompt_format: str
    settings: dict
    versions: dict[str, str]
    load_seconds: float
    outputs: list[GeneratedText]
    warnings: list[str]


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}.")


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number between {low} and {high}.")


def _inputs(dataset, prompt, text_column, max_rows):
    if not isinstance(prompt, str) or len(prompt) > 16000:
        raise ValueError("prompt must be text containing at most 16000 characters.")
    if text_column is None:
        if not prompt.strip():
            raise ValueError("Write a prompt before generating text.")
        return [(None, prompt)], []
    if dataset is None or text_column not in dataset.df.columns:
        raise ValueError("Choose a text column from the dataset.")
    if dataset.schema().var_type(text_column) not in (VarType.TEXT, VarType.CATEGORICAL):
        raise ValueError("The input column must have a text or categorical role and contain strings.")
    values = dataset.df[text_column].astype(object)
    if not values.dropna().map(lambda value: isinstance(value, str)).all():
        raise ValueError("The input column contains non-text values; values are not converted silently.")
    valid = values.notna() & values.fillna("").map(lambda value: bool(value.strip()))
    rows = [(position, value) for position, (value, keep) in enumerate(zip(values, valid)) if keep]
    if not rows:
        raise ValueError("The input column has no non-empty text rows.")
    warnings = []
    if missing := len(values) - len(rows):
        warnings.append(f"Skipped {missing} missing or blank input row(s).")
    if len(rows) > max_rows:
        warnings.append(f"Generated for the first {max_rows} non-empty rows of {len(rows)}; increase the row limit to include more.")
    if any(len(text) > 16000 for _, text in rows[:max_rows]):
        raise ValueError("Each input row must contain at most 16000 characters.")
    return rows[:max_rows], warnings


def _model_files(model_path):
    if not isinstance(model_path, (str, Path)) or not str(model_path).strip():
        raise ValueError("Choose a local Hugging Face model folder containing config.json and safetensors weights.")
    directory = Path(model_path).expanduser().resolve()
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise ValueError("Choose a local Hugging Face model folder with config.json; GGUF and Hub IDs are not supported.")
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Cannot read the model configuration: {exc}") from exc
    if not isinstance(config, dict) or config.get("model_type") not in MODEL_TYPES:
        raise ValueError(f"Supported model types: {', '.join(MODEL_TYPES)}. Choose a compatible causal language model.")
    if config.get("quantization_config"):
        raise ValueError("Choose an unquantized safetensors model. This recipe loads float32 on CPU or float16 on CUDA.")
    if not (directory / "model.safetensors").is_file() and not (directory / "model.safetensors.index.json").is_file():
        raise ValueError("This recipe requires model.safetensors or a sharded model.safetensors.index.json checkpoint.")
    files = [path for path in directory.rglob("*")
             if path.is_file() and (path.suffix in (".json", ".safetensors", ".model", ".jinja")
                                    or path.name in ("vocab.txt", "merges.txt"))]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return directory, config, digest.hexdigest()


def _load_model(directory, model_type, device):
    import torch
    from transformers import (
        AutoTokenizer,
        GPT2LMHeadModel,
        LlamaForCausalLM,
        Qwen2ForCausalLM,
        Qwen3ForCausalLM,
    )

    models = {"gpt2": GPT2LMHeadModel, "llama": LlamaForCausalLM,
              "qwen2": Qwen2ForCausalLM, "qwen3": Qwen3ForCausalLM}
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    dtype = torch.float32 if device == "cpu" else torch.float16
    with torch.random.fork_rng(devices=[]):
        model = models[model_type].from_pretrained(str(directory), local_files_only=True,
                                                 trust_remote_code=False, use_safetensors=True,
                                                 dtype=dtype, attn_implementation="eager")
    return tokenizer, model.to(device).eval()


def generate_text(
    dataset: Dataset | None = None,
    *,
    model_path: str | Path,
    prompt: str,
    text_column: str | None = None,
    max_rows: int = 1,
    max_new_tokens: int = 128,
    max_input_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    random_state: int = 0,
    device: str = "cpu",
    prompt_format: str = "auto",
    max_seconds: float = 30.0,
    stop_requested: Callable[[], bool] | None = None,
) -> TextGenerationResult:
    """Generate locally from a prompt, optionally followed by each selected text row.

    ``auto`` uses the tokenizer's chat template when present; ``plain`` sends
    raw text. Temperature zero is greedy decoding. Limits apply per output.
    No network downloads, training, sklearn preprocessing, or CV are performed.
    The time limit is checked between decoding steps, so it is not a hard
    deadline for model loading or an individual forward pass.
    """
    _integer(max_rows, "max_rows", 1, 10)
    _integer(max_new_tokens, "max_new_tokens", 1, 512)
    _integer(max_input_tokens, "max_input_tokens", 1, 8192)
    _integer(random_state, "random_state", 0, 2147483647)
    _number(temperature, "temperature", 0.0, 2.0)
    _number(top_p, "top_p", 0.01, 1.0)
    _number(max_seconds, "max_seconds", 1.0, 120.0)
    if device not in ("cpu", "cuda") or prompt_format not in ("auto", "plain", "chat"):
        raise ValueError("device must be cpu or cuda; prompt_format must be auto, plain, or chat.")
    if stop_requested is not None and not callable(stop_requested):
        raise ValueError("stop_requested must be a callable or None.")

    def check_cancelled():
        if stop_requested is not None and stop_requested():
            raise InterruptedError("Text generation cancelled.")

    check_cancelled()
    rows, warnings = _inputs(dataset, prompt, text_column, max_rows)
    directory, config, fingerprint = _model_files(model_path)
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError('Install the optional LLM runtime with pip install -e ".[llm]".') from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch installation. Choose CPU or install a CUDA-enabled PyTorch build.")
    started = perf_counter()
    tokenizer, model = _load_model(directory, config["model_type"], device)
    load_seconds = perf_counter() - started
    check_cancelled()
    context_window = int(getattr(model.config, "max_position_embeddings", None)
                         or getattr(model.config, "n_positions", 2048))
    chat = prompt_format == "chat" or (prompt_format == "auto" and bool(tokenizer.chat_template))
    if chat and not tokenizer.chat_template:
        raise ValueError("This tokenizer has no chat template. Choose Plain prompt format.")
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos[0] if isinstance(eos, list) else eos
    if pad is None:
        raise ValueError("The tokenizer/model needs a pad or end-of-sequence token for generation.")
    from transformers import GenerationConfig, StoppingCriteria

    class CancelGeneration(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return stop_requested is not None and stop_requested()

    settings = {"max_rows": max_rows, "max_new_tokens": max_new_tokens,
                "max_input_tokens": max_input_tokens, "temperature": temperature,
                "top_p": top_p, "random_state": random_state, "max_seconds": max_seconds}
    generation = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                  "max_time": max_seconds, "pad_token_id": pad, "eos_token_id": eos,
                  "use_cache": True}
    if temperature > 0:
        generation.update(temperature=temperature, top_p=top_p)
    prepared = []
    for position, text in rows:
        combined = prompt.strip() + "\n\n" + text if text_column and prompt.strip() else text
        if chat:
            # Tokenizing once avoids duplicating special tokens around chat templates.
            formatted = tokenizer.apply_chat_template([{"role": "user", "content": combined}],
                                                       tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        else:
            formatted = combined
            encoded = tokenizer(formatted, return_tensors="pt")
        input_count = int(encoded["input_ids"].shape[1])
        if input_count == 0 or input_count > max_input_tokens or input_count + max_new_tokens > context_window:
            raise ValueError(f"Prompt needs {input_count} input tokens plus {max_new_tokens} output tokens. "
                             f"Input limit: {max_input_tokens}; model context: {context_window}. "
                             "Shorten the prompt or adjust the token limits; no text was truncated.")
        prepared.append((position, text, formatted, encoded, input_count))
    outputs = []
    cuda_devices = [torch.cuda.current_device()] if device == "cuda" else []
    # Do not change the caller's random stream, including when generation fails.
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        torch.random.default_generator.manual_seed(random_state)
        if device == "cuda":
            torch.cuda.manual_seed(random_state)
        for position, text, formatted, encoded, input_count in prepared:
            check_cancelled()
            encoded = {key: value.to(device) for key, value in encoded.items()}
            started = perf_counter()
            tokens = model.generate(**encoded, generation_config=GenerationConfig(**generation),
                                    stopping_criteria=[CancelGeneration()])[0, input_count:]
            check_cancelled()
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = perf_counter() - started
            ids = tokens.tolist()
            eos_ids = eos if isinstance(eos, list) else [eos]
            reason = "eos" if ids and ids[-1] in eos_ids else "token_limit" if len(ids) >= max_new_tokens else "time_limit"
            if reason == "time_limit":
                warnings.append(f"Output for row {position if position is not None else 'prompt'} reached the decoding time limit.")
            outputs.append(GeneratedText(position, text, formatted, tokenizer.decode(tokens, skip_special_tokens=True),
                                         input_count, len(ids), seconds, reason))
    warnings.append("Generated text is a model prediction and may contain factual errors. No labeled evaluation was performed.")
    warnings.append("Experiment files save prompts and outputs, not model weights. Reruns need the same local model folder.")
    return TextGenerationResult(
        method="generate_text", model_path=str(directory), model_type=config["model_type"], model_sha256=fingerprint,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()), context_window=context_window,
        device=device, dtype=str(next(model.parameters()).dtype), prompt=prompt, text_column=text_column,
        prompt_format="chat" if chat else "plain", settings=settings,
        versions={"torch": torch.__version__, "transformers": transformers.__version__},
        load_seconds=load_seconds, outputs=outputs, warnings=warnings,
    )
