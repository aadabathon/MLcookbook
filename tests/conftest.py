import numpy as np
import pandas as pd
import pytest


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
