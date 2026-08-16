import numpy as np
import pandas as pd
import pytest


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
