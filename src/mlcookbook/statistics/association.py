"""Association between two numeric variables (Pearson/Spearman correlation).

This module is the template for the other tests in ``statistics/``: validate
applicability against the schema, compute with scipy, and return a frozen
result dataclass with explicit warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType

CorrelationMethod = Literal["pearson", "spearman"]

MIN_OBSERVATIONS = 3
NORMALITY_ALPHA = 0.05
SHAPIRO_MAX_ROWS = 5000


@dataclass(frozen=True)
class CorrelationResult(Result):
    method: str
    x: str
    y: str
    statistic: float
    p_value: float
    n_obs: int
    confidence_level: float
    ci_low: float  # NaN (-> null) when n_obs <= 3
    ci_high: float
    warnings: list[str] = field(default_factory=list)


def correlation(
    dataset: Dataset,
    x: str,
    y: str,
    method: CorrelationMethod = "pearson",
    confidence_level: float = 0.95,
) -> CorrelationResult:
    """Correlate two numeric columns, dropping rows where either is missing.

    Pearson measures linear association and assumes roughly normal marginals
    for its p-value and interval to be exact; Spearman is rank-based and only
    assumes a monotonic relationship.

    The confidence interval uses the Fisher z-transform, z = artanh(r), which
    is approximately normal with standard error 1/sqrt(n-3) for Pearson and
    sqrt(1.06/(n-3)) for Spearman (Fieller, Hartley & Pearson, 1957).
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method {method!r}; expected 'pearson' or 'spearman'.")
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1); got {confidence_level}.")
    schema = dataset.schema()
    for name in (x, y):
        var_type = schema.var_type(name)  # raises KeyError for unknown columns
        if var_type is not VarType.NUMERIC:
            raise ValueError(
                f"correlation requires numeric columns; {name!r} is {var_type.value}. "
                "Use Dataset.set_type() if the inferred type is wrong."
            )

    pairs = dataset.df[[x, y]].dropna().astype(float)
    warnings: list[str] = []
    n_dropped = len(dataset.df) - len(pairs)
    if n_dropped:
        warnings.append(f"Dropped {n_dropped} row(s) with missing values in {x!r} or {y!r}.")
    if len(pairs) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Need at least {MIN_OBSERVATIONS} complete observations, got {len(pairs)}."
        )
    constant = [name for name in (x, y) if pairs[name].nunique() <= 1]
    for name in constant:
        warnings.append(f"Column {name!r} is constant; correlation is undefined.")

    if method == "pearson":
        res = stats.pearsonr(pairs[x], pairs[y])
        if not constant:
            warnings.extend(_normality_warnings(pairs, (x, y)))
    else:
        res = stats.spearmanr(pairs[x], pairs[y])
    r = float(res.statistic)
    ci_low, ci_high = _fisher_interval(r, len(pairs), method, confidence_level)

    return CorrelationResult(
        method=method,
        x=x,
        y=y,
        statistic=r,
        p_value=float(res.pvalue),
        n_obs=len(pairs),
        confidence_level=confidence_level,
        ci_low=ci_low,
        ci_high=ci_high,
        warnings=warnings,
    )


def _fisher_interval(r: float, n: int, method: str, confidence_level: float) -> tuple[float, float]:
    if n <= 3 or r != r:
        return float("nan"), float("nan")
    variance = 1.0 if method == "pearson" else 1.06
    half_width = stats.norm.ppf(0.5 + confidence_level / 2) * np.sqrt(variance / (n - 3))
    with np.errstate(divide="ignore"):
        z = np.arctanh(np.clip(r, -1.0, 1.0))
    return float(np.tanh(z - half_width)), float(np.tanh(z + half_width))


def _normality_warnings(pairs, names) -> list[str]:
    warnings = []
    for name in names:
        values = pairs[name]
        test = stats.shapiro if len(values) <= SHAPIRO_MAX_ROWS else stats.normaltest
        p = float(test(values).pvalue)
        if p < NORMALITY_ALPHA:
            warnings.append(
                f"{name!r} departs from normality (p={p:.3g}, skew {float(stats.skew(values)):.2f}); "
                "Pearson's p-value and interval assume roughly normal data. Spearman is more robust."
            )
    return warnings
