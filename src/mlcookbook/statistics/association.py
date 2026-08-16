"""Association analyses between pairs of variables.

Currently only numeric-numeric correlation (Pearson/Spearman). This module is
the template for future tests (chi-square, Cramér's V, ANOVA, mutual
information): validate applicability against the schema, compute with scipy,
and return a frozen result dataclass with explicit warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from scipy import stats

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType

CorrelationMethod = Literal["pearson", "spearman"]

MIN_OBSERVATIONS = 3


@dataclass(frozen=True)
class CorrelationResult(Result):
    method: str
    x: str
    y: str
    statistic: float
    p_value: float
    n_obs: int
    warnings: list[str] = field(default_factory=list)


def correlation(
    dataset: Dataset,
    x: str,
    y: str,
    method: CorrelationMethod = "pearson",
) -> CorrelationResult:
    """Correlate two numeric columns, dropping rows where either is missing.

    Pearson measures linear association and assumes roughly normal marginals
    for its p-value to be exact; Spearman is rank-based and only assumes a
    monotonic relationship.
    """
    schema = dataset.schema()
    for name in (x, y):
        var_type = schema.var_type(name)  # raises KeyError for unknown columns
        if var_type is not VarType.NUMERIC:
            raise ValueError(
                f"correlation requires numeric columns; {name!r} is {var_type.value}. "
                "Use Dataset.set_type() if the inferred type is wrong."
            )

    pairs = dataset.df[[x, y]].dropna()
    warnings: list[str] = []
    n_dropped = len(dataset.df) - len(pairs)
    if n_dropped:
        warnings.append(f"Dropped {n_dropped} row(s) with missing values in {x!r} or {y!r}.")
    if len(pairs) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Need at least {MIN_OBSERVATIONS} complete observations, got {len(pairs)}."
        )
    for name in (x, y):
        if pairs[name].nunique() <= 1:
            warnings.append(f"Column {name!r} is constant; correlation is undefined.")

    if method == "pearson":
        res = stats.pearsonr(pairs[x], pairs[y])
    elif method == "spearman":
        res = stats.spearmanr(pairs[x], pairs[y])
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'pearson' or 'spearman'.")

    return CorrelationResult(
        method=method,
        x=x,
        y=y,
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        n_obs=len(pairs),
        warnings=warnings,
    )
