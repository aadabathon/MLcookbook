"""Comparing a numeric variable across groups: one-way ANOVA and Kruskal-Wallis.

One-way ANOVA tests whether group means differ, assuming roughly normal
values with equal variances in every group. Kruskal-Wallis is its rank-based
fallback: it only asks whether one group tends to have larger values. The
result runs both, checks the assumptions (Shapiro-Wilk per group, Levene for
equal variances), and says which test to trust.

Effect size: eta squared = between-group sum of squares / total sum of squares,
the share of the variance explained by group membership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType

ALPHA = 0.05
SMALL_GROUP = 30  # below this, non-normality matters for ANOVA


@dataclass(frozen=True)
class GroupSummary(Result):
    level: Any
    n_obs: int
    mean: float
    std: float
    median: float


@dataclass(frozen=True)
class AnovaResult(Result):
    method: str
    value: str
    group: str
    n_obs: int
    groups: list[GroupSummary]
    f_statistic: float
    p_value: float  # ANOVA
    eta_squared: float
    kruskal_statistic: float
    kruskal_p_value: float
    levene_p_value: float  # small: group variances differ
    normality_ok: bool  # no small group fails Shapiro-Wilk
    recommended_test: str  # "anova" or "kruskal"
    warnings: list[str] = field(default_factory=list)


def anova(dataset: Dataset, value: str, group: str) -> AnovaResult:
    """Compare numeric column *value* across the levels of categorical/boolean *group*.

    Rows missing either column are dropped, and levels with fewer than two rows
    are left out (with a warning).
    """
    schema = dataset.schema()
    if schema.var_type(value) is not VarType.NUMERIC:
        raise ValueError(f"anova requires a numeric value column; {value!r} is {schema.var_type(value).value}.")
    if schema.var_type(group) not in (VarType.CATEGORICAL, VarType.BOOLEAN):
        raise ValueError(
            f"anova requires a categorical or boolean group column; {group!r} is {schema.var_type(group).value}."
        )
    pairs = dataset.df[[value, group]].dropna()
    warnings: list[str] = []
    n_dropped = len(dataset.df) - len(pairs)
    if n_dropped:
        warnings.append(f"Dropped {n_dropped} row(s) with missing values in {value!r} or {group!r}.")
    samples = {level: rows[value].astype(float).to_numpy() for level, rows in pairs.groupby(group, observed=True)}
    tiny = [level for level, values in samples.items() if len(values) < 2]
    if tiny:
        warnings.append(f"Left out level(s) with fewer than 2 rows: {tiny}.")
        samples = {level: values for level, values in samples.items() if len(values) >= 2}
    if len(samples) < 2:
        raise ValueError(f"Need at least two levels of {group!r} with 2+ rows each; got {len(samples)}.")
    values = list(samples.values())
    if all(np.ptp(v) == 0 for v in values):
        raise ValueError(f"{value!r} does not vary within any group; the tests are undefined.")

    f_res = stats.f_oneway(*values)
    k_res = stats.kruskal(*values)
    levene_p = float(stats.levene(*values, center="median").pvalue)
    everything = np.concatenate(values)
    grand_mean = everything.mean()
    between = sum(len(v) * (v.mean() - grand_mean) ** 2 for v in values)
    total = ((everything - grand_mean) ** 2).sum()

    failing = [
        level for level, v in samples.items()
        if 3 <= len(v) < SMALL_GROUP and np.ptp(v) > 0 and stats.shapiro(v).pvalue < ALPHA
    ]
    normality_ok = not failing
    if failing:
        warnings.append(
            f"Small group(s) {failing} depart from normality (Shapiro-Wilk p < {ALPHA}); "
            "ANOVA's p-value is unreliable, so rely on Kruskal-Wallis."
        )
    if levene_p < ALPHA:
        warnings.append(
            f"Group variances differ (Levene p={levene_p:.3g}); ANOVA assumes equal variances, "
            "so rely on Kruskal-Wallis (or a Welch-type test)."
        )

    return AnovaResult(
        method="anova",
        value=value,
        group=group,
        n_obs=int(sum(len(v) for v in values)),
        groups=[
            GroupSummary(
                level=level.item() if hasattr(level, "item") else level,
                n_obs=len(v), mean=float(v.mean()), std=float(v.std(ddof=1)), median=float(np.median(v)),
            )
            for level, v in samples.items()
        ],
        f_statistic=float(f_res.statistic),
        p_value=float(f_res.pvalue),
        eta_squared=float(between / total) if total > 0 else float("nan"),
        kruskal_statistic=float(k_res.statistic),
        kruskal_p_value=float(k_res.pvalue),
        levene_p_value=levene_p,
        normality_ok=normality_ok,
        recommended_test="anova" if normality_ok and levene_p >= ALPHA else "kruskal",
        warnings=warnings,
    )
