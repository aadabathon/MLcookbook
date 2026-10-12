"""Association between two categorical variables: chi-square test and Cramér's V.

The chi-square test compares the observed contingency table with the counts
expected if the two variables were independent. Its p-value answers "is there
any association?"; Cramér's V (0 = none, 1 = perfect) answers "how strong?".
With many rows, tiny, unimportant associations become significant, so read
both. The chi-square approximation needs reasonably large expected counts; for
2x2 tables the result also carries Fisher's exact test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from scipy import stats
from scipy.stats.contingency import association

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType

CATEGORICAL_TYPES = (VarType.CATEGORICAL, VarType.BOOLEAN)
MIN_EXPECTED = 5
MAX_LOW_EXPECTED_SHARE = 0.20  # Cochran's rule: at most 20% of cells may expect fewer than 5


@dataclass(frozen=True)
class ChiSquareResult(Result):
    method: str
    x: str
    y: str
    statistic: float
    p_value: float
    dof: int
    cramers_v: float
    n_obs: int
    x_levels: list[Any]  # table rows
    y_levels: list[Any]  # table columns
    observed: list[list[int]]
    expected: list[list[float]]
    min_expected: float
    fisher_p_value: float | None  # 2x2 tables only
    warnings: list[str] = field(default_factory=list)


def chi_square(dataset: Dataset, x: str, y: str) -> ChiSquareResult:
    """Chi-square test of independence between two categorical/boolean columns.

    Rows missing either value are dropped. 2x2 tables use Yates' continuity
    correction for the test statistic (scipy's default); Cramér's V is computed
    from the uncorrected statistic.
    """
    schema = dataset.schema()
    if x == y:
        raise ValueError("chi_square needs two different columns.")
    for name in (x, y):
        var_type = schema.var_type(name)  # raises KeyError for unknown columns
        if var_type not in CATEGORICAL_TYPES:
            raise ValueError(
                f"chi_square requires categorical or boolean columns; {name!r} is {var_type.value}. "
                "Use Dataset.set_type() if the column holds category codes."
            )
    pairs = dataset.df[[x, y]].dropna()
    warnings: list[str] = []
    n_dropped = len(dataset.df) - len(pairs)
    if n_dropped:
        warnings.append(f"Dropped {n_dropped} row(s) with missing values in {x!r} or {y!r}.")
    table = pd.crosstab(pairs[x], pairs[y])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(f"Both columns need at least two observed levels; table shape is {table.shape}.")

    res = stats.chi2_contingency(table.to_numpy())
    expected = res.expected_freq
    low_share = float((expected < MIN_EXPECTED).mean())
    fisher_p = None
    if table.shape == (2, 2):
        fisher_p = float(stats.fisher_exact(table.to_numpy()).pvalue)
    if low_share > MAX_LOW_EXPECTED_SHARE:
        advice = "use Fisher's exact p-value" if fisher_p is not None else "merge rare levels or collect more data"
        warnings.append(
            f"{low_share:.0%} of cells expect fewer than {MIN_EXPECTED} rows; the chi-square approximation "
            f"is unreliable here, so {advice}."
        )
    return ChiSquareResult(
        method="chi_square",
        x=x,
        y=y,
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        dof=int(res.dof),
        cramers_v=float(association(table.to_numpy(), method="cramer")),
        n_obs=len(pairs),
        x_levels=[v.item() if hasattr(v, "item") else v for v in table.index],
        y_levels=[v.item() if hasattr(v, "item") else v for v in table.columns],
        observed=table.to_numpy().astype(int).tolist(),
        expected=expected.tolist(),
        min_expected=float(expected.min()),
        fisher_p_value=fisher_p,
        warnings=warnings,
    )
