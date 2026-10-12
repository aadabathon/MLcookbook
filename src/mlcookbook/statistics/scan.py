"""Pairwise association scan: the right test for every pair of columns.

This is where the schema pays off. For each pair, the column types pick the
test, and every test reports an effect size on a 0..1 scale:

==========================  =====================  ===========================
pair                        test                   effect size
==========================  =====================  ===========================
numeric - numeric           Spearman correlation   |rho|
categorical - categorical   chi-square             Cramér's V
numeric - categorical       Kruskal-Wallis         epsilon² = H / (n - 1)
==========================  =====================  ===========================

Rank-based tests keep the scan robust to outliers and skew without checking
assumptions pair by pair. Testing many pairs inflates false positives, so
p-values are also adjusted with the Benjamini-Hochberg procedure (false
discovery rate). Effect sizes from different tests are not directly
comparable; sort by them to find candidates, then look closer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import association

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType

SCAN_TYPES = (VarType.NUMERIC, VarType.CATEGORICAL, VarType.BOOLEAN)
MIN_OBSERVATIONS = 5
MAX_COLUMNS = 60


@dataclass(frozen=True)
class PairAssociation(Result):
    x: str
    y: str
    kind: str  # "numeric-numeric", "categorical-categorical", or "numeric-categorical"
    test: str  # "spearman", "chi_square", or "kruskal"
    statistic: float
    effect_measure: str  # "abs_spearman_rho", "cramers_v", or "epsilon_squared"
    effect_size: float  # 0..1
    p_value: float
    p_adjusted: float  # Benjamini-Hochberg
    n_obs: int


@dataclass(frozen=True)
class AssociationScanResult(Result):
    method: str
    columns: list[str]
    n_pairs: int
    pairs: list[PairAssociation]  # strongest effect first
    warnings: list[str] = field(default_factory=list)


def association_scan(dataset: Dataset, columns: list[str] | None = None) -> AssociationScanResult:
    """Test every pair of *columns* (default: all numeric/categorical/boolean columns)."""
    schema = dataset.schema()
    warnings: list[str] = []
    if columns is None:
        columns = [c.name for c in schema.columns if c.var_type in SCAN_TYPES]
        skipped = [c.name for c in schema.columns if c.var_type not in SCAN_TYPES]
        if skipped:
            warnings.append(f"Skipped columns that are not numeric/categorical/boolean: {skipped}.")
    else:
        if len(set(columns)) != len(columns):
            raise ValueError(f"Columns must be unique; got {columns}.")
        for name in columns:
            if schema.var_type(name) not in SCAN_TYPES:
                raise ValueError(f"Column {name!r} has type {schema.var_type(name).value}; scan needs "
                                 "numeric, categorical, or boolean columns.")
    if len(columns) < 2:
        raise ValueError("association_scan needs at least two columns.")
    if len(columns) > MAX_COLUMNS:
        raise ValueError(f"Scanning {len(columns)} columns is too many pairs; choose at most {MAX_COLUMNS}.")

    found = []
    for x, y in combinations(columns, 2):
        numeric = [schema.var_type(c) is VarType.NUMERIC for c in (x, y)]
        pairs = dataset.df[[x, y]].dropna()
        if len(pairs) < MIN_OBSERVATIONS or pairs[x].nunique() < 2 or pairs[y].nunique() < 2:
            warnings.append(f"Skipped {x!r} vs {y!r}: fewer than {MIN_OBSERVATIONS} complete rows or a constant column.")
            continue
        if all(numeric):
            found.append(_spearman(x, y, pairs))
        elif not any(numeric):
            found.append(_chi_square(x, y, pairs))
        else:
            value, group = (x, y) if numeric[0] else (y, x)
            found.append(_kruskal(x, y, value, group, pairs))
    found = [f for f in found if f is not None]
    if found:
        adjusted = stats.false_discovery_control([f["p_value"] for f in found])
        for f, p in zip(found, adjusted):
            f["p_adjusted"] = float(p)
    ordered = sorted(found, key=lambda f: -f["effect_size"] if f["effect_size"] == f["effect_size"] else 1.0)
    return AssociationScanResult(
        method="association_scan",
        columns=list(columns),
        n_pairs=len(ordered),
        pairs=[PairAssociation(**f) for f in ordered],
        warnings=warnings,
    )


def _spearman(x: str, y: str, pairs: pd.DataFrame) -> dict:
    res = stats.spearmanr(pairs[x].astype(float), pairs[y].astype(float))
    rho = float(res.statistic)
    return dict(x=x, y=y, kind="numeric-numeric", test="spearman", statistic=rho,
                effect_measure="abs_spearman_rho", effect_size=abs(rho), p_value=float(res.pvalue),
                n_obs=len(pairs))


def _chi_square(x: str, y: str, pairs: pd.DataFrame) -> dict:
    table = pd.crosstab(pairs[x], pairs[y]).to_numpy()
    res = stats.chi2_contingency(table)
    return dict(x=x, y=y, kind="categorical-categorical", test="chi_square", statistic=float(res.statistic),
                effect_measure="cramers_v", effect_size=float(association(table, method="cramer")),
                p_value=float(res.pvalue), n_obs=len(pairs))


def _kruskal(x: str, y: str, value: str, group: str, pairs: pd.DataFrame) -> dict | None:
    samples = [rows[value].astype(float).to_numpy() for _, rows in pairs.groupby(group, observed=True)]
    everything = np.concatenate(samples)
    if np.ptp(everything) == 0:
        return None
    res = stats.kruskal(*samples)
    n = len(pairs)
    return dict(x=x, y=y, kind="numeric-categorical", test="kruskal", statistic=float(res.statistic),
                effect_measure="epsilon_squared", effect_size=float(res.statistic) / (n - 1),
                p_value=float(res.pvalue), n_obs=n)
