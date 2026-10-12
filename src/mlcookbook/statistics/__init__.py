"""Statistical association and hypothesis-testing workflows.

Implemented: correlation (Pearson/Spearman with confidence intervals),
chi-square + Cramér's V, one-way ANOVA + Kruskal-Wallis, and a pairwise
association scan that picks the test from the column types.
"""

from mlcookbook.statistics.association import CorrelationResult, correlation
from mlcookbook.statistics.categorical import ChiSquareResult, chi_square
from mlcookbook.statistics.groups import AnovaResult, GroupSummary, anova
from mlcookbook.statistics.scan import AssociationScanResult, PairAssociation, association_scan

__all__ = [
    "AnovaResult",
    "AssociationScanResult",
    "ChiSquareResult",
    "CorrelationResult",
    "GroupSummary",
    "PairAssociation",
    "anova",
    "association_scan",
    "chi_square",
    "correlation",
]
