"""Statistical association and hypothesis-testing workflows.

Implemented: numeric-numeric correlation.
Planned (see ROADMAP.md): chi-square, Cramér's V, ANOVA, nonparametric tests,
mutual information, effect sizes, confidence intervals.
"""

from mlcookbook.statistics.association import CorrelationResult, correlation

__all__ = ["CorrelationResult", "correlation"]
