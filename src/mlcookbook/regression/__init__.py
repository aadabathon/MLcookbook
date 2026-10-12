"""Regression workflows.

Implemented: linear regression (OLS, ridge, lasso) with residual diagnostics,
decision tree regression, and random forest regression.
"""

from mlcookbook.regression.diagnostics import ResidualDiagnostics, residual_diagnostics
from mlcookbook.regression.forest import RandomForestRegressionResult, random_forest_regression
from mlcookbook.regression.linear import LinearRegressionResult, linear_regression
from mlcookbook.regression.tree import RegressionTreeResult, decision_tree_regression

__all__ = [
    "LinearRegressionResult",
    "RandomForestRegressionResult",
    "RegressionTreeResult",
    "ResidualDiagnostics",
    "decision_tree_regression",
    "linear_regression",
    "random_forest_regression",
    "residual_diagnostics",
]
