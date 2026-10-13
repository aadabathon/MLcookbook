"""Regression workflows.

Implemented: linear regression (OLS, ridge, lasso) with residual diagnostics,
decision tree regression, random forest regression, support vector
regression, and neural network regression.
"""

from mlcookbook.regression.diagnostics import ResidualDiagnostics, residual_diagnostics
from mlcookbook.regression.forest import RandomForestRegressionResult, random_forest_regression
from mlcookbook.regression.linear import LinearRegressionResult, linear_regression
from mlcookbook.regression.neural_network import NeuralNetworkRegressionResult, neural_network_regression
from mlcookbook.regression.svm import SVMRegressionResult, svm_regression
from mlcookbook.regression.tree import RegressionTreeResult, decision_tree_regression

__all__ = [
    "LinearRegressionResult",
    "NeuralNetworkRegressionResult",
    "RandomForestRegressionResult",
    "RegressionTreeResult",
    "ResidualDiagnostics",
    "SVMRegressionResult",
    "decision_tree_regression",
    "linear_regression",
    "neural_network_regression",
    "random_forest_regression",
    "residual_diagnostics",
    "svm_regression",
]
