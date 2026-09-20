"""ML Cookbook: a statistics and classical-ML workbench for tabular data.

Public API:

    from mlcookbook import Dataset, VarType, correlation, linear_regression, logistic_regression

Everything else lives in submodules (mlcookbook.preprocessing,
mlcookbook.evaluation, ...) and is imported explicitly.
"""

from mlcookbook.classification.logistic import LogisticRegressionResult, logistic_regression
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.schema import VarType
from mlcookbook.regression.linear import LinearRegressionResult, linear_regression
from mlcookbook.statistics.association import CorrelationResult, correlation

__version__ = "0.1.0"

__all__ = [
    "CorrelationResult",
    "Dataset",
    "LinearRegressionResult",
    "LogisticRegressionResult",
    "VarType",
    "correlation",
    "linear_regression",
    "logistic_regression",
]
