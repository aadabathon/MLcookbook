"""ML Cookbook: statistics, classical ML, and optional local LLM generation.

Public API:

    from mlcookbook import Dataset, VarType, PreprocessingConfig, load_dataset

    # statistics
    correlation, chi_square, anova, association_scan
    # supervised learning (regression)
    linear_regression, decision_tree_regression, random_forest_regression,
    svm_regression, neural_network_regression
    # supervised learning (classification: binary or multiclass)
    logistic_regression, decision_tree, random_forest, svm, neural_network
    # unsupervised learning
    kmeans, pca

    # shared-fold regression comparison
    compare_regressors

    # optional local PyTorch / Transformers inference
    generate_text

Everything else lives in submodules (mlcookbook.preprocessing,
mlcookbook.evaluation, ...) and is imported explicitly.

Public symbols load on first access, so launching the desktop shell does not
first import every analysis and its scientific dependencies.
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from mlcookbook.classification.forest import RandomForestResult, random_forest
    from mlcookbook.classification.logistic import LogisticRegressionResult, logistic_regression
    from mlcookbook.classification.neural_network import NeuralNetworkResult, neural_network
    from mlcookbook.classification.svm import SVMResult, svm
    from mlcookbook.classification.tree import DecisionTreeResult, decision_tree
    from mlcookbook.clustering.kmeans import KMeansResult, kmeans
    from mlcookbook.comparison import RegressionComparisonResult, compare_regressors
    from mlcookbook.core.dataset import Dataset
    from mlcookbook.core.schema import VarType
    from mlcookbook.dimensionality.pca import PCAResult, pca
    from mlcookbook.io import load_dataset
    from mlcookbook.llm import TextGenerationResult, generate_text
    from mlcookbook.preprocessing.config import PreprocessingConfig
    from mlcookbook.regression.forest import RandomForestRegressionResult, random_forest_regression
    from mlcookbook.regression.linear import LinearRegressionResult, linear_regression
    from mlcookbook.regression.neural_network import NeuralNetworkRegressionResult, neural_network_regression
    from mlcookbook.regression.svm import SVMRegressionResult, svm_regression
    from mlcookbook.regression.tree import RegressionTreeResult, decision_tree_regression
    from mlcookbook.statistics.association import CorrelationResult, correlation
    from mlcookbook.statistics.categorical import ChiSquareResult, chi_square
    from mlcookbook.statistics.groups import AnovaResult, anova
    from mlcookbook.statistics.scan import AssociationScanResult, association_scan
    from mlcookbook.wearable import (
        WearableEvaluationResult,
        WearableImportReport,
        evaluate_wearable,
        load_wearable_study,
    )

__version__ = "1.0.0"

__all__ = [
    "WearableEvaluationResult", "WearableImportReport", "evaluate_wearable", "load_wearable_study",
    "AnovaResult",
    "AssociationScanResult",
    "ChiSquareResult",
    "CorrelationResult",
    "Dataset",
    "DecisionTreeResult",
    "KMeansResult",
    "LinearRegressionResult",
    "LogisticRegressionResult",
    "NeuralNetworkRegressionResult",
    "NeuralNetworkResult",
    "PCAResult",
    "PreprocessingConfig",
    "RandomForestRegressionResult",
    "RandomForestResult",
    "RegressionTreeResult",
    "RegressionComparisonResult",
    "SVMRegressionResult",
    "SVMResult",
    "VarType",
    "TextGenerationResult",
    "anova",
    "association_scan",
    "chi_square",
    "correlation",
    "generate_text",
    "compare_regressors",
    "decision_tree",
    "decision_tree_regression",
    "kmeans",
    "linear_regression",
    "load_dataset",
    "logistic_regression",
    "neural_network",
    "neural_network_regression",
    "pca",
    "random_forest",
    "random_forest_regression",
    "svm",
    "svm_regression",
]

_EXPORTS = {
    "Dataset": "mlcookbook.core.dataset",
    "VarType": "mlcookbook.core.schema",
    "PreprocessingConfig": "mlcookbook.preprocessing.config",
    "load_dataset": "mlcookbook.io",
    "CorrelationResult": "mlcookbook.statistics.association",
    "correlation": "mlcookbook.statistics.association",
    "ChiSquareResult": "mlcookbook.statistics.categorical",
    "chi_square": "mlcookbook.statistics.categorical",
    "AnovaResult": "mlcookbook.statistics.groups",
    "anova": "mlcookbook.statistics.groups",
    "AssociationScanResult": "mlcookbook.statistics.scan",
    "association_scan": "mlcookbook.statistics.scan",
    "LinearRegressionResult": "mlcookbook.regression.linear",
    "linear_regression": "mlcookbook.regression.linear",
    "RegressionTreeResult": "mlcookbook.regression.tree",
    "decision_tree_regression": "mlcookbook.regression.tree",
    "RandomForestRegressionResult": "mlcookbook.regression.forest",
    "random_forest_regression": "mlcookbook.regression.forest",
    "SVMRegressionResult": "mlcookbook.regression.svm",
    "svm_regression": "mlcookbook.regression.svm",
    "NeuralNetworkRegressionResult": "mlcookbook.regression.neural_network",
    "neural_network_regression": "mlcookbook.regression.neural_network",
    "LogisticRegressionResult": "mlcookbook.classification.logistic",
    "logistic_regression": "mlcookbook.classification.logistic",
    "DecisionTreeResult": "mlcookbook.classification.tree",
    "decision_tree": "mlcookbook.classification.tree",
    "RandomForestResult": "mlcookbook.classification.forest",
    "random_forest": "mlcookbook.classification.forest",
    "SVMResult": "mlcookbook.classification.svm",
    "svm": "mlcookbook.classification.svm",
    "NeuralNetworkResult": "mlcookbook.classification.neural_network",
    "neural_network": "mlcookbook.classification.neural_network",
    "KMeansResult": "mlcookbook.clustering.kmeans",
    "kmeans": "mlcookbook.clustering.kmeans",
    "PCAResult": "mlcookbook.dimensionality.pca",
    "pca": "mlcookbook.dimensionality.pca",
    "RegressionComparisonResult": "mlcookbook.comparison",
    "compare_regressors": "mlcookbook.comparison",
    "TextGenerationResult": "mlcookbook.llm",
    "generate_text": "mlcookbook.llm",
    "WearableEvaluationResult": "mlcookbook.wearable",
    "WearableImportReport": "mlcookbook.wearable",
    "evaluate_wearable": "mlcookbook.wearable",
    "load_wearable_study": "mlcookbook.wearable",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
