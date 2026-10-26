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
"""

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
