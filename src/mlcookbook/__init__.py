"""ML Cookbook: a statistics and classical-ML workbench for tabular data.

Public API:

    from mlcookbook import Dataset, VarType, PreprocessingConfig, load_dataset

    # statistics
    correlation, chi_square, anova, association_scan
    # supervised learning (regression)
    linear_regression, decision_tree_regression, random_forest_regression
    # supervised learning (classification: binary or multiclass)
    logistic_regression, decision_tree, random_forest
    # unsupervised learning
    kmeans, pca

Everything else lives in submodules (mlcookbook.preprocessing,
mlcookbook.evaluation, ...) and is imported explicitly.
"""

from mlcookbook.classification.forest import RandomForestResult, random_forest
from mlcookbook.classification.logistic import LogisticRegressionResult, logistic_regression
from mlcookbook.classification.tree import DecisionTreeResult, decision_tree
from mlcookbook.clustering.kmeans import KMeansResult, kmeans
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.schema import VarType
from mlcookbook.dimensionality.pca import PCAResult, pca
from mlcookbook.io import load_dataset
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.regression.forest import RandomForestRegressionResult, random_forest_regression
from mlcookbook.regression.linear import LinearRegressionResult, linear_regression
from mlcookbook.regression.tree import RegressionTreeResult, decision_tree_regression
from mlcookbook.statistics.association import CorrelationResult, correlation
from mlcookbook.statistics.categorical import ChiSquareResult, chi_square
from mlcookbook.statistics.groups import AnovaResult, anova
from mlcookbook.statistics.scan import AssociationScanResult, association_scan

__version__ = "0.2.0"

__all__ = [
    "AnovaResult",
    "AssociationScanResult",
    "ChiSquareResult",
    "CorrelationResult",
    "Dataset",
    "DecisionTreeResult",
    "KMeansResult",
    "LinearRegressionResult",
    "LogisticRegressionResult",
    "PCAResult",
    "PreprocessingConfig",
    "RandomForestRegressionResult",
    "RandomForestResult",
    "RegressionTreeResult",
    "VarType",
    "anova",
    "association_scan",
    "chi_square",
    "correlation",
    "decision_tree",
    "decision_tree_regression",
    "kmeans",
    "linear_regression",
    "load_dataset",
    "logistic_regression",
    "pca",
    "random_forest",
    "random_forest_regression",
]
