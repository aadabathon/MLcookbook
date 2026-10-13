"""Classification workflows (binary or multiclass): logistic regression, decision trees,
random forests, support vector machines, and neural networks."""

from mlcookbook.classification.forest import RandomForestResult, random_forest
from mlcookbook.classification.logistic import LogisticRegressionResult, logistic_regression
from mlcookbook.classification.neural_network import NetworkTraining, NeuralNetworkResult, neural_network
from mlcookbook.classification.svm import SVMResult, svm
from mlcookbook.classification.tree import DecisionTreeResult, decision_tree

__all__ = [
    "DecisionTreeResult", "LogisticRegressionResult", "NetworkTraining", "NeuralNetworkResult",
    "RandomForestResult", "SVMResult",
    "decision_tree", "logistic_regression", "neural_network", "random_forest", "svm",
]
