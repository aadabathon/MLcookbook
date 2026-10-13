"""Neural network classification: a multilayer perceptron (scikit-learn's MLP).

Each hidden layer is a matrix multiply plus bias followed by a nonlinearity
(ReLU by default); the output layer is a sigmoid (binary) or softmax
(multiclass). Weights are trained by Adam on mini-batches to minimize
cross-entropy, with an L2 penalty *alpha* on the weights.

With ``early_stopping`` (the default), 10% of the training rows are held out
inside the fit and training stops once their score has not improved for 30
epochs; the result carries both the training-loss curve and that validation
curve, which is where overfitting shows up. Inputs are standard-scaled by
default: gradient descent on unscaled features is slow and lopsided.
"""

from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from mlcookbook.classification._problem import (
    class_probabilities,
    metric_scores,
    prepare_classification_problem,
    task_for,
)
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.evaluation.cross_validation import CrossValidationResult, training_cross_validation
from mlcookbook.evaluation.metrics import ClassificationMetrics, classification_metrics
from mlcookbook.preprocessing.builders import build_preprocessor, resolve_config
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.summary import PreprocessingSummary, describe_preprocessor

ACTIVATIONS = ("relu", "tanh", "logistic")
BATCH_SIZE = 32  # small batches: several gradient steps per epoch even on small tables
PATIENCE = 30  # early stopping: epochs without validation improvement before stopping
OVERFIT_GAP = 0.10


@dataclass(frozen=True)
class NetworkTraining(Result):
    """How the fit went, shared by the classifier and the regressor."""

    architecture: list[int]  # units per layer: inputs, hidden layers..., outputs
    n_parameters: int  # weights + biases
    activation: str
    alpha: float  # L2 penalty
    learning_rate: float
    epochs: int
    converged: bool
    early_stopping: bool
    loss_curve: list[float]  # training loss per epoch
    validation_curve: list[float]  # per epoch, on the internal validation rows (empty without early stopping)
    validation_metric: str  # "accuracy" or "r2"


@dataclass(frozen=True)
class NeuralNetworkResult(Result):
    method: str
    target: str
    features: list[str]
    classes: list[Any]  # binary: [negative, positive]; multiclass: sorted classes
    positive_class: Any  # None for multiclass
    n_train: int
    n_test: int
    training: NetworkTraining
    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    test_actual: list[Any]
    test_predicted: list[Any]
    test_probability: list[float]  # binary: P(positive_class); [] if multiclass
    test_class_probabilities: list[list[float]]
    preprocessing: PreprocessingSummary
    cross_validation: CrossValidationResult | None  # on the training rows only
    warnings: list[str] = field(default_factory=list)


def validate_network_settings(hidden_layers, activation: str, alpha: float, learning_rate: float, max_epochs: int):
    hidden_layers = tuple(hidden_layers)
    if not hidden_layers or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in hidden_layers):
        raise ValueError(f"hidden_layers must be positive layer sizes, e.g. (64, 32); got {hidden_layers!r}.")
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {list(ACTIVATIONS)}; got {activation!r}.")
    if not alpha >= 0:
        raise ValueError(f"alpha must be non-negative; got {alpha}.")
    if not learning_rate > 0:
        raise ValueError(f"learning_rate must be positive; got {learning_rate}.")
    if max_epochs < 1:
        raise ValueError(f"max_epochs must be at least 1; got {max_epochs}.")
    return hidden_layers


def fit_quietly(fit) -> bool:
    """Run *fit*; return False if sklearn warned that the optimizer did not converge."""
    with python_warnings.catch_warnings(record=True) as caught:
        python_warnings.simplefilter("always", ConvergenceWarning)
        fit()
    return not any(issubclass(w.category, ConvergenceWarning) for w in caught)


def describe_training(network, n_inputs: int, converged: bool, early_stopping: bool,
                      validation_metric: str, alpha: float, learning_rate: float, warnings: list[str]) -> NetworkTraining:
    n_parameters = int(sum(w.size for w in network.coefs_) + sum(b.size for b in network.intercepts_))
    if not converged:
        warnings.append(
            f"Training stopped at the epoch limit ({network.n_iter_}) before the loss settled; "
            "raise max_epochs or the learning rate."
        )
    return NetworkTraining(
        architecture=[n_inputs, *network.hidden_layer_sizes, network.n_outputs_],
        n_parameters=n_parameters,
        activation=network.activation,
        alpha=float(alpha),
        learning_rate=float(learning_rate),
        epochs=int(network.n_iter_),
        converged=converged,
        early_stopping=early_stopping,
        loss_curve=[float(v) for v in network.loss_curve_],
        validation_curve=[float(v) for v in (network.validation_scores_ or [])] if early_stopping else [],
        validation_metric=validation_metric,
    )


def capacity_warning(n_parameters: int, n_train: int) -> list[str]:
    if n_parameters > n_train:
        return [
            f"The network has {n_parameters} parameters for {n_train} training rows; it can memorize them. "
            "Rely on alpha and early stopping, and watch the train/test gap."
        ]
    return []


def neural_network(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    positive_class: Any = None,
    hidden_layers: tuple[int, ...] = (64, 32),
    activation: str = "relu",
    alpha: float = 1e-4,
    learning_rate: float = 1e-3,
    max_epochs: int = 500,
    early_stopping: bool = True,
    test_size: float = 0.25,
    random_state: int = 0,
    preprocessing: PreprocessingConfig | None = None,
    cv_folds: int = 5,
) -> NeuralNetworkResult:
    """Fit a multilayer perceptron classifier of *target* on *features*.

    Target and feature rules match the other classifiers. *hidden_layers*
    gives the units per hidden layer, *alpha* the L2 penalty, *learning_rate*
    Adam's initial step size, and *max_epochs* the limit on passes over the data.
    """
    hidden_layers = validate_network_settings(hidden_layers, activation, alpha, learning_rate, max_epochs)
    features, frame, labels, warnings = prepare_classification_problem(
        dataset, target, features, positive_class, "neural_network"
    )
    binary = len(labels) == 2
    positive_class = labels[1] if binary else None
    config = resolve_config(preprocessing, scale=True)
    if config.scaling == "none":
        warnings.append("Gradient descent on unscaled features is slow and uneven; use a scaling option.")
    X_train, X_test, y_train, y_test = train_test_split(
        frame[features], frame[target], test_size=test_size, random_state=random_state, stratify=frame[target]
    )
    # The network trains on integer class codes (position in sorted order): sklearn's MLP early
    # stopping cannot score string labels. Sorted order keeps the fit independent of which
    # class is called positive. Predictions are decoded back before evaluation.
    ordered = sorted(labels)
    code_of = {label: i for i, label in enumerate(ordered)}
    codes = [code_of[label] for label in labels]  # probability columns in `labels` order
    code_train = y_train.map(code_of).astype(int)

    def make_model() -> Pipeline:
        return Pipeline([
            ("preprocess", build_preprocessor(dataset, features, config=config)),
            ("network", MLPClassifier(
                hidden_layer_sizes=hidden_layers, activation=activation, alpha=alpha,
                learning_rate_init=learning_rate, max_iter=max_epochs, early_stopping=early_stopping,
                batch_size=BATCH_SIZE, n_iter_no_change=PATIENCE, random_state=random_state,
            )),
        ])

    model = make_model()
    converged = fit_quietly(lambda: model.fit(X_train, code_train))
    with python_warnings.catch_warnings():
        python_warnings.simplefilter("ignore", ConvergenceWarning)  # reported once, for the main fit
        cross_validation = training_cross_validation(
            make_model(), X_train, code_train, task_for(labels), cv_folds, random_state, warnings,
            code_of[positive_class] if binary else None,
        )

    n_inputs = len(model.named_steps["preprocess"].get_feature_names_out())
    training = describe_training(model.named_steps["network"], n_inputs, converged, early_stopping,
                                 "accuracy", alpha, learning_rate, warnings)
    warnings += capacity_warning(training.n_parameters, len(X_train))
    train_probabilities = class_probabilities(model, X_train, codes)
    test_probabilities = class_probabilities(model, X_test, codes)
    test_predicted = pd.Series([ordered[i] for i in model.predict(X_test)], dtype=object)
    train_predicted = [ordered[i] for i in model.predict(X_train)]
    train_metrics = classification_metrics(y_train, train_predicted, metric_scores(train_probabilities, labels), labels)
    test_metrics = classification_metrics(y_test, test_predicted, metric_scores(test_probabilities, labels), labels)
    if train_metrics.accuracy - test_metrics.accuracy > OVERFIT_GAP:
        warnings.append(
            f"Possible overfitting: train accuracy {train_metrics.accuracy:.3f} vs test "
            f"{test_metrics.accuracy:.3f}. Try a larger alpha, a smaller network, or early stopping."
        )
    return NeuralNetworkResult(
        method="neural_network", target=target, features=list(features), classes=labels,
        positive_class=positive_class, n_train=len(X_train), n_test=len(X_test),
        training=training,
        train_metrics=train_metrics, test_metrics=test_metrics,
        test_actual=y_test.tolist(), test_predicted=test_predicted.tolist(),
        test_probability=test_probabilities[:, 1].tolist() if binary else [],
        test_class_probabilities=test_probabilities.tolist(),
        preprocessing=describe_preprocessor(model.named_steps["preprocess"], X_train, dataset, config),
        cross_validation=cross_validation,
        warnings=warnings,
    )
