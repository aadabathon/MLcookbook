"""Neural network regression: a multilayer perceptron (scikit-learn's MLP).

The regression twin of ``classification/neural_network.py``: the same hidden
layers, but a linear output unit trained on squared error. The target is
standardized inside the pipeline (training rows only), because gradient
descent and the weight penalty both assume outputs of roughly unit scale.
"""

from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass, field

from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlcookbook.classification.neural_network import (
    BATCH_SIZE,
    PATIENCE,
    NetworkTraining,
    capacity_warning,
    describe_training,
    fit_quietly,
    validate_network_settings,
)
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.evaluation.cross_validation import CrossValidationResult, training_cross_validation
from mlcookbook.evaluation.metrics import RegressionMetrics, regression_metrics
from mlcookbook.preprocessing.builders import build_preprocessor, resolve_config
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.summary import PreprocessingSummary, describe_preprocessor
from mlcookbook.regression._problem import overfit_warning, prepare_regression_problem


@dataclass(frozen=True)
class NeuralNetworkRegressionResult(Result):
    method: str
    target: str
    features: list[str]
    n_train: int
    n_test: int
    training: NetworkTraining  # the loss curve is in standardized-target units
    train_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    test_actual: list[float]
    test_predicted: list[float]
    preprocessing: PreprocessingSummary
    cross_validation: CrossValidationResult | None  # on the training rows only
    warnings: list[str] = field(default_factory=list)


def neural_network_regression(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
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
) -> NeuralNetworkRegressionResult:
    """Fit a multilayer perceptron regressor of a numeric *target* on *features*.

    Settings match ``neural_network``; with early stopping the validation
    curve is R² on the internal validation rows.
    """
    hidden_layers = validate_network_settings(hidden_layers, activation, alpha, learning_rate, max_epochs)
    features, frame, warnings = prepare_regression_problem(dataset, target, features, "neural_network_regression")
    config = resolve_config(preprocessing, scale=True)
    if config.scaling == "none":
        warnings.append("Gradient descent on unscaled features is slow and uneven; use a scaling option.")
    X_train, X_test, y_train, y_test = train_test_split(
        frame[features], frame[target], test_size=test_size, random_state=random_state
    )

    def make_model() -> Pipeline:
        network = MLPRegressor(
            hidden_layer_sizes=hidden_layers, activation=activation, alpha=alpha,
            learning_rate_init=learning_rate, max_iter=max_epochs, early_stopping=early_stopping,
            batch_size=BATCH_SIZE, n_iter_no_change=PATIENCE, random_state=random_state,
        )
        return Pipeline([
            ("preprocess", build_preprocessor(dataset, features, config=config)),
            ("network", TransformedTargetRegressor(regressor=network, transformer=StandardScaler())),
        ])

    model = make_model()
    converged = fit_quietly(lambda: model.fit(X_train, y_train))
    with python_warnings.catch_warnings():
        python_warnings.simplefilter("ignore", ConvergenceWarning)  # reported once, for the main fit
        cross_validation = training_cross_validation(
            make_model(), X_train, y_train, "regression", cv_folds, random_state, warnings
        )

    n_inputs = len(model.named_steps["preprocess"].get_feature_names_out())
    training = describe_training(model.named_steps["network"].regressor_, n_inputs, converged, early_stopping,
                                 "r2", alpha, learning_rate, warnings)
    warnings += capacity_warning(training.n_parameters, len(X_train))
    train_metrics = regression_metrics(y_train, model.predict(X_train))
    test_predicted = model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_predicted)
    warnings += overfit_warning(
        train_metrics.r2, test_metrics.r2, "Try a larger alpha, a smaller network, or early stopping."
    )
    return NeuralNetworkRegressionResult(
        method="neural_network_regression", target=target, features=list(features),
        n_train=len(X_train), n_test=len(X_test), training=training,
        train_metrics=train_metrics, test_metrics=test_metrics,
        test_actual=[float(v) for v in y_test], test_predicted=[float(v) for v in test_predicted],
        preprocessing=describe_preprocessor(model.named_steps["preprocess"], X_train, dataset, config),
        cross_validation=cross_validation,
        warnings=warnings,
    )
