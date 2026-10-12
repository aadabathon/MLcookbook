"""Logistic regression workflow (binary or multinomial).

Mirrors ``regression/linear.py``: validate against the schema, build a
leakage-safe Pipeline, hold out a (stratified) test split, evaluate both splits,
and return a frozen result dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlcookbook.classification._problem import (
    class_probabilities,
    metric_scores,
    prepare_classification_problem,
    task_for,
)
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType
from mlcookbook.evaluation.cross_validation import CrossValidationResult, training_cross_validation
from mlcookbook.evaluation.metrics import ClassificationMetrics, classification_metrics
from mlcookbook.preprocessing.builders import build_preprocessor, resolve_config
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.summary import PreprocessingSummary, describe_preprocessor
from mlcookbook.regression.linear import ORDINAL_LINEAR_WARNING


@dataclass(frozen=True)
class LogisticRegressionResult(Result):
    method: str
    target: str
    features: list[str]
    classes: list[Any]  # binary: [negative_class, positive_class]; multiclass: sorted classes
    positive_class: Any  # None for multiclass
    n_train: int
    n_test: int
    intercept: float | None  # binary: log-odds of the positive class at the feature means
    coefficients: dict[str, float]  # binary: log-odds per post-preprocessing feature; {} if multiclass
    odds_ratios: dict[str, float]  # exp(coefficient); {} if multiclass
    class_intercepts: dict[Any, float]  # multiclass softmax intercepts; {} if binary
    class_coefficients: dict[Any, dict[str, float]]  # multiclass softmax coefficients; {} if binary
    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    test_actual: list[Any]  # held-out class labels, for plotting
    test_predicted: list[Any]
    test_probability: list[float]  # binary: P(positive_class) per held-out row; [] if multiclass
    test_class_probabilities: list[list[float]]  # per held-out row, one probability per class
    preprocessing: PreprocessingSummary
    cross_validation: CrossValidationResult | None  # on the training rows only
    warnings: list[str] = field(default_factory=list)


def logistic_regression(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    positive_class: Any = None,
    test_size: float = 0.25,
    random_state: int = 0,
    preprocessing: PreprocessingConfig | None = None,
    cv_folds: int = 5,
) -> LogisticRegressionResult:
    """Fit a logistic regression of *target* on *features*.

    With two classes, *positive_class* picks which one is modeled as "1"; by
    default it is the larger of the two in sorted order (e.g. True over False).
    With three or more classes the model is multinomial (softmax): each class
    gets its own coefficients, and the difference between two classes'
    coefficients is the change in their log-odds per unit of the feature.
    The train/test split is stratified so both splits contain every class.

    If *features* is None, all numeric/categorical/boolean/datetime columns
    except the target are used; other columns are skipped with a warning. Rows
    with a missing target are dropped (feature missingness is imputed in-pipeline).

    *preprocessing* configures imputation/scaling/encoding; *cv_folds* runs
    stratified k-fold cross-validation on the training rows (0 disables it).

    Note: coefficients are log-odds; with the default standard scaling, numeric
    features' coefficients are per standard deviation.
    """
    features, frame, labels, warnings = prepare_classification_problem(
        dataset, target, features, positive_class, "logistic_regression"
    )
    binary = len(labels) == 2
    positive_class = labels[1] if binary else None
    y = frame[target]
    config = resolve_config(preprocessing, scale=True)
    schema = dataset.schema()
    if config.categorical_encoding == "ordinal" and any(
        schema.var_type(f) in (VarType.CATEGORICAL, VarType.BOOLEAN) for f in features
    ):
        warnings.append(ORDINAL_LINEAR_WARNING)

    X = frame[features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    def make_model() -> Pipeline:
        return Pipeline(
            [
                ("preprocess", build_preprocessor(dataset, features, config=config)),
                ("classify", LogisticRegression(max_iter=1000)),
            ]
        )

    model = make_model()
    model.fit(X_train, y_train)
    cross_validation = training_cross_validation(
        make_model(), X_train, y_train, task_for(labels), cv_folds, random_state, warnings, positive_class
    )

    classifier = model.named_steps["classify"]
    feature_names = [str(n) for n in model.named_steps["preprocess"].get_feature_names_out()]
    if binary:
        # coef_ describes classes_[1]; flip the sign if the user chose the other as positive.
        sign = 1.0 if list(classifier.classes_).index(positive_class) == 1 else -1.0
        coefs = sign * classifier.coef_[0]
        intercept = float(sign * classifier.intercept_[0])
        coefficients = {name: float(c) for name, c in zip(feature_names, coefs)}
        odds_ratios = {name: float(np.exp(c)) for name, c in zip(feature_names, coefs)}
        class_intercepts, class_coefficients = {}, {}
    else:
        intercept, coefficients, odds_ratios = None, {}, {}
        rows = {label: i for i, label in enumerate(classifier.classes_)}
        class_intercepts = {label: float(classifier.intercept_[rows[label]]) for label in labels}
        class_coefficients = {
            label: {name: float(c) for name, c in zip(feature_names, classifier.coef_[rows[label]])}
            for label in labels
        }

    train_probabilities = class_probabilities(model, X_train, labels)
    test_probabilities = class_probabilities(model, X_test, labels)
    test_predicted = model.predict(X_test)
    return LogisticRegressionResult(
        method="logistic_regression",
        target=target,
        features=list(features),
        classes=labels,
        positive_class=positive_class,
        n_train=len(X_train),
        n_test=len(X_test),
        intercept=intercept,
        coefficients=coefficients,
        odds_ratios=odds_ratios,
        class_intercepts=class_intercepts,
        class_coefficients=class_coefficients,
        train_metrics=classification_metrics(
            y_train, model.predict(X_train), metric_scores(train_probabilities, labels), labels
        ),
        test_metrics=classification_metrics(y_test, test_predicted, metric_scores(test_probabilities, labels), labels),
        test_actual=y_test.tolist(),
        test_predicted=test_predicted.tolist(),
        test_probability=[float(p) for p in test_probabilities[:, 1]] if binary else [],
        test_class_probabilities=test_probabilities.tolist(),
        preprocessing=describe_preprocessor(model.named_steps["preprocess"], X_train, dataset, config),
        cross_validation=cross_validation,
        warnings=warnings,
    )
