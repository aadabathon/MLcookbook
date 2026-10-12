"""Binary logistic regression workflow.

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

from mlcookbook.classification._binary import prepare_binary_problem
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
    classes: list[Any]  # [negative_class, positive_class]
    positive_class: Any
    n_train: int
    n_test: int
    intercept: float  # log-odds of the positive class at the feature means
    coefficients: dict[str, float]  # log-odds, keyed by post-preprocessing feature name
    odds_ratios: dict[str, float]  # exp(coefficient)
    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    test_actual: list[Any]  # held-out class labels, for plotting
    test_probability: list[float]  # predicted P(positive_class) for each held-out row
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
    """Fit a binary logistic regression of *target* on *features*.

    The target must have exactly two classes (after dropping missing values).
    *positive_class* picks which one is modeled as "1"; by default it is the
    larger of the two in sorted order (e.g. True over False). The train/test
    split is stratified so both splits contain both classes.

    If *features* is None, all numeric/categorical/boolean/datetime columns
    except the target are used; other columns are skipped with a warning. Rows
    with a missing target are dropped (feature missingness is imputed in-pipeline).

    *preprocessing* configures imputation/scaling/encoding; *cv_folds* runs
    stratified k-fold cross-validation on the training rows (0 disables it).

    Note: coefficients are log-odds; with the default standard scaling, numeric
    features' coefficients are per standard deviation.
    """
    features, frame, labels, warnings = prepare_binary_problem(
        dataset, target, features, positive_class, "logistic_regression"
    )
    positive_class = labels[1]
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
        make_model(), X_train, y_train, "binary", cv_folds, random_state, warnings, positive_class
    )

    classifier = model.named_steps["classify"]
    # sklearn orders classes_ ascending; the positive class's probability is its column.
    positive_col = list(classifier.classes_).index(positive_class)
    # coef_ describes classes_[1]; flip the sign if the user chose the other as positive.
    sign = 1.0 if positive_col == 1 else -1.0

    def score(X_part) -> np.ndarray:
        return model.predict_proba(X_part)[:, positive_col]

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = sign * classifier.coef_[0]
    test_probability = score(X_test)
    return LogisticRegressionResult(
        method="logistic_regression",
        target=target,
        features=list(features),
        classes=labels,
        positive_class=positive_class,
        n_train=len(X_train),
        n_test=len(X_test),
        intercept=float(sign * classifier.intercept_[0]),
        coefficients={str(name): float(c) for name, c in zip(feature_names, coefs)},
        odds_ratios={str(name): float(np.exp(c)) for name, c in zip(feature_names, coefs)},
        train_metrics=classification_metrics(y_train, model.predict(X_train), score(X_train), labels),
        test_metrics=classification_metrics(y_test, model.predict(X_test), test_probability, labels),
        test_actual=y_test.tolist(),
        test_probability=[float(p) for p in test_probability],
        preprocessing=describe_preprocessor(model.named_steps["preprocess"], X_train, dataset, config),
        cross_validation=cross_validation,
        warnings=warnings,
    )
