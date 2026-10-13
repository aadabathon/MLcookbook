"""Random forest classification (binary or multiclass) with training-only preprocessing."""
from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass, field
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
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
from mlcookbook.evaluation.cross_validation import CrossValidationResult, training_cross_validation
from mlcookbook.evaluation.metrics import ClassificationMetrics, classification_metrics
from mlcookbook.preprocessing.builders import build_preprocessor, resolve_config
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.summary import PreprocessingSummary, describe_preprocessor

OVERFIT_GAP = 0.10


@dataclass(frozen=True)
class RandomForestResult(Result):
    method: str
    target: str
    features: list[str]
    classes: list[Any]  # binary: [negative, positive]; multiclass: sorted classes
    positive_class: Any  # None for multiclass
    n_train: int
    n_test: int
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    max_features: str | int | float | None
    feature_importances: dict[str, float]  # post-preprocessing names; impurity decrease
    tree_depths: list[int]
    tree_leaf_counts: list[int]
    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    test_actual: list[Any]
    test_predicted: list[Any]
    test_probability: list[float]  # binary: probability of positive_class; [] if multiclass
    test_class_probabilities: list[list[float]]  # per held-out row, one probability per class
    roc_fpr: list[float]  # binary test-set ROC; empty if multiclass or only one class occurs
    roc_tpr: list[float]
    preprocessing: PreprocessingSummary
    cross_validation: CrossValidationResult | None  # on the training rows only
    warnings: list[str] = field(default_factory=list)


def random_forest(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    positive_class: Any = None,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 5,
    max_features: str | int | float | None = "sqrt",
    test_size: float = 0.25,
    random_state: int = 0,
    preprocessing: PreprocessingConfig | None = None,
    cv_folds: int = 5,
) -> RandomForestResult:
    """Fit a random forest classifier with held-out probabilities and feature importances.

    Each tree uses bootstrap samples and a random subset of transformed features
    at each split. sklearn averages the trees' class probabilities. Numeric
    features are imputed without scaling; categorical/boolean features are
    imputed and one-hot encoded. All preprocessing is fitted on training rows.

    Target validation (binary or multiclass) and the stratified split follow
    the other classifiers.
    The result carries test ROC coordinates and per-tree sizes, never estimators.
    Impurity importances are per encoded feature, may favor high-cardinality
    features, and are not causal effects. Model parameters are validated by sklearn.
    *preprocessing* configures imputation/encoding (scaling defaults to none);
    *cv_folds* runs stratified k-fold cross-validation on the training rows.
    """
    features, frame, labels, warnings = prepare_classification_problem(
        dataset, target, features, positive_class, "random_forest"
    )
    binary = len(labels) == 2
    positive_class = labels[1] if binary else None
    X_train, X_test, y_train, y_test = train_test_split(
        frame[features], frame[target], test_size=test_size,
        random_state=random_state, stratify=frame[target],
    )
    config = resolve_config(preprocessing, scale=False)
    if not X_train.notna().any().any():
        raise ValueError("No observed feature values in the training rows.")
    missing = int(X_train.isna().sum().sum() + X_test.isna().sum().sum())
    if missing:
        warnings.append(f"Found {missing} missing feature value(s); imputation is fitted on training rows only.")

    def make_model() -> Pipeline:
        return Pipeline([
            ("preprocess", build_preprocessor(dataset, features, config=config)),
            ("forest", RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_leaf=min_samples_leaf, max_features=max_features,
                bootstrap=True, random_state=random_state, n_jobs=1,
            )),
        ])

    model = make_model()
    # sklearn may warn about entirely missing columns. Surface those warnings in
    # the result too, rather than writing them to the desktop app's console.
    with python_warnings.catch_warnings(record=True) as caught:
        python_warnings.simplefilter("always", UserWarning)
        model.fit(X_train, y_train)
        classifier = model.named_steps["forest"]
        names = model.named_steps["preprocess"].get_feature_names_out()
        train_predicted = model.predict(X_train)
        test_predicted = model.predict(X_test)
        train_probabilities = class_probabilities(model, X_train, labels)
        test_probabilities = class_probabilities(model, X_test, labels)
        cross_validation = training_cross_validation(
            make_model(), X_train, y_train, task_for(labels), cv_folds, random_state, warnings, positive_class
        )
    warnings.extend(dict.fromkeys(str(w.message) for w in caught))

    train_metrics = classification_metrics(y_train, train_predicted, metric_scores(train_probabilities, labels), labels)
    test_metrics = classification_metrics(y_test, test_predicted, metric_scores(test_probabilities, labels), labels)
    test_probability = test_probabilities[:, 1].tolist() if binary else []
    if train_metrics.accuracy - test_metrics.accuracy > OVERFIT_GAP:
        warnings.append(
            f"Possible overfitting: train accuracy {train_metrics.accuracy:.3f} vs "
            f"test {test_metrics.accuracy:.3f}. Consider limiting max_depth or increasing min_samples_leaf."
        )
    warnings.append(
        "Impurity-based feature importances can favor high-cardinality features; they are not causal effects."
    )
    if not binary:
        fpr, tpr = [], []
    elif y_test.nunique() < 2:
        fpr, tpr = [], []
        warnings.append("ROC curve and ROC AUC are undefined: the test split contains only one class.")
    else:
        fpr, tpr, _ = roc_curve(y_test, test_probability, pos_label=positive_class)

    return RandomForestResult(
        method="random_forest", target=target, features=list(features),
        classes=labels, positive_class=positive_class,
        n_train=len(X_train), n_test=len(X_test),
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features=max_features,
        feature_importances={str(name): float(value) for name, value in zip(names, classifier.feature_importances_)},
        tree_depths=[int(tree.get_depth()) for tree in classifier.estimators_],
        tree_leaf_counts=[int(tree.get_n_leaves()) for tree in classifier.estimators_],
        train_metrics=train_metrics, test_metrics=test_metrics,
        test_actual=y_test.tolist(), test_predicted=test_predicted.tolist(),
        test_probability=test_probability,
        test_class_probabilities=test_probabilities.tolist(),
        roc_fpr=list(map(float, fpr)), roc_tpr=list(map(float, tpr)),
        preprocessing=describe_preprocessor(model.named_steps["preprocess"], X_train, dataset, config),
        cross_validation=cross_validation,
        warnings=warnings,
    )
