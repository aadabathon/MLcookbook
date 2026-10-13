"""Support vector machine classification (binary or multiclass).

An SVM finds the boundary with the widest margin between classes; only the
training rows on or inside the margin (the *support vectors*) define it. The
kernel decides the boundary's shape: ``"linear"`` is a flat hyperplane,
``"rbf"`` and ``"poly"`` implicitly map rows into a richer space where a flat
boundary becomes a curved one. *C* trades margin width against training
errors: small C = wide, tolerant margin; large C = narrow margin that chases
every training row (overfitting risk).

Distances drive everything, so features are standard-scaled by default.
Probabilities come from Platt scaling (an internal cross-validated logistic
fit on the SVM scores), which sklearn computes when ``probability=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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

KERNELS = ("rbf", "linear", "poly")
OVERFIT_GAP = 0.10
LARGE_DATA_ROWS = 10_000  # kernel SVM training time grows roughly with the square of the rows


@dataclass(frozen=True)
class SVMResult(Result):
    method: str
    target: str
    features: list[str]
    classes: list[Any]  # binary: [negative, positive]; multiclass: sorted classes
    positive_class: Any  # None for multiclass
    n_train: int
    n_test: int
    kernel: str
    C: float
    support_vectors: dict[Any, int]  # per class: training rows that define the boundary
    support_vector_share: float  # share of training rows that are support vectors
    coefficients: dict[str, float]  # linear kernel, binary only: boundary weights (toward positive_class)
    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    test_actual: list[Any]
    test_predicted: list[Any]
    test_probability: list[float]  # binary: Platt-scaled P(positive_class); [] if multiclass
    test_class_probabilities: list[list[float]]
    roc_fpr: list[float]  # binary test-set ROC; empty if multiclass or only one class occurs
    roc_tpr: list[float]
    preprocessing: PreprocessingSummary
    cross_validation: CrossValidationResult | None  # on the training rows only
    warnings: list[str] = field(default_factory=list)


def svm(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    positive_class: Any = None,
    kernel: str = "rbf",
    C: float = 1.0,
    test_size: float = 0.25,
    random_state: int = 0,
    preprocessing: PreprocessingConfig | None = None,
    cv_folds: int = 5,
) -> SVMResult:
    """Fit a support vector machine classifier of *target* on *features*.

    Target and feature rules match the other classifiers (two classes with a
    *positive_class*, or several). *kernel* is ``"rbf"`` (default), ``"linear"``,
    or ``"poly"``; *C* is the margin penalty. Features are standard-scaled by
    default because the margin is measured in feature units.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {list(KERNELS)}; got {kernel!r}.")
    if not C > 0:
        raise ValueError(f"C must be positive; got {C}.")
    features, frame, labels, warnings = prepare_classification_problem(dataset, target, features, positive_class, "svm")
    binary = len(labels) == 2
    positive_class = labels[1] if binary else None
    config = resolve_config(preprocessing, scale=True)
    if config.scaling == "none":
        warnings.append("SVM margins are measured in feature units; without scaling, large-unit features dominate.")
    if len(frame) > LARGE_DATA_ROWS:
        warnings.append(f"{len(frame)} rows: kernel SVM training time grows roughly quadratically; expect a wait.")
    X_train, X_test, y_train, y_test = train_test_split(
        frame[features], frame[target], test_size=test_size, random_state=random_state, stratify=frame[target]
    )

    def make_model() -> Pipeline:
        return Pipeline([
            ("preprocess", build_preprocessor(dataset, features, config=config)),
            ("svm", SVC(kernel=kernel, C=C, gamma="scale", probability=True, random_state=random_state)),
        ])

    model = make_model()
    model.fit(X_train, y_train)
    cross_validation = training_cross_validation(
        make_model(), X_train, y_train, task_for(labels), cv_folds, random_state, warnings, positive_class
    )
    classifier = model.named_steps["svm"]
    train_probabilities = class_probabilities(model, X_train, labels)
    test_probabilities = class_probabilities(model, X_test, labels)
    test_predicted = model.predict(X_test)
    train_metrics = classification_metrics(y_train, model.predict(X_train), metric_scores(train_probabilities, labels), labels)
    test_metrics = classification_metrics(y_test, test_predicted, metric_scores(test_probabilities, labels), labels)
    if train_metrics.accuracy - test_metrics.accuracy > OVERFIT_GAP:
        warnings.append(
            f"Possible overfitting: train accuracy {train_metrics.accuracy:.3f} vs test "
            f"{test_metrics.accuracy:.3f}. Try a smaller C."
        )
    share = float(classifier.n_support_.sum() / len(X_train))
    if share > 0.5:
        warnings.append(
            f"{share:.0%} of training rows are support vectors: the classes overlap heavily, "
            "so the boundary rests on many uncertain rows."
        )

    coefficients = {}
    if kernel == "linear" and binary:
        # coef_ points toward classes_[1]; flip it if the user chose the other class as positive.
        sign = 1.0 if list(classifier.classes_).index(positive_class) == 1 else -1.0
        names = model.named_steps["preprocess"].get_feature_names_out()
        coefficients = {str(n): float(sign * c) for n, c in zip(names, classifier.coef_[0])}
    if binary and y_test.nunique() == 2:
        fpr, tpr, _ = roc_curve(y_test, test_probabilities[:, 1], pos_label=positive_class)
    else:
        fpr, tpr = [], []

    support = dict(zip(classifier.classes_, classifier.n_support_))
    return SVMResult(
        method="svm", target=target, features=list(features), classes=labels, positive_class=positive_class,
        n_train=len(X_train), n_test=len(X_test), kernel=kernel, C=float(C),
        support_vectors={label: int(support[label]) for label in labels},
        support_vector_share=share,
        coefficients=coefficients,
        train_metrics=train_metrics, test_metrics=test_metrics,
        test_actual=y_test.tolist(), test_predicted=test_predicted.tolist(),
        test_probability=test_probabilities[:, 1].tolist() if binary else [],
        test_class_probabilities=test_probabilities.tolist(),
        roc_fpr=[float(v) for v in fpr], roc_tpr=[float(v) for v in tpr],
        preprocessing=describe_preprocessor(model.named_steps["preprocess"], X_train, dataset, config),
        cross_validation=cross_validation,
        warnings=warnings,
    )
