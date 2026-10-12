"""Random forest regression with training-only preprocessing and held-out evaluation.

The regression twin of ``classification/forest.py``: each tree is grown on a
bootstrap sample, and the forest predicts the average of the trees' leaf means.
Averaging many decorrelated, deep trees keeps their low bias while cancelling
much of their variance.
"""

from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.evaluation.cross_validation import CrossValidationResult, training_cross_validation
from mlcookbook.evaluation.metrics import RegressionMetrics, regression_metrics
from mlcookbook.preprocessing.builders import build_preprocessor, resolve_config
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.summary import PreprocessingSummary, describe_preprocessor
from mlcookbook.regression._problem import overfit_warning, prepare_regression_problem


@dataclass(frozen=True)
class RandomForestRegressionResult(Result):
    method: str
    target: str
    features: list[str]
    n_train: int
    n_test: int
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    max_features: str | int | float | None
    feature_importances: dict[str, float]  # post-preprocessing names; squared-error reduction
    tree_depths: list[int]
    tree_leaf_counts: list[int]
    train_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    test_actual: list[float]
    test_predicted: list[float]
    preprocessing: PreprocessingSummary
    cross_validation: CrossValidationResult | None  # on the training rows only
    warnings: list[str] = field(default_factory=list)


def random_forest_regression(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 5,
    max_features: str | int | float | None = 1.0,
    test_size: float = 0.25,
    random_state: int = 0,
    preprocessing: PreprocessingConfig | None = None,
    cv_folds: int = 5,
) -> RandomForestRegressionResult:
    """Fit a random forest regressor of a numeric *target* on *features*.

    Defaults follow sklearn's regressor: every split may consider all encoded
    features (``max_features=1.0``; smaller values decorrelate the trees more).
    Numeric features are imputed without scaling. Impurity importances are per
    encoded feature, may favor high-cardinality features, and are not causal.
    """
    features, frame, warnings = prepare_regression_problem(dataset, target, features, "random_forest_regression")
    config = resolve_config(preprocessing, scale=False)
    X_train, X_test, y_train, y_test = train_test_split(
        frame[features], frame[target], test_size=test_size, random_state=random_state
    )
    if not X_train.notna().any().any():
        raise ValueError("No observed feature values in the training rows.")

    def make_model() -> Pipeline:
        return Pipeline([
            ("preprocess", build_preprocessor(dataset, features, config=config)),
            ("forest", RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                max_features=max_features, bootstrap=True, random_state=random_state, n_jobs=1,
            )),
        ])

    model = make_model()
    with python_warnings.catch_warnings(record=True) as caught:
        python_warnings.simplefilter("always", UserWarning)
        model.fit(X_train, y_train)
        train_predicted = model.predict(X_train)
        test_predicted = model.predict(X_test)
        cross_validation = training_cross_validation(
            make_model(), X_train, y_train, "regression", cv_folds, random_state, warnings
        )
    warnings.extend(dict.fromkeys(str(w.message) for w in caught if issubclass(w.category, UserWarning)))

    regressor = model.named_steps["forest"]
    names = model.named_steps["preprocess"].get_feature_names_out()
    train_metrics = regression_metrics(y_train, train_predicted)
    test_metrics = regression_metrics(y_test, test_predicted)
    warnings += overfit_warning(
        train_metrics.r2, test_metrics.r2, "Consider limiting max_depth or increasing min_samples_leaf."
    )
    warnings.append(
        "Impurity-based feature importances can favor high-cardinality features; they are not causal effects."
    )
    return RandomForestRegressionResult(
        method="random_forest_regression", target=target, features=list(features),
        n_train=len(X_train), n_test=len(X_test),
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features=max_features,
        feature_importances={str(n): float(v) for n, v in zip(names, regressor.feature_importances_)},
        tree_depths=[int(tree.get_depth()) for tree in regressor.estimators_],
        tree_leaf_counts=[int(tree.get_n_leaves()) for tree in regressor.estimators_],
        train_metrics=train_metrics, test_metrics=test_metrics,
        test_actual=[float(v) for v in y_test], test_predicted=[float(v) for v in test_predicted],
        preprocessing=describe_preprocessor(model.named_steps["preprocess"], X_train, dataset, config),
        cross_validation=cross_validation,
        warnings=warnings,
    )
