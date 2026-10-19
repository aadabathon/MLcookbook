"""Compare two regression pipelines on one materialized split and set of folds."""

from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.evaluation.cross_validation import CrossValidationResult, cross_validate_pipeline
from mlcookbook.evaluation.metrics import RegressionMetrics, regression_metrics
from mlcookbook.preprocessing.builders import build_preprocessor, resolve_config
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.summary import PreprocessingSummary, describe_preprocessor
from mlcookbook.regression._problem import prepare_regression_problem


@dataclass(frozen=True)
class ComparisonCandidate(Result):
    model: str
    parameters: dict
    cross_validation: CrossValidationResult
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RegressionComparisonResult(Result):
    method: str
    target: str
    features: list[str]
    random_state: int
    test_size: float
    # Zero-based positions in the original dataset, independent of its index labels.
    train_rows: list[int]
    test_rows: list[int]
    folds: list[dict[str, list[int]]]
    candidates: list[ComparisonCandidate]
    selected_model: str
    selection_metric: str
    train_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    test_actual: list[float]
    test_predicted: list[float]
    preprocessing: PreprocessingSummary
    warnings: list[str] = field(default_factory=list)


def compare_regressors(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    *,
    alpha: float = 1.0,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 5,
    test_size: float = 0.25,
    random_state: int = 0,
    preprocessing: PreprocessingConfig | None = None,
    cv_folds: int = 5,
) -> RegressionComparisonResult:
    """Compare ridge and random forest using shared training-only CV folds.

    Lowest mean CV RMSE selects the pipeline (ridge wins an exact tie). Only
    that pipeline is refitted on all training rows and evaluated on the test
    rows. Both candidates use the same preprocessing configuration, refitted
    separately in each fold. Row positions and folds are returned for audit.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if isinstance(cv_folds, bool) or not isinstance(cv_folds, int) or cv_folds < 2:
        raise ValueError("Model comparison needs at least 2 cross-validation folds.")
    if isinstance(random_state, bool) or not isinstance(random_state, int) or not 0 <= random_state < 2**32:
        raise ValueError("random_state must be an integer from 0 to 2**32 - 1.")
    features, frame, warnings = prepare_regression_problem(dataset, target, features, "compare_regressors")
    source_rows = np.flatnonzero(dataset.df[target].notna().to_numpy())
    train, test = train_test_split(np.arange(len(frame)), test_size=test_size, random_state=random_state)
    if len(train) < 4 or len(test) < 2:
        raise ValueError("Comparison needs at least 4 training rows and 2 test rows after dropping missing targets.")
    folds = min(cv_folds, len(train) // 2)
    if folds != cv_folds:
        warnings.append(f"Cross-validation used {folds} folds instead of {cv_folds}: too few training rows.")
    X_train, y_train = frame[features].iloc[train], frame[target].iloc[train]
    X_test, y_test = frame[features].iloc[test], frame[target].iloc[test]
    splits = list(KFold(n_splits=folds, shuffle=True, random_state=random_state).split(X_train))
    config = resolve_config(preprocessing, scale=True)
    if config.scaling == "none":
        warnings.append("Ridge penalizes coefficients by size; scaling helps when features use different units.")
    if config.categorical_encoding == "ordinal":
        warnings.append("Ordinal category codes imply an order and spacing to ridge; consider one-hot encoding.")
    specifications = [
        ("ridge", {"alpha": alpha}, Ridge(alpha=alpha)),
        ("random_forest", {"n_estimators": n_estimators, "max_depth": max_depth,
                           "min_samples_leaf": min_samples_leaf, "max_features": 1.0},
         RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                               min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=1)),
    ]
    candidates, pipelines = [], {}
    for name, parameters, estimator in specifications:
        pipeline = Pipeline([("preprocess", build_preprocessor(dataset, features, config=config)),
                             ("model", estimator)])
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always", UserWarning)
            cv = cross_validate_pipeline(pipeline, X_train, y_train, "regression", folds, random_state,
                                         splits=splits)
        if not np.isfinite(cv.mean["rmse"]):
            raise ValueError(f"{name} did not produce a finite cross-validation RMSE.")
        notes = list(dict.fromkeys(str(w.message) for w in caught))
        candidates.append(ComparisonCandidate(name, parameters, cv, notes))
        warnings.extend(f"{name}: {note}" for note in notes)
        pipelines[name] = pipeline
    selected = min(candidates, key=lambda candidate: candidate.cross_validation.mean["rmse"])
    model = pipelines[selected.model]
    with python_warnings.catch_warnings(record=True) as caught:
        python_warnings.simplefilter("always", UserWarning)
        model.fit(X_train, y_train)
        train_predicted, test_predicted = model.predict(X_train), model.predict(X_test)
    warnings.extend(dict.fromkeys(str(w.message) for w in caught))
    warnings.append("Selection uses mean CV RMSE, not test performance. Repeated tuning after viewing the test "
                    "score makes it less independent; use new held-out data for a final assessment.")
    return RegressionComparisonResult(
        method="compare_regressors", target=target, features=features, random_state=random_state, test_size=test_size,
        train_rows=source_rows[train].tolist(), test_rows=source_rows[test].tolist(),
        folds=[{"train_rows": source_rows[train[a]].tolist(), "validation_rows": source_rows[train[b]].tolist()}
               for a, b in splits],
        candidates=candidates, selected_model=selected.model, selection_metric="rmse",
        train_metrics=regression_metrics(y_train, train_predicted),
        test_metrics=regression_metrics(y_test, test_predicted),
        test_actual=y_test.astype(float).tolist(), test_predicted=test_predicted.astype(float).tolist(),
        preprocessing=describe_preprocessor(model.named_steps["preprocess"], X_train, dataset, config),
        warnings=warnings,
    )
