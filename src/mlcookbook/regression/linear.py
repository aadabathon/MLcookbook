"""Linear regression workflow (ordinary least squares, ridge, or lasso).

This is the template for every future supervised workflow:

1. Validate target and features against the Dataset schema.
2. Build a leakage-safe sklearn Pipeline (preprocessing fitted on train only).
3. Hold out a test split and evaluate on both splits.
4. Return a frozen result dataclass with metrics, coefficients, and warnings.
"""

from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType
from mlcookbook.evaluation.cross_validation import CrossValidationResult, training_cross_validation
from mlcookbook.evaluation.metrics import RegressionMetrics, regression_metrics
from mlcookbook.preprocessing.builders import build_preprocessor, resolve_config
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.summary import PreprocessingSummary, describe_preprocessor
from mlcookbook.regression._problem import prepare_regression_problem
from mlcookbook.regression.diagnostics import ResidualDiagnostics, residual_diagnostics

PENALTIES = ("none", "ridge", "lasso")
ORDINAL_LINEAR_WARNING = (
    "Ordinal encoding gives categories arbitrary integer codes; a linear model treats those codes "
    "as evenly spaced numbers. Prefer one-hot encoding unless the categories are truly ordered."
)


@dataclass(frozen=True)
class LinearRegressionResult(Result):
    method: str
    target: str
    features: list[str]
    penalty: str  # "none" (OLS), "ridge" (L2), or "lasso" (L1)
    alpha: float | None  # penalty strength; None for OLS
    n_train: int
    n_test: int
    intercept: float
    coefficients: dict[str, float]  # keyed by post-preprocessing feature name
    n_zero_coefficients: int  # lasso sets weak features exactly to zero
    train_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    test_actual: list[float]  # held-out targets, for plotting actual vs predicted
    test_predicted: list[float]
    residual_diagnostics: ResidualDiagnostics  # on the training residuals
    preprocessing: PreprocessingSummary
    cross_validation: CrossValidationResult | None  # on the training rows only
    warnings: list[str] = field(default_factory=list)
    train_rows: list[int] = field(default_factory=list)  # source positions, independent of index labels
    test_rows: list[int] = field(default_factory=list)


def linear_regression(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    test_size: float = 0.25,
    random_state: int = 0,
    preprocessing: PreprocessingConfig | None = None,
    cv_folds: int = 5,
    penalty: str = "none",
    alpha: float = 1.0,
) -> LinearRegressionResult:
    """Fit a linear regression of *target* on *features*.

    *penalty* chooses ordinary least squares (``"none"``), ridge (``"ridge"``,
    shrinks all coefficients toward zero) or lasso (``"lasso"``, can set weak
    coefficients exactly to zero); *alpha* is the penalty strength. Penalties
    act on coefficient size, so they assume comparably scaled features.

    If *features* is None, all numeric/categorical/boolean/datetime columns
    except the target are used; other columns are skipped with a warning. Rows
    with a missing target are dropped (feature missingness is imputed in-pipeline).

    *preprocessing* configures imputation/scaling/encoding (default: median
    imputation, standard scaling, one-hot encoding). *cv_folds* runs k-fold
    cross-validation on the training rows (0 disables it). The result also
    diagnoses the training residuals (normality, constant spread).

    Note: with the default standard scaling, coefficients for numeric features
    are per standard deviation of the feature.
    """
    if penalty not in PENALTIES:
        raise ValueError(f"penalty must be one of {list(PENALTIES)}; got {penalty!r}.")
    if penalty != "none" and not alpha > 0:
        raise ValueError(f"alpha must be positive; got {alpha}.")
    features, frame, warnings = prepare_regression_problem(dataset, target, features, "linear_regression")
    schema = dataset.schema()
    config = resolve_config(preprocessing, scale=True)
    if config.categorical_encoding == "ordinal" and any(
        schema.var_type(f) in (VarType.CATEGORICAL, VarType.BOOLEAN) for f in features
    ):
        warnings.append(ORDINAL_LINEAR_WARNING)
    if penalty != "none" and config.scaling == "none":
        warnings.append(
            "The penalty shrinks coefficients by size, but features are unscaled: "
            "features measured in large units are penalized less. Use a scaling option."
        )

    X = frame[features]
    y = frame[target]
    source_rows = np.flatnonzero(dataset.df[target].notna().to_numpy())
    X_train, X_test, y_train, y_test, train_rows, test_rows = train_test_split(
        X, y, source_rows, test_size=test_size, random_state=random_state
    )

    def make_model() -> Pipeline:
        estimator = {
            "none": LinearRegression,
            "ridge": lambda: Ridge(alpha=alpha),
            "lasso": lambda: Lasso(alpha=alpha, max_iter=10_000),
        }[penalty]()
        return Pipeline(
            [
                ("preprocess", build_preprocessor(dataset, features, config=config)),
                ("regress", estimator),
            ]
        )

    with python_warnings.catch_warnings(record=True) as caught:
        python_warnings.simplefilter("always", ConvergenceWarning)
        model = make_model()
        model.fit(X_train, y_train)
        cross_validation = training_cross_validation(
            make_model(), X_train, y_train, "regression", cv_folds, random_state, warnings
        )
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        warnings.append("The lasso solver did not fully converge; try a larger alpha or scaled features.")

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = model.named_steps["regress"].coef_
    train_predicted = model.predict(X_train)
    test_predicted = model.predict(X_test)
    diagnostics = residual_diagnostics(y_train, train_predicted, warnings)
    return LinearRegressionResult(
        method="linear_regression",
        target=target,
        features=list(features),
        penalty=penalty,
        alpha=None if penalty == "none" else float(alpha),
        n_train=len(X_train),
        n_test=len(X_test),
        intercept=float(model.named_steps["regress"].intercept_),
        coefficients={str(name): float(c) for name, c in zip(feature_names, coefs)},
        n_zero_coefficients=int((coefs == 0).sum()) if penalty == "lasso" else 0,
        train_metrics=regression_metrics(y_train, train_predicted),
        test_metrics=regression_metrics(y_test, test_predicted),
        test_actual=[float(v) for v in y_test],
        test_predicted=[float(v) for v in test_predicted],
        residual_diagnostics=diagnostics,
        preprocessing=describe_preprocessor(model.named_steps["preprocess"], X_train, dataset, config),
        cross_validation=cross_validation,
        train_rows=train_rows.tolist(), test_rows=test_rows.tolist(),
        warnings=warnings,
    )
