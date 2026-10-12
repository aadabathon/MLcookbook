"""Linear regression workflow.

This is the template for every future supervised workflow:

1. Validate target and features against the Dataset schema.
2. Build a leakage-safe sklearn Pipeline (preprocessing fitted on train only).
3. Hold out a test split and evaluate on both splits.
4. Return a frozen result dataclass with metrics, coefficients, and warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType
from mlcookbook.evaluation.cross_validation import CrossValidationResult, training_cross_validation
from mlcookbook.evaluation.metrics import RegressionMetrics, regression_metrics
from mlcookbook.preprocessing.builders import build_preprocessor, resolve_config
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.features import resolve_features
from mlcookbook.preprocessing.summary import PreprocessingSummary, describe_preprocessor

ORDINAL_LINEAR_WARNING = (
    "Ordinal encoding gives categories arbitrary integer codes; a linear model treats those codes "
    "as evenly spaced numbers. Prefer one-hot encoding unless the categories are truly ordered."
)


@dataclass(frozen=True)
class LinearRegressionResult(Result):
    method: str
    target: str
    features: list[str]
    n_train: int
    n_test: int
    intercept: float
    coefficients: dict[str, float]  # keyed by post-preprocessing feature name
    train_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    test_actual: list[float]  # held-out targets, for plotting actual vs predicted
    test_predicted: list[float]
    preprocessing: PreprocessingSummary
    cross_validation: CrossValidationResult | None  # on the training rows only
    warnings: list[str] = field(default_factory=list)


def linear_regression(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    test_size: float = 0.25,
    random_state: int = 0,
    preprocessing: PreprocessingConfig | None = None,
    cv_folds: int = 5,
) -> LinearRegressionResult:
    """Fit ordinary least squares of *target* on *features*.

    If *features* is None, all numeric/categorical/boolean/datetime columns
    except the target are used; other columns are skipped with a warning. Rows
    with a missing target are dropped (feature missingness is imputed in-pipeline).

    *preprocessing* configures imputation/scaling/encoding (default: median
    imputation, standard scaling, one-hot encoding). *cv_folds* runs k-fold
    cross-validation on the training rows (0 disables it).

    Note: with the default standard scaling, coefficients for numeric features
    are per standard deviation of the feature.
    """
    schema = dataset.schema()
    if schema.var_type(target) is not VarType.NUMERIC:
        raise ValueError(
            f"linear_regression requires a numeric target; {target!r} is "
            f"{schema.var_type(target).value}."
        )

    features, warnings = resolve_features(dataset, target, features)
    config = resolve_config(preprocessing, scale=True)
    if config.categorical_encoding == "ordinal" and any(
        schema.var_type(f) in (VarType.CATEGORICAL, VarType.BOOLEAN) for f in features
    ):
        warnings.append(ORDINAL_LINEAR_WARNING)

    frame = dataset.df[features + [target]]
    n_missing_target = int(frame[target].isna().sum())
    if n_missing_target:
        warnings.append(f"Dropped {n_missing_target} row(s) with missing target {target!r}.")
        frame = frame.dropna(subset=[target])

    X = frame[features]
    y = frame[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    def make_model() -> Pipeline:
        return Pipeline(
            [
                ("preprocess", build_preprocessor(dataset, features, config=config)),
                ("regress", LinearRegression()),
            ]
        )

    model = make_model()
    model.fit(X_train, y_train)
    cross_validation = training_cross_validation(
        make_model(), X_train, y_train, "regression", cv_folds, random_state, warnings
    )

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = model.named_steps["regress"].coef_
    test_predicted = model.predict(X_test)
    return LinearRegressionResult(
        method="linear_regression",
        target=target,
        features=list(features),
        n_train=len(X_train),
        n_test=len(X_test),
        intercept=float(model.named_steps["regress"].intercept_),
        coefficients={str(name): float(c) for name, c in zip(feature_names, coefs)},
        train_metrics=regression_metrics(y_train, model.predict(X_train)),
        test_metrics=regression_metrics(y_test, test_predicted),
        test_actual=[float(v) for v in y_test],
        test_predicted=[float(v) for v in test_predicted],
        preprocessing=describe_preprocessor(model.named_steps["preprocess"], X_train, dataset, config),
        cross_validation=cross_validation,
        warnings=warnings,
    )
