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
from mlcookbook.evaluation.metrics import RegressionMetrics, regression_metrics
from mlcookbook.preprocessing.builders import build_preprocessor

FEATURE_TYPES = (VarType.NUMERIC, VarType.CATEGORICAL, VarType.BOOLEAN)


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
    warnings: list[str] = field(default_factory=list)


def linear_regression(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    test_size: float = 0.25,
    random_state: int = 0,
) -> LinearRegressionResult:
    """Fit ordinary least squares of *target* on *features*.

    If *features* is None, all numeric/categorical/boolean columns except the
    target are used; other columns are skipped with a warning. Rows with a
    missing target are dropped (feature missingness is imputed in-pipeline).

    Note: coefficients for numeric features are on the standardized scale
    (per standard deviation of the feature), because preprocessing scales them.
    """
    schema = dataset.schema()
    if schema.var_type(target) is not VarType.NUMERIC:
        raise ValueError(
            f"linear_regression requires a numeric target; {target!r} is "
            f"{schema.var_type(target).value}."
        )

    warnings: list[str] = []
    if features is None:
        features = [c.name for c in schema.columns if c.name != target and c.var_type in FEATURE_TYPES]
        skipped = [c.name for c in schema.columns if c.name != target and c.var_type not in FEATURE_TYPES]
        if skipped:
            warnings.append(
                f"Skipped columns with non-feature types (datetime/text/unsupported): {skipped}."
            )
    else:
        if target in features:
            raise ValueError(f"Target {target!r} cannot also be a feature.")
        for name in features:
            var_type = schema.var_type(name)  # raises KeyError for unknown columns
            if var_type not in FEATURE_TYPES:
                raise ValueError(f"Feature {name!r} has unsupported type {var_type.value}.")
    if not features:
        raise ValueError("No usable feature columns.")

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

    model = Pipeline(
        [
            ("preprocess", build_preprocessor(dataset, features)),
            ("regress", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = model.named_steps["regress"].coef_
    return LinearRegressionResult(
        method="linear_regression",
        target=target,
        features=list(features),
        n_train=len(X_train),
        n_test=len(X_test),
        intercept=float(model.named_steps["regress"].intercept_),
        coefficients={str(name): float(c) for name, c in zip(feature_names, coefs)},
        train_metrics=regression_metrics(y_train, model.predict(X_train)),
        test_metrics=regression_metrics(y_test, model.predict(X_test)),
        warnings=warnings,
    )
