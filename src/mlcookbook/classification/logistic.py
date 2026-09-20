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

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType
from mlcookbook.evaluation.metrics import ClassificationMetrics, classification_metrics
from mlcookbook.preprocessing.builders import build_preprocessor

FEATURE_TYPES = (VarType.NUMERIC, VarType.CATEGORICAL, VarType.BOOLEAN)
TARGET_TYPES = (VarType.CATEGORICAL, VarType.BOOLEAN, VarType.NUMERIC)
IMBALANCE_WARN_FRACTION = 0.10  # warn if the minority class is below this share


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
    warnings: list[str] = field(default_factory=list)


def logistic_regression(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    positive_class: Any = None,
    test_size: float = 0.25,
    random_state: int = 0,
) -> LogisticRegressionResult:
    """Fit a binary logistic regression of *target* on *features*.

    The target must have exactly two classes (after dropping missing values).
    *positive_class* picks which one is modeled as "1"; by default it is the
    larger of the two in sorted order (e.g. True over False). The train/test
    split is stratified so both splits contain both classes.

    If *features* is None, all numeric/categorical/boolean columns except the
    target are used; other columns are skipped with a warning. Rows with a
    missing target are dropped (feature missingness is imputed in-pipeline).

    Note: coefficients are log-odds, and for numeric features they are on the
    standardized scale (per standard deviation), because preprocessing scales them.
    """
    schema = dataset.schema()
    target_type = schema.var_type(target)  # raises KeyError for unknown columns
    if target_type not in TARGET_TYPES:
        raise ValueError(
            f"logistic_regression requires a categorical, boolean, or two-valued numeric "
            f"target; {target!r} is {target_type.value}."
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

    y = frame[target]
    classes = sorted(y.unique().tolist())
    if len(classes) != 2:
        raise ValueError(
            f"logistic_regression is binary; target {target!r} has {len(classes)} class(es): {classes}."
        )
    if positive_class is None:
        positive_class = classes[1]
    elif positive_class not in classes:
        raise ValueError(f"positive_class {positive_class!r} is not one of the target classes {classes}.")
    labels = [c for c in classes if c != positive_class] + [positive_class]

    counts = y.value_counts()
    if counts.min() < 2:
        raise ValueError(
            f"Each class needs at least 2 rows for a stratified split; counts: {counts.to_dict()}."
        )
    minority_share = counts.min() / len(y)
    if minority_share < IMBALANCE_WARN_FRACTION:
        warnings.append(
            f"Class imbalance: minority class is {minority_share:.1%} of rows; "
            "accuracy can be misleading, prefer recall/precision/ROC AUC."
        )

    X = frame[features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = Pipeline(
        [
            ("preprocess", build_preprocessor(dataset, features)),
            ("classify", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)

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
        warnings=warnings,
    )
