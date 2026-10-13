"""Support vector regression.

The regression twin of ``classification/svm.py``. SVR fits a function that
keeps as many training rows as possible inside a tube of half-width
*epsilon* around it; rows inside the tube cost nothing, rows outside pay a
linear penalty scaled by *C*. The rows on or outside the tube are the support
vectors.

C and epsilon would otherwise be in target units (a C that suits a 0-1 target
is useless for prices). So the target is standardized too, inside the
pipeline and on training rows only: *epsilon* is then a share of the target's
standard deviation and C means the same thing for every dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from mlcookbook.classification.svm import KERNELS, LARGE_DATA_ROWS
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.evaluation.cross_validation import CrossValidationResult, training_cross_validation
from mlcookbook.evaluation.metrics import RegressionMetrics, regression_metrics
from mlcookbook.preprocessing.builders import build_preprocessor, resolve_config
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.summary import PreprocessingSummary, describe_preprocessor
from mlcookbook.regression._problem import overfit_warning, prepare_regression_problem


@dataclass(frozen=True)
class SVMRegressionResult(Result):
    method: str
    target: str
    features: list[str]
    n_train: int
    n_test: int
    kernel: str
    C: float
    epsilon: float  # tube half-width, in standard deviations of the training target
    n_support_vectors: int
    support_vector_share: float
    train_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    test_actual: list[float]
    test_predicted: list[float]
    preprocessing: PreprocessingSummary
    cross_validation: CrossValidationResult | None  # on the training rows only
    warnings: list[str] = field(default_factory=list)


def svm_regression(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    kernel: str = "rbf",
    C: float = 1.0,
    epsilon: float = 0.1,
    test_size: float = 0.25,
    random_state: int = 0,
    preprocessing: PreprocessingConfig | None = None,
    cv_folds: int = 5,
) -> SVMRegressionResult:
    """Fit support vector regression of a numeric *target* on *features*.

    *epsilon* is the tube half-width in standard deviations of the target (so
    0.1 ignores errors smaller than a tenth of an SD). Features are standard-
    scaled by default and the target always is, so *C* is unit-free: a large C
    lets the fit follow the training rows closely.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {list(KERNELS)}; got {kernel!r}.")
    if not C > 0:
        raise ValueError(f"C must be positive; got {C}.")
    if not epsilon >= 0:
        raise ValueError(f"epsilon must be non-negative; got {epsilon}.")
    features, frame, warnings = prepare_regression_problem(dataset, target, features, "svm_regression")
    config = resolve_config(preprocessing, scale=True)
    if config.scaling == "none":
        warnings.append("SVR distances are measured in feature units; without scaling, large-unit features dominate.")
    if len(frame) > LARGE_DATA_ROWS:
        warnings.append(f"{len(frame)} rows: kernel SVM training time grows roughly quadratically; expect a wait.")
    X_train, X_test, y_train, y_test = train_test_split(
        frame[features], frame[target], test_size=test_size, random_state=random_state
    )
    def make_model() -> Pipeline:
        return Pipeline([
            ("preprocess", build_preprocessor(dataset, features, config=config)),
            ("svr", TransformedTargetRegressor(
                regressor=SVR(kernel=kernel, C=C, epsilon=epsilon, gamma="scale"), transformer=StandardScaler(),
            )),
        ])

    model = make_model()
    model.fit(X_train, y_train)
    cross_validation = training_cross_validation(
        make_model(), X_train, y_train, "regression", cv_folds, random_state, warnings
    )
    regressor = model.named_steps["svr"].regressor_
    train_metrics = regression_metrics(y_train, model.predict(X_train))
    test_predicted = model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_predicted)
    warnings += overfit_warning(train_metrics.r2, test_metrics.r2, "Try a smaller C or a larger epsilon.")
    n_support = int(len(regressor.support_))
    return SVMRegressionResult(
        method="svm_regression", target=target, features=list(features),
        n_train=len(X_train), n_test=len(X_test), kernel=kernel, C=float(C),
        epsilon=float(epsilon),
        n_support_vectors=n_support, support_vector_share=n_support / len(X_train),
        train_metrics=train_metrics, test_metrics=test_metrics,
        test_actual=[float(v) for v in y_test], test_predicted=[float(v) for v in test_predicted],
        preprocessing=describe_preprocessor(model.named_steps["preprocess"], X_train, dataset, config),
        cross_validation=cross_validation,
        warnings=warnings,
    )
