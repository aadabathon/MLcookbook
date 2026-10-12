"""Decision tree regression workflow.

The regression twin of ``classification/tree.py``: numerics stay unscaled so
thresholds read in real units, and the fitted tree comes back as plain data.
A leaf predicts the *mean* training target of its rows, and splits are chosen
to reduce squared error, so node "impurity" is the mean squared error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.evaluation.cross_validation import CrossValidationResult, training_cross_validation
from mlcookbook.evaluation.metrics import RegressionMetrics, regression_metrics
from mlcookbook.preprocessing.builders import build_preprocessor, resolve_config
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.summary import PreprocessingSummary, describe_preprocessor
from mlcookbook.regression._problem import overfit_warning, prepare_regression_problem

MAX_CV_FOLDS = 5


@dataclass(frozen=True)
class RegressionTreeNode(Result):
    id: int  # index into RegressionTreeResult.nodes
    depth: int
    n_samples: int
    impurity: float  # mean squared error of the node's training targets
    value: float  # mean training target: the prediction if this node were a leaf
    feature: str | None = None  # split feature; None for a leaf
    threshold: float | None = None  # rows with feature <= threshold go left
    left: int | None = None
    right: int | None = None


@dataclass(frozen=True)
class RegressionTreeRule(Result):
    """One root-to-leaf path."""

    conditions: list[str]
    prediction: float
    n_samples: int
    std: float  # spread of the leaf's training targets around the prediction


@dataclass(frozen=True)
class RegressionDepthScore(Result):
    max_depth: int
    train_r2: float
    cv_r2: float  # cross-validated on the training rows only


@dataclass(frozen=True)
class RegressionTreeResult(Result):
    method: str
    target: str
    features: list[str]
    n_train: int
    n_test: int
    max_depth: int  # requested cap
    min_samples_leaf: int
    depth: int  # actual fitted depth
    n_leaves: int
    nodes: list[RegressionTreeNode]  # nodes[0] is the root
    rules: list[RegressionTreeRule]
    feature_importances: dict[str, float]  # share of squared-error reduction; post-preprocessing names
    depth_curve: list[RegressionDepthScore]
    train_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    test_actual: list[float]
    test_predicted: list[float]
    preprocessing: PreprocessingSummary
    cross_validation: CrossValidationResult | None  # at max_depth, on the training rows only
    warnings: list[str] = field(default_factory=list)


def decision_tree_regression(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    test_size: float = 0.25,
    random_state: int = 0,
    preprocessing: PreprocessingConfig | None = None,
    cv_folds: int = 5,
) -> RegressionTreeResult:
    """Fit a regression tree of a numeric *target* on *features*.

    Feature rules match linear_regression; rows with a missing target are
    dropped. The result includes a depth-vs-R² curve cross-validated on the
    training rows, so you can see where extra depth stops generalizing.
    """
    if max_depth < 1:
        raise ValueError(f"max_depth must be at least 1; got {max_depth}.")
    features, frame, warnings = prepare_regression_problem(dataset, target, features, "decision_tree_regression")
    config = resolve_config(preprocessing, scale=False)
    X_train, X_test, y_train, y_test = train_test_split(
        frame[features], frame[target], test_size=test_size, random_state=random_state
    )

    def make_model(depth: int) -> Pipeline:
        return Pipeline([
            ("preprocess", build_preprocessor(dataset, features, config=config)),
            ("tree", DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_samples_leaf,
                                           random_state=random_state)),
        ])

    model = make_model(max_depth)
    model.fit(X_train, y_train)
    regressor = model.named_steps["tree"]
    names = [str(n) for n in model.named_steps["preprocess"].get_feature_names_out()]
    nodes = _extract_nodes(regressor, names)
    test_predicted = model.predict(X_test)
    train_metrics = regression_metrics(y_train, model.predict(X_train))
    test_metrics = regression_metrics(y_test, test_predicted)
    warnings += overfit_warning(
        train_metrics.r2, test_metrics.r2, "Try a smaller max_depth or larger min_samples_leaf."
    )
    depth_curve = _depth_curve(make_model, X_train, y_train, max(8, max_depth + 2), random_state, warnings)
    cross_validation = training_cross_validation(
        make_model(max_depth), X_train, y_train, "regression", cv_folds, random_state, warnings
    )
    return RegressionTreeResult(
        method="decision_tree_regression",
        target=target,
        features=list(features),
        n_train=len(X_train),
        n_test=len(X_test),
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        depth=int(regressor.get_depth()),
        n_leaves=int(regressor.get_n_leaves()),
        nodes=nodes,
        rules=_extract_rules(nodes),
        feature_importances={n: float(v) for n, v in zip(names, regressor.feature_importances_)},
        depth_curve=depth_curve,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        test_actual=[float(v) for v in y_test],
        test_predicted=[float(v) for v in test_predicted],
        preprocessing=describe_preprocessor(model.named_steps["preprocess"], X_train, dataset, config),
        cross_validation=cross_validation,
        warnings=warnings,
    )


def _extract_nodes(regressor: DecisionTreeRegressor, names: list[str]) -> list[RegressionTreeNode]:
    t = regressor.tree_
    depths = [0] * t.node_count
    nodes = []
    for i in range(t.node_count):  # sklearn numbers nodes in pre-order: parents come first
        left, right = int(t.children_left[i]), int(t.children_right[i])
        is_leaf = left == -1
        if not is_leaf:
            depths[left] = depths[right] = depths[i] + 1
        nodes.append(
            RegressionTreeNode(
                id=i,
                depth=depths[i],
                n_samples=int(t.n_node_samples[i]),
                impurity=float(t.impurity[i]),
                value=float(t.value[i][0][0]),
                feature=None if is_leaf else names[int(t.feature[i])],
                threshold=None if is_leaf else float(t.threshold[i]),
                left=None if is_leaf else left,
                right=None if is_leaf else right,
            )
        )
    return nodes


def _extract_rules(nodes: list[RegressionTreeNode]) -> list[RegressionTreeRule]:
    rules: list[RegressionTreeRule] = []

    def walk(i: int, conditions: list[str]) -> None:
        node = nodes[i]
        if node.left is None:
            rules.append(RegressionTreeRule(list(conditions), node.value, node.n_samples, node.impurity ** 0.5))
            return
        walk(node.left, conditions + [f"{node.feature} <= {node.threshold:.4g}"])
        walk(node.right, conditions + [f"{node.feature} > {node.threshold:.4g}"])

    walk(0, [])
    return rules


def _depth_curve(make_model, X_train, y_train, max_scan: int, random_state: int,
                 warnings: list[str]) -> list[RegressionDepthScore]:
    folds = min(MAX_CV_FOLDS, len(y_train) // 2)
    if folds < 2:
        warnings.append("Skipped depth curve: too few training rows to cross-validate.")
        return []
    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    curve = []
    for depth in range(1, max_scan + 1):
        scores = cross_validate(make_model(depth), X_train, y_train, cv=splitter, scoring="r2",
                                return_train_score=True)
        curve.append(RegressionDepthScore(depth, float(scores["train_score"].mean()),
                                          float(scores["test_score"].mean())))
    return curve
