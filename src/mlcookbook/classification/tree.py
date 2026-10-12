"""Binary decision tree workflow.

Same shape as ``logistic.py``, with two differences worth knowing:

* The preprocessor does NOT scale numerics, so split thresholds stay in the
  feature's real units ("mean radius <= 14.6", not "<= -0.42 SD").
* The result carries the fitted tree itself as a flat list of nodes, plus
  root-to-leaf rules and a depth-vs-accuracy curve, so a UI can draw it
  without touching sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from mlcookbook.classification._binary import prepare_binary_problem
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.evaluation.metrics import ClassificationMetrics, classification_metrics
from mlcookbook.preprocessing.builders import build_preprocessor

OVERFIT_GAP = 0.10  # warn if train accuracy beats test accuracy by more than this
MAX_CV_FOLDS = 5


@dataclass(frozen=True)
class TreeNode(Result):
    id: int  # index into DecisionTreeResult.nodes
    depth: int
    n_samples: int
    impurity: float  # gini
    class_counts: list[float]  # training rows per class, ordered as result.classes
    prediction: Any  # majority class at this node
    feature: str | None = None  # split feature; None for a leaf
    threshold: float | None = None  # rows with feature <= threshold go left
    left: int | None = None
    right: int | None = None


@dataclass(frozen=True)
class TreeRule(Result):
    """One root-to-leaf path."""

    conditions: list[str]
    prediction: Any
    n_samples: int
    confidence: float  # share of the leaf's training rows in the predicted class


@dataclass(frozen=True)
class DepthScore(Result):
    max_depth: int
    train_accuracy: float
    cv_accuracy: float  # cross-validated on the training rows only


@dataclass(frozen=True)
class DecisionTreeResult(Result):
    method: str
    target: str
    features: list[str]
    classes: list[Any]  # [negative_class, positive_class]
    positive_class: Any
    n_train: int
    n_test: int
    max_depth: int  # requested cap
    min_samples_leaf: int
    depth: int  # actual fitted depth
    n_leaves: int
    nodes: list[TreeNode]  # nodes[0] is the root
    rules: list[TreeRule]
    feature_importances: dict[str, float]  # sums to 1; keyed by post-preprocessing name
    depth_curve: list[DepthScore]
    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    warnings: list[str] = field(default_factory=list)


def decision_tree(
    dataset: Dataset,
    target: str,
    features: list[str] | None = None,
    positive_class: Any = None,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    test_size: float = 0.25,
    random_state: int = 0,
) -> DecisionTreeResult:
    """Fit a binary decision tree of *target* on *features*.

    Target and feature rules match logistic_regression: the target must have
    exactly two classes, *positive_class* picks the "1" class, the split is
    stratified, and rows with a missing target are dropped. Numeric features
    are imputed but not scaled, so thresholds are in real units.

    The result includes a depth-vs-accuracy curve (cross-validated on the
    training rows only) so you can see where extra depth stops helping.
    """
    if max_depth < 1:
        raise ValueError(f"max_depth must be at least 1; got {max_depth}.")
    features, frame, labels, warnings = prepare_binary_problem(
        dataset, target, features, positive_class, "decision_tree"
    )
    positive_class = labels[1]
    y = frame[target]

    X = frame[features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    def make_model(depth: int) -> Pipeline:
        return Pipeline(
            [
                ("preprocess", build_preprocessor(dataset, features, scale=False)),
                (
                    "tree",
                    DecisionTreeClassifier(
                        max_depth=depth, min_samples_leaf=min_samples_leaf, random_state=random_state
                    ),
                ),
            ]
        )

    model = make_model(max_depth)
    model.fit(X_train, y_train)
    classifier = model.named_steps["tree"]
    names = [str(n) for n in model.named_steps["preprocess"].get_feature_names_out()]

    nodes = _extract_nodes(classifier, names, labels)
    positive_col = list(classifier.classes_).index(positive_class)

    def score(X_part) -> np.ndarray:
        return model.predict_proba(X_part)[:, positive_col]

    train_metrics = classification_metrics(y_train, model.predict(X_train), score(X_train), labels)
    test_metrics = classification_metrics(y_test, model.predict(X_test), score(X_test), labels)
    if train_metrics.accuracy - test_metrics.accuracy > OVERFIT_GAP:
        warnings.append(
            f"Possible overfitting: train accuracy {train_metrics.accuracy:.3f} vs "
            f"test {test_metrics.accuracy:.3f}. Try a smaller max_depth or larger min_samples_leaf."
        )

    depth_curve = _depth_curve(make_model, X_train, y_train, max(8, max_depth + 2), warnings)

    return DecisionTreeResult(
        method="decision_tree",
        target=target,
        features=list(features),
        classes=labels,
        positive_class=positive_class,
        n_train=len(X_train),
        n_test=len(X_test),
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        depth=int(classifier.get_depth()),
        n_leaves=int(classifier.get_n_leaves()),
        nodes=nodes,
        rules=_extract_rules(nodes),
        feature_importances={n: float(v) for n, v in zip(names, classifier.feature_importances_)},
        depth_curve=depth_curve,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        warnings=warnings,
    )


def _extract_nodes(classifier: DecisionTreeClassifier, names: list[str], labels: list[Any]) -> list[TreeNode]:
    t = classifier.tree_
    classes = list(classifier.classes_)
    order = [classes.index(label) for label in labels]  # tree column for each of [neg, pos]
    depths = [0] * t.node_count
    nodes = []
    for i in range(t.node_count):  # sklearn numbers nodes in pre-order: parents come first
        left, right = int(t.children_left[i]), int(t.children_right[i])
        is_leaf = left == -1
        if not is_leaf:
            depths[left] = depths[right] = depths[i] + 1
        value = t.value[i][0]
        counts = value / value.sum() * t.n_node_samples[i]  # value may be counts or proportions
        nodes.append(
            TreeNode(
                id=i,
                depth=depths[i],
                n_samples=int(t.n_node_samples[i]),
                impurity=float(t.impurity[i]),
                class_counts=[float(counts[j]) for j in order],
                prediction=classes[int(np.argmax(counts))],
                feature=None if is_leaf else names[int(t.feature[i])],
                threshold=None if is_leaf else float(t.threshold[i]),
                left=None if is_leaf else left,
                right=None if is_leaf else right,
            )
        )
    return nodes


def _extract_rules(nodes: list[TreeNode]) -> list[TreeRule]:
    rules: list[TreeRule] = []

    def walk(i: int, conditions: list[str]) -> None:
        node = nodes[i]
        if node.left is None:
            share = max(node.class_counts) / sum(node.class_counts)
            rules.append(TreeRule(list(conditions), node.prediction, node.n_samples, float(share)))
            return
        walk(node.left, conditions + [f"{node.feature} <= {node.threshold:.4g}"])
        walk(node.right, conditions + [f"{node.feature} > {node.threshold:.4g}"])

    walk(0, [])
    return rules


def _depth_curve(make_model, X_train, y_train, max_scan: int, warnings: list[str]) -> list[DepthScore]:
    folds = min(MAX_CV_FOLDS, int(y_train.value_counts().min()))
    if folds < 2:
        warnings.append("Skipped depth curve: too few rows in the smaller class to cross-validate.")
        return []
    curve = []
    for depth in range(1, max_scan + 1):
        scores = cross_validate(
            make_model(depth), X_train, y_train, cv=folds, scoring="accuracy", return_train_score=True
        )
        curve.append(
            DepthScore(
                max_depth=depth,
                train_accuracy=float(scores["train_score"].mean()),
                cv_accuracy=float(scores["test_score"].mean()),
            )
        )
    return curve
