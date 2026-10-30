"""MLcookbook desktop application coordinator.

This is the seam between the two halves of the project — the ONLY module that
knows about both:

    gui/                 dumb presentation: widgets, signals, theme
    src/mlcookbook/      UI-free analytical core

It owns the QApplication, the main window, and navigation. Navigation is
delete-and-redraw: screens are constructed on entry and destroyed on exit.
Routing is data-driven — a ready Recipe's ``key`` looks up a screen factory in
SCREEN_FACTORIES; the factory binds the relevant core entry point to a dumb
screen widget. To bring a new recipe alive: flip its variant to "ready" in
gui/recipes.py, give it a key, add a screen, a demo in DEMOS, and a report
builder in REPORT_BUILDERS.

Launch with:  python -m mlcookbook
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if not getattr(sys, "frozen", False):  # a PyInstaller bundle manages its own paths
    for _p in (str(_ROOT), str(_ROOT / "src")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

from PySide6.QtCore import QCoreApplication, QEvent, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from gui import theme
from gui.front_page import FrontPage
from gui.main_window import MainWindow
from gui.recipes import RECIPES, Recipe
from gui.screens.association_scan import AssociationScanScreen
from gui.screens.data import DataScreen
from gui.screens.decision_tree import DecisionTreeScreen
from gui.screens.kitchen import KitchenScreen
from gui.screens.kmeans import KMeansScreen
from gui.screens.linear_regression import LinearRegressionScreen
from gui.screens.llm import LLMScreen
from gui.screens.logistic_regression import LogisticRegressionScreen
from gui.screens.model_comparison import ModelComparisonScreen
from gui.screens.neural_network import NeuralNetworkScreen
from gui.screens.pca import PCAScreen
from gui.screens.playground import DataPlaygroundScreen
from gui.screens.random_forest import RandomForestScreen
from gui.screens.report_screen import MAX_CLASSES, ReportScreen
from gui.screens.svm import SVMScreen
from gui.screens.wearable import WearableScreen

MAX_PLOT_POINTS = 3000  # scatter plots draw a deterministic sample beyond this


def _assets_dir() -> Path:
    # Frozen builds unpack data files under sys._MEIPASS.
    return Path(getattr(sys, "_MEIPASS", _ROOT)) / "gui" / "assets"


def _playground_data_dir() -> Path:
    """Use the repository's data folder, independent of shortcut working directory."""
    if not getattr(sys, "frozen", False):
        return _ROOT / "data"
    executable = Path(sys.executable).resolve()
    for parent in executable.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "mlcookbook").is_dir():
            return parent / "data"
    # A portable bundle copied outside the checkout keeps downloads beside itself.
    return executable.parent / "data"


# --- Default datasets ---------------------------------------------------------
# Built-in datasets are loaded only after an explicit example action.

def _diabetes_frame():
    from sklearn.datasets import load_diabetes

    return load_diabetes(as_frame=True, scaled=False).frame.rename(columns={"target": "progression"})


def _breast_cancer_frame():
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame[[c for c in bunch.frame.columns if c.startswith("mean ")]].copy()
    frame["diagnosis"] = bunch.target.map(dict(enumerate(bunch.target_names)))  # 0 = malignant
    return frame


def _wine_frame():
    from sklearn.datasets import load_wine

    return load_wine(as_frame=True).data  # 13 chemical measurements of 178 wines, no labels


def _wine_labeled_frame():
    from sklearn.datasets import load_wine

    bunch = load_wine(as_frame=True)  # the three classes are cultivars of Piedmont grapes
    return bunch.data.assign(cultivar=bunch.target.map({0: "barolo", 1: "grignolino", 2: "barbera"}))


def _prompt_frame():
    import pandas as pd

    return pd.DataFrame({"question": [
        "Why should preprocessing be fitted only on training data?",
        "What is the difference between cross-validation and a test holdout?",
        "When would you choose ridge regression over a random forest?",
    ]})


# key -> (frame factory, display name, target, positive class)
def _movement_placeholder():
    import pandas as pd

    return pd.DataFrame({"movement_study": ["Open the wearable export.json to begin."]})


DEMOS = {
    "wearable-study": (_movement_placeholder, "No movement study selected", None, None),
    "llm": (_prompt_frame, "Example prompts", None, None),
    "model-comparison": (_diabetes_frame, "Diabetes", "progression", None),
    "linear-regression": (_diabetes_frame, "Diabetes", "progression", None),
    "logistic-regression": (_breast_cancer_frame, "Breast cancer (Wisconsin)", "diagnosis", "malignant"),
    "decision-tree": (_breast_cancer_frame, "Breast cancer (Wisconsin)", "diagnosis", "malignant"),
    "random-forest": (_breast_cancer_frame, "Breast cancer (Wisconsin)", "diagnosis", "malignant"),
    "svm": (_breast_cancer_frame, "Breast cancer (Wisconsin)", "diagnosis", "malignant"),
    "neural-network": (_wine_labeled_frame, "Wine cultivars", "cultivar", None),
    "kmeans": (_wine_frame, "Wine measurements", None, None),
    "pca": (_wine_frame, "Wine measurements", None, None),
    "association-scan": (_breast_cancer_frame, "Breast cancer (Wisconsin)", None, None),
}


# --- Result formatting (core result -> plain strings for the dumb screens) ----

def _num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None or value != value else f"{value:.{digits}f}"


def _p(value: float | None) -> str:
    if value is None or value != value:
        return "—"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def _engine_info(func) -> dict[str, str]:
    doc = (inspect.getdoc(func) or "").splitlines()
    return {
        "callable": f"{func.__module__}.{func.__qualname__}",
        "signature": str(inspect.signature(func)),
        "summary": (doc[0] if doc else "").replace("*", ""),
    }


def _sample(points: list, limit: int = MAX_PLOT_POINTS) -> list:
    step = max(1, -(-len(points) // limit))
    return points[::step]


def _coefficient_rows(coefficients: dict[str, float], extra: dict[str, float] | None = None):
    rows = []
    for name, value in sorted(coefficients.items(), key=lambda kv: abs(kv[1]), reverse=True):
        row = [name, _num(value)]
        if extra is not None:
            row.append(_num(extra[name]))
        rows.append(row)
    return rows


CV_LABELS = {"r2": "R²", "rmse": "RMSE", "mae": "MAE", "accuracy": "ACCURACY", "f1": "F1",
             "roc_auc": "ROC AUC", "f1_macro": "MACRO F1", "roc_auc_ovr": "ROC AUC (OVR)"}


def _cv_group(cv) -> list[dict]:
    if cv is None:
        return []
    digits = {"rmse": 2, "mae": 2}
    return [{
        "title": f"Cross-validation · {cv.n_folds}-fold {cv.strategy} on {cv.n_rows} training rows · mean ± sd",
        "items": [(CV_LABELS.get(m, m.upper()), f"{_num(cv.mean[m], digits.get(m, 3))} ± {_num(cv.std[m], digits.get(m, 3))}")
                  for m in cv.metrics],
    }]


def _preprocessing_table(summary) -> dict:
    rows = []
    for col in summary.columns:
        for i, step in enumerate(col.steps):
            outputs = col.output_features
            shown = (", ".join(outputs) if len(outputs) <= 3 else f"{len(outputs)} columns") if i == 0 else ""
            rows.append([col.column if i == 0 else "", step, shown])
    return {
        "title": f"Preprocessing · fitted on {summary.n_rows_fitted} rows · {summary.n_output_features} model columns",
        "headers": ["COLUMN", "STEP (FITTED VALUES)", "BECOMES"],
        "rows": rows,
        "stretch_column": 1,
    }


def _regression_group(title: str, m) -> dict:
    return {"title": f"{title} · {m.n_obs} rows",
            "items": [("R²", _num(m.r2)), ("RMSE", _num(m.rmse, 2)), ("MAE", _num(m.mae, 2))]}


def _classification_group(title: str, m) -> dict:
    if m.average == "binary":
        items = [("ACCURACY", _num(m.accuracy)), ("PRECISION", _num(m.precision)), ("RECALL", _num(m.recall)),
                 ("F1", _num(m.f1)), ("ROC AUC", _num(m.roc_auc))]
    else:
        items = [("ACCURACY", _num(m.accuracy)), ("BALANCED ACC.", _num(m.balanced_accuracy)),
                 ("MACRO F1", _num(m.f1)), ("ROC AUC (OVR)", _num(m.roc_auc))]
    return {"title": f"{title} · {m.n_obs} rows", "items": items}


def _confusion_table(m) -> dict:
    return {
        "title": "Confusion matrix (test set) · rows = actual, columns = predicted",
        "headers": ["ACTUAL / PREDICTED", *[str(label).upper() for label in m.labels]],
        "rows": [[str(label), *map(str, row)] for label, row in zip(m.labels, m.confusion_matrix)],
    }


def _per_class_table(m) -> dict:
    return {
        "title": "Per-class metrics (test set, each class vs the rest)",
        "headers": ["CLASS", "PRECISION", "RECALL", "F1", "ROWS"],
        "rows": [[str(c.label), _num(c.precision), _num(c.recall), _num(c.f1), str(c.support)] for c in m.per_class],
    }


def _recall_bars(m) -> dict:
    return {"title": "Recall by class (test set): share of each class found",
            "items": [(str(c.label), c.recall, f"{c.recall:.0%}") for c in m.per_class]}


def _importance_bars(importances: dict[str, float], title: str) -> dict:
    return {"title": title,
            "items": [(name, value, f"{value:.1%}")
                      for name, value in sorted(importances.items(), key=lambda kv: -kv[1]) if value > 0]}


def _actual_vs_predicted(result) -> dict:
    lo = min(result.test_actual + result.test_predicted)
    hi = max(result.test_actual + result.test_predicted)
    return {
        "title": "Actual vs predicted · held-out rows",
        "x_label": f"predicted {result.target}",
        "y_label": f"actual {result.target}",
        "series": [
            {"kind": "line", "color": "muted", "label": "perfect fit", "points": [(lo, lo), (hi, hi)]},
            {"kind": "points", "color": "accent", "label": "test rows",
             "points": _sample(list(zip(result.test_predicted, result.test_actual)))},
        ],
    }


def _residual_plot(result) -> dict:
    residuals = [(p, a - p) for a, p in zip(result.test_actual, result.test_predicted)]
    lo, hi = min(result.test_predicted), max(result.test_predicted)
    return {
        "title": "Residuals vs predicted · held-out rows",
        "x_label": f"predicted {result.target}",
        "y_label": "actual − predicted",
        "series": [
            {"kind": "line", "color": "muted", "label": "zero error", "points": [(lo, 0), (hi, 0)]},
            {"kind": "points", "color": "alt", "label": "test rows", "points": _sample(residuals)},
        ],
    }


def _sigmoid_plot(result) -> dict:
    """Logistic fit: held-out rows on the log-odds axis, with the sigmoid they follow."""
    import math

    eps = 1e-6
    z = [math.log(max(eps, min(1 - eps, p)) / (1 - max(eps, min(1 - eps, p)))) for p in result.test_probability]
    lo, hi = min(z) - 0.5, max(z) + 0.5
    curve = [(lo + (hi - lo) * i / 100, 1 / (1 + math.exp(-(lo + (hi - lo) * i / 100)))) for i in range(101)]
    negative, positive = result.classes
    # Deterministic vertical jitter so overlapping points stay readable.
    def jitter(i: int) -> float:
        return ((i * 0.618034) % 1 - 0.5) * 0.05

    pos = [(zi, 1 + jitter(i)) for i, (zi, a) in enumerate(zip(z, result.test_actual)) if a == positive]
    neg = [(zi, 0 + jitter(i)) for i, (zi, a) in enumerate(zip(z, result.test_actual)) if a != positive]
    return {
        "title": "Fitted sigmoid · held-out rows by true class",
        "x_label": "log-odds of positive class (linear predictor)",
        "y_label": f"P({positive})",
        "series": [
            {"kind": "line", "color": "muted", "label": "sigmoid", "points": curve},
            {"kind": "points", "color": "alt", "label": f"{negative} (actual)", "points": neg},
            {"kind": "points", "color": "accent", "label": f"{positive} (actual)", "points": pos},
        ],
    }


def _target_line(result) -> str:
    if result.positive_class is None:
        return f"{result.target}  ({len(result.classes)} classes: {', '.join(map(str, result.classes))})"
    return f"{result.target}  (positive = {result.positive_class})"


def _base(result, dataset, name, settings: str = "") -> dict:
    report = {"dataset": f"{name} · {len(dataset.df)} rows", "features": ", ".join(result.features)}
    if getattr(result, "target", None):
        report["target"] = _target_line(result) if hasattr(result, "classes") else result.target
    if settings:
        report["settings"] = settings
    metrics = getattr(result, "test_metrics", None)
    if metrics is not None:
        if hasattr(result, "classes"):
            report["summary"] = (f"Correct predictions on {metrics.accuracy:.1%} of {metrics.n_obs} held-out rows. "
                                 "Review recall and the confusion matrix to see which classes were missed.")
        else:
            report["summary"] = (f"Mean absolute error on {metrics.n_obs} held-out rows: {_num(metrics.mae, 2)} "
                                 f"in {result.target} units. Lower error means closer predictions.")
    return report


def _config(request):
    from mlcookbook import PreprocessingConfig

    return PreprocessingConfig(**request["preprocessing"]) if request.get("preprocessing") else None


def _supervised_kwargs(request) -> dict:
    kwargs = {"preprocessing": _config(request)}
    if request.get("cv_folds") is not None:
        kwargs["cv_folds"] = request["cv_folds"]
    return kwargs


# --- Report builders: where a recipe meets its core module ---------------------
# Each takes (configured Dataset, the screen's request dict, display name).

def _linear_regression_report(dataset, request, name) -> dict:
    from mlcookbook.regression import linear_regression

    params = request.get("params", {})
    result = linear_regression(dataset, target=request["target"], features=request["features"],
                               penalty=params.get("penalty", "none"), alpha=params.get("alpha", 1.0),
                               **_supervised_kwargs(request))
    d = result.residual_diagnostics
    penalty = "ordinary least squares" if result.alpha is None else f"{result.penalty}, alpha={result.alpha:g}"
    scaling = result.preprocessing.config.scaling
    per = " (per SD of feature)" if scaling == "standard" else f" ({scaling} scaling)"
    title = f"Coefficients{per} · intercept {_num(result.intercept, 2)}"
    if result.penalty == "lasso":
        title += f" · {result.n_zero_coefficients} set to zero"
    return {
        **_base(result, dataset, name, penalty),
        "plots": [_actual_vs_predicted(result), _residual_plot(result)],
        "metric_groups": [
            _regression_group("Test set", result.test_metrics), _regression_group("Train set", result.train_metrics),
            *_cv_group(result.cross_validation),
            {"title": f"Residual checks · {d.n_obs} training rows", "items": [
                ("SKEW", _num(d.skewness, 2)), ("EXCESS KURTOSIS", _num(d.excess_kurtosis, 2)),
                ("NORMALITY p", _p(d.normality_p_value)), ("SPREAD RHO", _num(d.spread_correlation, 2)),
                ("SPREAD p", _p(d.spread_p_value)),
            ]},
        ],
        "tables": [
            {"title": title, "headers": ["FEATURE", "COEFFICIENT"], "rows": _coefficient_rows(result.coefficients)},
            _preprocessing_table(result.preprocessing),
        ],
        "warnings": result.warnings,
    }


def _logistic_regression_report(dataset, request, name) -> dict:
    from mlcookbook.classification import logistic_regression

    result = logistic_regression(dataset, target=request["target"], features=request["features"],
                                 positive_class=request.get("positive_class"), **_supervised_kwargs(request))
    m = result.test_metrics
    report = {
        **_base(result, dataset, name),
        "metric_groups": [_classification_group("Test set", m), _classification_group("Train set", result.train_metrics),
                          *_cv_group(result.cross_validation)],
        "tables": [_confusion_table(m)],
        "warnings": result.warnings,
    }
    if result.positive_class is not None:
        positive = result.positive_class
        report["plots"] = [_sigmoid_plot(result)]
        report["tables"].append({
            "title": f"Coefficients (log-odds of {positive}) · intercept {_num(result.intercept)}",
            "headers": ["FEATURE", "COEFFICIENT", "ODDS RATIO"],
            "rows": _coefficient_rows(result.coefficients, result.odds_ratios),
        })
    else:
        report["bars"] = [_recall_bars(m)]
        report["tables"].append(_per_class_table(m))
        features = list(next(iter(result.class_coefficients.values())))
        report["tables"].append({
            "title": "Softmax coefficients per class · differences between two classes are log-odds ratios",
            "headers": ["FEATURE", *[str(c).upper() for c in result.classes]],
            "rows": [[f, *[_num(result.class_coefficients[c][f]) for c in result.classes]] for f in features],
        })
    report["tables"].append(_preprocessing_table(result.preprocessing))
    return report


def _depth_plot(points_train, points_cv, chosen, metric) -> dict:
    low = min(y for _, y in points_train + points_cv)
    top = max(y for _, y in points_train + points_cv)
    return {
        "title": f"{metric} vs tree depth · the overfitting curve",
        "x_label": "max_depth",
        "y_label": metric,
        "series": [
            {"kind": "line", "color": "muted", "label": f"chosen depth ({chosen})", "points": [(chosen, low), (chosen, top)]},
            {"kind": "line", "color": "alt", "label": "train", "points": points_train},
            {"kind": "points", "color": "alt", "label": "", "points": points_train},
            {"kind": "line", "color": "accent", "label": "cross-validated", "points": points_cv},
            {"kind": "points", "color": "accent", "label": "", "points": points_cv},
        ],
    }


def _class_legend(classes) -> str:
    return " &nbsp; ".join(f"<span style='color:{theme.palette(f'c{i}')}'>●</span> {c}" for i, c in enumerate(classes)) + \
        " &nbsp;·&nbsp; bar = share of the node's majority class &nbsp;·&nbsp; left branch = condition true (≤)"


def _decision_tree_report(dataset, request, name) -> dict:
    from mlcookbook.classification import decision_tree
    from mlcookbook.regression import decision_tree_regression

    params = request.get("params", {})
    kwargs = dict(target=request["target"], features=request["features"],
                  max_depth=params.get("max_depth", 3), min_samples_leaf=params.get("min_samples_leaf", 5),
                  **_supervised_kwargs(request))
    if request.get("task") == "regression":
        result = decision_tree_regression(dataset, **kwargs)
        values = [n.value for n in result.nodes]
        lo, hi = min(values), max(values)
        tree = {
            "nodes": [{
                "title": f"{n.feature} ≤ {n.threshold:.4g}" if n.left is not None else f"→ {n.value:.4g}",
                "subtitle": f"n={n.n_samples} · mean {n.value:.4g}",
                "fraction": (n.value - lo) / (hi - lo) if hi > lo else 0.5,
                "color": "accent", "left": n.left, "right": n.right,
            } for n in result.nodes],
            "legend": f"bar = node mean of {result.target}, from {lo:.4g} (empty) to {hi:.4g} (full) &nbsp;·&nbsp; "
                      "left branch = condition true (≤)",
        }
        curve = result.depth_curve
        plots = [_actual_vs_predicted(result)]
        if curve:
            plots.insert(0, _depth_plot([(d.max_depth, d.train_r2) for d in curve],
                                        [(d.max_depth, d.cv_r2) for d in curve], result.max_depth, "R²"))
        rules = [[f"IF {' AND '.join(r.conditions).replace('<=', '≤') or 'always'}  THEN  {r.prediction:.4g}",
                  str(r.n_samples), _num(r.std, 2)] for r in sorted(result.rules, key=lambda r: -r.n_samples)]
        return {
            **_base(result, dataset, name, f"regression · max_depth={result.max_depth} · "
                    f"min_samples_leaf={result.min_samples_leaf} · fitted depth {result.depth}, {result.n_leaves} leaves"),
            "tree": tree,
            "plots": plots,
            "metric_groups": [_regression_group("Test set", result.test_metrics),
                              _regression_group("Train set", result.train_metrics), *_cv_group(result.cross_validation)],
            "bars": [_importance_bars(result.feature_importances, "Feature importance (share of squared-error reduction)")],
            "tables": [{"title": "Decision rules (one per leaf) · SD = spread of the leaf's training targets",
                        "headers": ["RULE", "SAMPLES", "SD"], "stretch_column": 0, "rows": rules},
                       _preprocessing_table(result.preprocessing)],
            "warnings": result.warnings,
        }

    result = decision_tree(dataset, positive_class=request.get("positive_class"), **kwargs)
    binary = result.positive_class is not None
    if binary:
        negative, positive = result.classes
        nodes = [{"fraction": n.class_counts[1] / sum(n.class_counts), "positive": n.prediction == positive}
                 for n in result.nodes]
        tree = {"negative_label": str(negative), "positive_label": str(positive)}
    else:
        nodes = [{"fraction": max(n.class_counts) / sum(n.class_counts),
                  "color": f"c{result.classes.index(n.prediction)}"} for n in result.nodes]
        tree = {"legend": _class_legend(result.classes)}
    tree["nodes"] = [{
        **extra,
        "title": f"{n.feature} ≤ {n.threshold:.4g}" if n.left is not None else f"→ {n.prediction}",
        "subtitle": f"n={n.n_samples} · gini {n.impurity:.2f}",
        "left": n.left, "right": n.right,
    } for n, extra in zip(result.nodes, nodes)]
    curve = result.depth_curve
    m = result.test_metrics
    tables = [{
        "title": "Decision rules (one per leaf)",
        "headers": ["RULE", "SAMPLES", "CONFIDENCE"],
        "stretch_column": 0,
        "rows": [[f"IF {' AND '.join(r.conditions).replace('<=', '≤') or 'always'}  THEN  {r.prediction}",
                  str(r.n_samples), f"{r.confidence:.0%}"] for r in sorted(result.rules, key=lambda r: -r.n_samples)],
    }, _confusion_table(m)]
    if not binary:
        tables.append(_per_class_table(m))
    tables.append(_preprocessing_table(result.preprocessing))
    return {
        **_base(result, dataset, name, f"max_depth={result.max_depth} · min_samples_leaf={result.min_samples_leaf}"
                f" · fitted depth {result.depth}, {result.n_leaves} leaves"),
        "tree": tree,
        "plots": [_depth_plot([(d.max_depth, d.train_accuracy) for d in curve],
                              [(d.max_depth, d.cv_accuracy) for d in curve], result.max_depth, "accuracy")]
        if curve else [],
        "metric_groups": [_classification_group("Test set", m), _classification_group("Train set", result.train_metrics),
                          *_cv_group(result.cross_validation)],
        "bars": [_importance_bars(result.feature_importances, "Feature importance (share of impurity reduction)")],
        "tables": tables,
        "warnings": result.warnings,
    }


def _roc_plot(result, model: str) -> list[dict]:
    if not getattr(result, "roc_fpr", None):
        return []
    return [{
        "title": "ROC curve · held-out rows",
        "x_label": "false positive rate",
        "y_label": "true positive rate (recall)",
        "series": [
            {"kind": "line", "color": "muted", "label": "chance", "points": [(0, 0), (1, 1)]},
            {"kind": "line", "color": "accent", "label": f"{model} (AUC {_num(result.test_metrics.roc_auc)})",
             "points": list(zip(result.roc_fpr, result.roc_tpr))},
        ],
    }]


def _classifier_tables(result) -> tuple[list[dict], list[dict]]:
    """(bars, tables) shared by classifier reports: recall bars and per-class table for multiclass."""
    m = result.test_metrics
    if result.positive_class is not None:
        return [], [_confusion_table(m)]
    return [_recall_bars(m)], [_confusion_table(m), _per_class_table(m)]


def _random_forest_report(dataset, request, name) -> dict:
    from mlcookbook.classification import random_forest
    from mlcookbook.regression import random_forest_regression

    params = request.get("params", {})
    kwargs = dict(target=request["target"], features=request["features"],
                  n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth"),
                  min_samples_leaf=params.get("min_samples_leaf", 5), **_supervised_kwargs(request))
    regression = request.get("task") == "regression"
    if regression:
        result = random_forest_regression(dataset, **kwargs)
        metric_groups = [_regression_group("Test set", result.test_metrics),
                         _regression_group("Train set", result.train_metrics)]
        plots = [_actual_vs_predicted(result)]
        bars, tables = [], []
    else:
        result = random_forest(dataset, positive_class=request.get("positive_class"), **kwargs)
        m = result.test_metrics
        metric_groups = [_classification_group("Test set", m), _classification_group("Train set", result.train_metrics)]
        plots = [{
            "title": "ROC curve · held-out rows",
            "x_label": "false positive rate",
            "y_label": "true positive rate (recall)",
            "series": [
                {"kind": "line", "color": "muted", "label": "chance", "points": [(0, 0), (1, 1)]},
                {"kind": "line", "color": "accent", "label": f"forest (AUC {_num(m.roc_auc)})",
                 "points": list(zip(result.roc_fpr, result.roc_tpr))},
            ],
        }] if result.roc_fpr else []
        bars = [] if result.positive_class is not None else [_recall_bars(m)]
        tables = [_confusion_table(m)] + ([] if result.positive_class is not None else [_per_class_table(m)])
    depth = "unlimited" if result.max_depth is None else result.max_depth
    task = "regression" if regression else "classification"
    return {
        **_base(result, dataset, name, f"{task} · {result.n_estimators} trees on bootstrap samples · "
                f"max_features={result.max_features} · max_depth={depth} · min_samples_leaf={result.min_samples_leaf}"),
        "plots": plots,
        "metric_groups": [*metric_groups, *_cv_group(result.cross_validation),
                          {"title": "Forest structure", "items": [
                              ("TREES", str(result.n_estimators)),
                              ("DEPTH RANGE", f"{min(result.tree_depths)} – {max(result.tree_depths)}"),
                              ("LEAF COUNT RANGE", f"{min(result.tree_leaf_counts)} – {max(result.tree_leaf_counts)}"),
                          ]}],
        "bars": [*bars, _importance_bars(result.feature_importances,
                                         "Feature importance (impurity reduction; encoded columns)")],
        "tables": [*tables, _preprocessing_table(result.preprocessing)],
        "warnings": result.warnings,
    }


def _kmeans_report(dataset, request, name) -> dict:
    from mlcookbook.clustering import kmeans

    params = request.get("params", {})
    result = kmeans(dataset, features=request["features"], n_clusters=params.get("n_clusters", 3),
                    max_k=params.get("max_k", 10), preprocessing=_config(request))
    by_cluster: dict[int, list] = {}
    for point, label in zip(result.projection, result.labels):
        by_cluster.setdefault(label, []).append(tuple(point))
    variance = sum(result.projection_variance)
    scan = result.k_scan
    plots = [{
        "title": f"Clusters on the first two principal components · {variance:.0%} of the variance",
        "x_label": "PC1", "y_label": "PC2",
        "series": [{"kind": "points", "color": f"c{c}", "label": f"cluster {c + 1}", "points": _sample(by_cluster.get(c, []))}
                   for c in range(result.n_clusters)],
    }]
    if scan:
        chosen = next((s for s in scan if s.k == result.n_clusters), None)
        inertia = [(s.k, s.inertia) for s in scan]
        silhouette = [(s.k, s.silhouette) for s in scan if s.silhouette == s.silhouette]
        plots.append({"title": "Elbow: inertia vs k (always falls; look for the bend)", "x_label": "k",
                      "y_label": "inertia", "series": [
                          {"kind": "line", "color": "accent", "label": "inertia", "points": inertia},
                          {"kind": "points", "color": "accent", "label": "", "points": inertia},
                          *([{"kind": "points", "color": "alt", "label": f"chosen k={chosen.k}",
                              "points": [(chosen.k, chosen.inertia)]}] if chosen else []),
                      ]})
        plots.append({"title": "Silhouette vs k (higher = better-separated clusters)", "x_label": "k",
                      "y_label": "silhouette", "series": [
                          {"kind": "line", "color": "alt", "label": "silhouette", "points": silhouette},
                          {"kind": "points", "color": "alt", "label": "", "points": silhouette},
                      ]})
    profiles = result.profiles
    rows = [[f"{feature} (mean)", *[_num(p.numeric_means[feature], 2) for p in profiles]]
            for feature in (profiles[0].numeric_means if profiles else {})]
    rows += [[f"{feature} (most common)", *[p.top_levels.get(feature, "—") for p in profiles]]
             for feature in (profiles[0].top_levels if profiles else {})]
    return {
        **_base(result, dataset, name, f"k={result.n_clusters} · k-means++ initialisation, 10 restarts"),
        "summary": f"Grouped {len(result.labels)} rows into {result.n_clusters} clusters. "
                   "Use the profiles and silhouette scan to assess whether the groups are useful.",
        "plots": plots,
        "metric_groups": [{"title": "Fit", "items": [
            ("CLUSTERS", str(result.n_clusters)), ("SILHOUETTE", _num(result.silhouette)),
            ("BEST k (SILHOUETTE)", str(result.best_k_by_silhouette or "—")), ("INERTIA", _num(result.inertia, 1)),
        ]}],
        "bars": [{"title": "Cluster sizes (rows · share)", "items": [(f"cluster {p.cluster + 1}", p.share, f"{p.size} · {p.share:.0%}")
                                                      for p in profiles]}],
        "tables": [{"title": "Cluster profiles (original units)", "headers": ["FEATURE", *[f"CLUSTER {p.cluster + 1}" for p in profiles]],
                    "rows": rows, "stretch_column": 0},
                   _preprocessing_table(result.preprocessing)],
        "warnings": result.warnings,
    }


def _pca_report(dataset, request, name) -> dict:
    from mlcookbook.dimensionality import pca

    result = pca(dataset, features=request["features"], preprocessing=_config(request))
    comps = result.components
    ratio = [(c.index, c.explained_variance_ratio) for c in comps]
    cumulative = [(c.index, c.cumulative_ratio) for c in comps]
    plots = [{"title": "Scree plot · share of variance per component", "x_label": "component", "y_label": "variance share",
              "series": [{"kind": "line", "color": "muted", "label": "90%", "points": [(1, 0.9), (len(comps), 0.9)]},
                         {"kind": "line", "color": "alt", "label": "cumulative", "points": cumulative},
                         {"kind": "points", "color": "alt", "label": "", "points": cumulative},
                         {"kind": "line", "color": "accent", "label": "per component", "points": ratio},
                         {"kind": "points", "color": "accent", "label": "", "points": ratio}]}]
    if len(comps) >= 2:
        plots.append({"title": "Rows on PC1 and PC2", "x_label": f"PC1 ({comps[0].explained_variance_ratio:.0%})",
                      "y_label": f"PC2 ({comps[1].explained_variance_ratio:.0%})",
                      "series": [{"kind": "points", "color": "accent", "label": "rows",
                                  "points": _sample([tuple(s) for s in result.scores])}]})
    shown = comps[:4]
    first = shown[0].loadings
    order = sorted(first, key=lambda f: -abs(first[f]))
    return {
        **_base(result, dataset, name, f"{result.n_components} components · scaling={result.preprocessing.config.scaling}"),
        "summary": f"The first component retains {comps[0].explained_variance_ratio:.1%} of the variation "
                   "in the preprocessed features. Components are weighted combinations of the inputs.",
        "plots": plots,
        "metric_groups": [{"title": "Variance", "items": [
            ("COMPONENTS", str(result.n_components)), ("PC1 SHARE", f"{comps[0].explained_variance_ratio:.1%}"),
            ("FOR 90%", str(result.components_for_90_percent or "—")),
            ("EIGENVALUE > 1", str(result.kaiser_components) if result.kaiser_components is not None else "—"),
        ]}],
        "tables": [{"title": "Loadings (weight of each encoded feature in a component; sorted by |PC1|)",
                    "headers": ["FEATURE", *[f"PC{c.index}" for c in shown]],
                    "rows": [[f, *[_num(c.loadings[f]) for c in shown]] for f in order], "stretch_column": 0},
                   _preprocessing_table(result.preprocessing)],
        "warnings": result.warnings,
    }


TEST_NAMES = {"spearman": "Spearman", "chi_square": "chi-square", "kruskal": "Kruskal-Wallis"}


def _association_scan_report(dataset, request, name) -> dict:
    from mlcookbook.statistics import anova, association_scan, chi_square, correlation

    result = association_scan(dataset, columns=request["features"])
    report = {
        "dataset": f"{name} · {len(dataset.df)} rows",
        "features": ", ".join(result.columns),
        "settings": f"{result.n_pairs} pairs · p-values adjusted for multiple testing (Benjamini-Hochberg)",
        "summary": f"Compared {result.n_pairs} column pairs using tests suited to their types. "
                   "Review effect sizes alongside adjusted p-values; associations do not establish causation.",
        "bars": [{"title": "Strongest associations (effect size, 0 to 1)",
                  "items": [(f"{p.x} ~ {p.y}  ({TEST_NAMES[p.test]})", p.effect_size, _num(p.effect_size))
                            for p in result.pairs[:12]]}],
        "tables": [{"title": "Every pair · strongest first",
                    "headers": ["PAIR", "TEST", "EFFECT", "SIZE", "p", "ADJ. p", "ROWS"], "stretch_column": 0,
                    "rows": [[f"{p.x} ~ {p.y}", TEST_NAMES[p.test], p.effect_measure.replace("_", " "),
                              _num(p.effect_size), _p(p.p_value), _p(p.p_adjusted), str(p.n_obs)]
                             for p in result.pairs]}],
        "warnings": list(result.warnings),
    }
    if not result.pairs:
        return report
    top = result.pairs[0]  # a closer look at the strongest pair, with its dedicated test
    if top.test == "spearman":
        detail = correlation(dataset, top.x, top.y, method="spearman")
        report["plots"] = [{"title": f"Strongest pair: {top.x} vs {top.y}", "x_label": top.x, "y_label": top.y,
                            "series": [{"kind": "points", "color": "accent", "label": "rows",
                                        "points": _sample(list(dataset.df[[top.x, top.y]].dropna()
                                                               .astype(float).itertuples(index=False, name=None)))}]}]
        report["metric_groups"] = [{"title": f"Spearman correlation · {top.x} ~ {top.y}", "items": [
            ("RHO", _num(detail.statistic)), ("95% CI", f"{_num(detail.ci_low, 2)} to {_num(detail.ci_high, 2)}"),
            ("p", _p(detail.p_value)), ("ROWS", str(detail.n_obs))]}]
    elif top.test == "chi_square":
        detail = chi_square(dataset, top.x, top.y)
        report["metric_groups"] = [{"title": f"Chi-square test · {top.x} ~ {top.y}", "items": [
            ("CHI²", _num(detail.statistic, 2)), ("DOF", str(detail.dof)), ("p", _p(detail.p_value)),
            ("CRAMÉR'S V", _num(detail.cramers_v)), ("FISHER p", _p(detail.fisher_p_value))]}]
        report["tables"].insert(0, {"title": f"Observed counts · rows = {top.x}, columns = {top.y}",
                                    "headers": [f"{top.x} / {top.y}".upper(), *[str(v) for v in detail.y_levels]],
                                    "rows": [[str(level), *map(str, row)] for level, row in zip(detail.x_levels, detail.observed)]})
    else:
        value, group = (top.x, top.y) if dataset.schema().var_type(top.x).value == "numeric" else (top.y, top.x)
        detail = anova(dataset, value, group)
        report["metric_groups"] = [{"title": f"Group comparison · {value} by {group}", "items": [
            ("ANOVA F", _num(detail.f_statistic, 2)), ("ANOVA p", _p(detail.p_value)), ("ETA²", _num(detail.eta_squared)),
            ("KRUSKAL p", _p(detail.kruskal_p_value)), ("TRUST", detail.recommended_test.upper())]}]
        report["tables"].insert(0, {"title": f"{value} by {group}", "headers": ["GROUP", "ROWS", "MEAN", "SD", "MEDIAN"],
                                    "rows": [[str(g.level), str(g.n_obs), _num(g.mean, 3), _num(g.std, 3), _num(g.median, 3)]
                                             for g in detail.groups]})
    report["warnings"] += [f"Strongest pair: {w}" for w in detail.warnings]
    return report


def _svm_report(dataset, request, name) -> dict:
    from mlcookbook.classification import svm
    from mlcookbook.regression import svm_regression

    params = request.get("params", {})
    kwargs = dict(target=request["target"], features=request["features"], kernel=params.get("kernel", "rbf"),
                  C=params.get("C", 1.0), **_supervised_kwargs(request))
    if request.get("task") == "regression":
        result = svm_regression(dataset, epsilon=params.get("epsilon", 0.1), **kwargs)
        return {
            **_base(result, dataset, name, f"regression · {result.kernel} kernel · C={result.C:g} · "
                    f"epsilon={result.epsilon:g} target SDs (target standardized on training rows)"),
            "plots": [_actual_vs_predicted(result), _residual_plot(result)],
            "metric_groups": [_regression_group("Test set", result.test_metrics),
                              _regression_group("Train set", result.train_metrics), *_cv_group(result.cross_validation),
                              {"title": "Support vectors (rows on or outside the epsilon tube)", "items": [
                                  ("SUPPORT VECTORS", str(result.n_support_vectors)),
                                  ("SHARE OF TRAINING ROWS", f"{result.support_vector_share:.0%}")]}],
            "tables": [_preprocessing_table(result.preprocessing)],
            "warnings": result.warnings,
        }
    result = svm(dataset, positive_class=request.get("positive_class"), **kwargs)
    bars, tables = _classifier_tables(result)
    if result.coefficients:
        tables.append({"title": f"Boundary weights (linear kernel, toward {result.positive_class})",
                       "headers": ["FEATURE", "WEIGHT"], "rows": _coefficient_rows(result.coefficients)})
    return {
        **_base(result, dataset, name, f"{result.kernel} kernel · C={result.C:g} · Platt-scaled probabilities"),
        "plots": _roc_plot(result, "SVM"),
        "metric_groups": [_classification_group("Test set", result.test_metrics),
                          _classification_group("Train set", result.train_metrics), *_cv_group(result.cross_validation),
                          {"title": "Support vectors (training rows that define the boundary)", "items": [
                              *[(f"CLASS {label}".upper(), str(n)) for label, n in result.support_vectors.items()],
                              ("SHARE OF TRAINING ROWS", f"{result.support_vector_share:.0%}")]}],
        "bars": bars,
        "tables": [*tables, _preprocessing_table(result.preprocessing)],
        "warnings": result.warnings,
    }


def _training_plots(training) -> list[dict]:
    loss = list(zip(range(1, len(training.loss_curve) + 1), training.loss_curve))
    plots = [{"title": "Training loss per epoch", "x_label": "epoch", "y_label": "loss", "series": [
        {"kind": "line", "color": "accent", "label": "training loss", "points": loss}]}]
    if training.validation_curve:
        validation = list(zip(range(1, len(training.validation_curve) + 1), training.validation_curve))
        best = max(validation, key=lambda point: point[1])
        plots.append({
            "title": f"Validation {training.validation_metric} per epoch (10% of training rows held out)",
            "x_label": "epoch", "y_label": training.validation_metric, "series": [
                {"kind": "line", "color": "alt", "label": "validation", "points": validation},
                {"kind": "points", "color": "accent", "label": f"best (epoch {best[0]})", "points": [best]},
            ]})
    return plots


def _network_group(training) -> dict:
    return {"title": "Network", "items": [
        ("ARCHITECTURE", " → ".join(map(str, training.architecture))),
        ("PARAMETERS", f"{training.n_parameters:,}"),
        ("EPOCHS", str(training.epochs)),
        ("CONVERGED", "yes" if training.converged else "no"),
    ]}


def _neural_network_report(dataset, request, name) -> dict:
    from mlcookbook.classification import neural_network
    from mlcookbook.regression import neural_network_regression

    params = request.get("params", {})
    layers = tuple(int(n) for n in str(params.get("hidden_layers", "64,32")).split(","))
    kwargs = dict(target=request["target"], features=request["features"], hidden_layers=layers,
                  activation=params.get("activation", "relu"), alpha=params.get("alpha", 1e-4),
                  max_epochs=params.get("max_epochs", 500), early_stopping=params.get("early_stopping", True),
                  **_supervised_kwargs(request))
    regression = request.get("task") == "regression"
    if regression:
        result = neural_network_regression(dataset, **kwargs)
        metric_groups = [_regression_group("Test set", result.test_metrics),
                         _regression_group("Train set", result.train_metrics)]
        plots, bars, tables = [_actual_vs_predicted(result)], [], []
    else:
        result = neural_network(dataset, positive_class=request.get("positive_class"), **kwargs)
        metric_groups = [_classification_group("Test set", result.test_metrics),
                         _classification_group("Train set", result.train_metrics)]
        plots = []
        bars, tables = _classifier_tables(result)
    t = result.training
    task = "regression" if regression else "classification"
    return {
        **_base(result, dataset, name, f"{task} · {t.activation} · alpha={t.alpha:g} · Adam, learning rate "
                f"{t.learning_rate:g}, batches of 32 · early stopping {'on' if t.early_stopping else 'off'}"),
        "plots": [*_training_plots(t), *plots],
        "metric_groups": [*metric_groups, *_cv_group(result.cross_validation), _network_group(t)],
        "bars": bars,
        "tables": [*tables, _preprocessing_table(result.preprocessing)],
        "warnings": result.warnings,
    }


def _comparison_report(dataset, request, name):
    from mlcookbook.comparison import compare_regressors

    result = compare_regressors(dataset, target=request["target"], features=request["features"],
                                preprocessing=_config(request), cv_folds=request["cv_folds"],
                                **request["params"])
    return _comparison_data_report(result.to_dict(), name, len(dataset.df))


def _comparison_data_report(result: dict, name: str, rows: int) -> dict:
    """Format existing comparison data; inspecting a Kitchen record never fits."""
    names = {"ridge": "Ridge", "random_forest": "Random forest"}
    chosen = names[result["selected_model"]]
    candidates = result["candidates"]
    low = min(result["test_actual"] + result["test_predicted"])
    high = max(result["test_actual"] + result["test_predicted"])
    preprocessing = result["preprocessing"]
    steps = []
    for col in preprocessing["columns"]:
        for index, step in enumerate(col["steps"]):
            outputs = col["output_features"]
            shown = (", ".join(outputs) if len(outputs) <= 3 else f"{len(outputs)} columns") if index == 0 else ""
            steps.append([col["column"] if index == 0 else "", step, shown])
    metrics = [{"title": f"{chosen} · {label} · {result[key]['n_obs']} rows",
                "items": [(caption, _num(result[key][metric], digits))
                          for caption, metric, digits in (("R²", "r2", 3), ("RMSE", "rmse", 2), ("MAE", "mae", 2))]}
               for key, label in (("test_metrics", "final test set"), ("train_metrics", "training set"))]
    ridge, forest = candidates[0]["parameters"], candidates[1]["parameters"]
    return {
        "dataset": f"{name} · {rows} rows", "target": result["target"], "features": ", ".join(result["features"]),
        "settings": f"Shared split · seed {result['random_state']} · {len(result['train_rows'])} train / "
                    f"{len(result['test_rows'])} test rows · {len(result['folds'])} CV folds",
        "summary": f"{chosen} had the lowest mean training-CV RMSE and was selected before test evaluation. "
                   "Both candidates use identical folds and preprocessing choices.",
        "plots": [{"title": "Model comparison · training cross-validation", "x_label": "fold", "y_label": "RMSE (lower is better)",
                   "series": [{"kind": "line", "color": color, "label": names[candidate["model"]],
                               "points": [(fold["fold"], fold["scores"]["rmse"]) for fold in candidate["cross_validation"]["folds"]]}
                              for candidate, color in zip(candidates, ("accent", "alt"))]},
                  {"title": "Actual vs predicted · held-out rows", "x_label": f"predicted {result['target']}",
                   "y_label": f"actual {result['target']}", "series": [
                       {"kind": "line", "color": "muted", "label": "perfect fit", "points": [(low, low), (high, high)]},
                       {"kind": "points", "color": "accent", "label": "test rows",
                        "points": _sample(list(zip(result["test_predicted"], result["test_actual"])))}]}],
        "metric_groups": metrics,
        "tables": [{"title": "Select by cross-validation · mean ± sample SD", "headers": ["MODEL", "RMSE", "MAE", "R²"],
                    "rows": [[names[c["model"]] + (" · selected" if c["model"] == result["selected_model"] else ""),
                              *[f"{_num(c['cross_validation']['mean'][m])} ± {_num(c['cross_validation']['std'][m])}"
                                for m in ("rmse", "mae", "r2")]] for c in candidates]},
                   {"title": "Candidate settings", "headers": ["MODEL", "SETTINGS"],
                    "rows": [["Ridge", f"Penalty strength: {ridge['alpha']:g}"],
                             ["Random forest", f"{forest['n_estimators']} trees · depth: {forest['max_depth'] or 'unlimited'} · "
                              f"min leaf: {forest['min_samples_leaf']}"]]},
                   {"title": f"Preprocessing · fitted on {preprocessing['n_rows_fitted']} rows · "
                             f"{preprocessing['n_output_features']} model columns",
                    "headers": ["COLUMN", "STEP (FITTED VALUES)", "BECOMES"], "rows": steps, "stretch_column": 1}],
        "warnings": list(result["warnings"]), "comparison": deepcopy(result),
    }


def _kitchen_linear_report(result: dict, name: str) -> dict:
    """Format a completed linear result, including an offline saved result."""
    from types import SimpleNamespace

    plotted = SimpleNamespace(target=result["target"], test_actual=result["test_actual"],
                              test_predicted=result["test_predicted"])
    preprocessing = result["preprocessing"]
    fitted = SimpleNamespace(**{**preprocessing, "columns": [SimpleNamespace(**column)
                                                           for column in preprocessing["columns"]]})
    metrics = [_regression_group(label, SimpleNamespace(**result[key]))
               for key, label in (("test_metrics", "Test set"), ("train_metrics", "Training set"))]
    cv = result["cross_validation"]
    if cv is not None:
        metrics.extend(_cv_group(SimpleNamespace(**cv)))
    diagnostics = result["residual_diagnostics"]
    metrics.append({"title": f"Residual checks · {diagnostics['n_obs']} training rows", "items": [
        ("Skew", _num(diagnostics["skewness"], 2)), ("Excess kurtosis", _num(diagnostics["excess_kurtosis"], 2)),
        ("Normality p", _p(diagnostics["normality_p_value"])), ("Spread rho", _num(diagnostics["spread_correlation"], 2)),
        ("Spread p", _p(diagnostics["spread_p_value"])),
    ]})
    penalty = "Ordinary least squares" if result["alpha"] is None else f"{result['penalty']} · alpha {result['alpha']:g}"
    mae = _num(result["test_metrics"]["mae"], 2)
    return {
        "dataset": f"{name} · {result['input_rows']} rows", "target": result["target"],
        "features": ", ".join(result["features"]), "settings": penalty,
        "summary": f"Mean absolute error on {result['n_test']} held-out rows: {mae} in target units.",
        "plots": [_actual_vs_predicted(plotted), _residual_plot(plotted)], "metric_groups": metrics,
        "tables": [{"title": f"Coefficients · intercept {_num(result['intercept'], 2)}",
                    "headers": ["Feature", "Coefficient"], "rows": _coefficient_rows(result["coefficients"])},
                   _preprocessing_table(fitted)],
        "warnings": list(result["warnings"]),
    }


def _llm_report(dataset, request, name, *, stop_requested=None, on_progress=None):
    from mlcookbook import generate_text

    params = dict(request["params"])
    workflow = params.pop("workflow", "generate")
    for key in ("template", "field_spec", "references"):
        params.pop(key, None)
    if workflow != "generate":
        return _run_llm_workflow(dataset, request, name, stop_requested=stop_requested, on_progress=on_progress)
    backend = params.pop("backend", "pytorch")
    ollama_params = {key: params.pop(key, default) for key, default in (
        ("ollama_url", "http://127.0.0.1:11434"), ("ollama_model", ""),
        ("context_tokens", 4096), ("thinking", "off"))}
    cpp_params = {key: params.pop(key, default) for key, default in (
        ("llamacpp_url", "http://127.0.0.1:8080"), ("llamacpp_model", ""), ("gguf_path", ""))}
    mode = params.pop("input_mode")
    if mode == "rows" and len(request["features"]) != 1:
        raise ValueError("Select exactly one input text column for dataset rows.")
    column = request["features"][0] if mode == "rows" else None
    if backend == "llamacpp":
        from mlcookbook.llamacpp import generate_llamacpp

        for key in ("model_path", "device", "max_input_tokens", "prompt_format"):
            params.pop(key)
        result = generate_llamacpp(dataset, text_column=column, stop_requested=stop_requested, on_progress=on_progress,
                                   model=cpp_params["llamacpp_model"], base_url=cpp_params["llamacpp_url"],
                                   gguf_path=cpp_params["gguf_path"] or None, **params)
        return _llamacpp_report(result, dataset, column, name)
    if backend == "ollama":
        from mlcookbook.ollama import generate_ollama

        for key in ("model_path", "device", "max_input_tokens"):
            params.pop(key)
        result = generate_ollama(dataset, text_column=column, stop_requested=stop_requested, on_progress=on_progress,
                                 base_url=ollama_params["ollama_url"], model=ollama_params["ollama_model"],
                                 context_tokens=ollama_params["context_tokens"], thinking=ollama_params["thinking"], **params)
        return _ollama_report(result, dataset, column, name)
    if backend != "pytorch":
        raise ValueError("Choose a supported inference engine.")
    result = generate_text(dataset, text_column=column, stop_requested=stop_requested, **params)
    seconds = sum(output.seconds for output in result.outputs)
    count = sum(output.generated_tokens for output in result.outputs)
    return {
        "dataset": f"{name} · {len(dataset.df)} rows" if column else "Typed prompt",
        "features": column or "Prompt text",
        "settings": f"{result.model_type} · {result.device} · {result.prompt_format} prompt · "
                    f"temperature {params['temperature']:g} · seed {params['random_state']}",
        "summary": f"Generated {count} tokens across {len(result.outputs)} output(s). "
                   "These are language-model outputs; no predictive accuracy was measured.",
        "text_sections": [{"title": "Generated text" + (f" · source row {output.row_position}" if column else ""),
                           "text": output.text or "[The model returned only special tokens; no visible text. Try a different prompt.]"}
                          for output in result.outputs],
        "metric_groups": [{"title": "Generation", "items": [
            ("OUTPUT TOKENS", str(count)), ("TOKENS / SECOND", _num(count / seconds if seconds else 0, 1)),
            ("MODEL LOAD", f"{result.load_seconds:.1f} s"), ("PARAMETERS", f"{result.parameter_count:,}")]}],
        "tables": [{"title": "Output details", "headers": ["SOURCE ROW", "INPUT TOKENS", "OUTPUT TOKENS", "SECONDS", "STOP"],
                    "rows": [[str(output.row_position) if column else "Prompt", str(output.input_tokens),
                              str(output.generated_tokens), _num(output.seconds, 2), output.finish_reason]
                             for output in result.outputs]},
                   {"title": "Model and runtime", "headers": ["SETTING", "VALUE"], "rows": [
                       ["Model folder", result.model_path], ["Model fingerprint (SHA-256)", result.model_sha256],
                       ["Context window", str(result.context_window)], ["Precision", result.dtype],
                       *[[key, value] for key, value in result.versions.items()]]}],
        "warnings": result.warnings,
        "generation": result.to_dict(),
    }


def _run_llm_workflow(dataset, request, name, *, stop_requested=None, on_progress=None, previous=None, positions=None):
    from mlcookbook.llm_workflow import run_workflow

    params = request["params"]
    backend = params["backend"]
    settings = {key: params[key] for key in ("max_new_tokens", "temperature", "top_p", "random_state", "max_seconds")}
    if backend == "ollama":
        settings.update(context_tokens=params["context_tokens"], thinking=params["thinking"], prompt_format="chat")
    elif backend == "llamacpp":
        settings["gguf_path"] = params["gguf_path"] or None
    run = run_workflow(dataset, backend=backend, model=params.get(backend + "_model", ""),
                       base_url=params.get(backend + "_url", ""), template=params["template"],
                       field_spec=params["field_spec"], references=params["references"], bindings=request["features"],
                       max_rows=params["max_rows"], settings=settings, previous=previous, positions=positions,
                       stop_requested=stop_requested, on_progress=on_progress).to_dict()
    return _workflow_report(run, dataset, name)


def _workflow_report(run, dataset, name, *, history=None, compare=False):
    from mlcookbook.llm_workflow import compare_runs, run_metrics

    metrics = run_metrics(dataset, run)
    fields = run["config"]["fields"]
    def shown(value, digits=2):
        return "Unavailable" if value is None else _num(value, digits)
    report = {
        "dataset": f"{name} · {len(dataset.df)} rows", "features": ", ".join(run["config"]["bindings"]),
        "settings": f"{run['config']['backend']} · {run['config']['model']} · run {run['run_id'][:8]}",
        "summary": f"{metrics['valid']} of {metrics['rows']} source rows passed output validation. "
                   "Review the generated fields before using them. Validation measures format and types, not factual accuracy.",
        "metric_groups": [{"title": "Text workflow", "items": [
            ("VALID ROWS", f"{metrics['valid']} / {metrics['rows']}"),
            ("WALL TIME", f"{metrics['wall_seconds']:.2f} s"),
            ("TOKENS / WALL SECOND", shown(metrics['tokens_per_second']))]}],
        "tables": [{"title": "Generated fields beside source inputs · zero-based rows",
                    "headers": ["ROW", "OUTCOME", *run["config"]["bindings"],
                                *[f"{field['name']} ({field['role']})" for field in fields], "DETAILS"],
                    "rows": [[str(row["position"]), row["status"],
                              *[str(row["inputs"].get(column) or "[missing]") for column in run["config"]["bindings"]],
                              *[str(row["values"][field["name"]]) if row["values"] and row["values"][field["name"]] is not None else "null" for field in fields],
                              "; ".join(row["errors"])] for row in run["rows"]]},
                   {"title": "Run identity", "headers": ["SETTING", "VALUE"], "rows": [
                       ["Run", run["run_id"]], ["Source fingerprint", run["dataset_fingerprint"]],
                       ["Configuration fingerprint", run["config_fingerprint"]],
                       ["Nullable output roles", ", ".join(f"{field['name']}: {field['role']}" for field in fields)],
                       ["Parent run (retry)", run["parent_run_id"] or "None"]]}],
        "warnings": list(run["warnings"]), "workflow_run": run, "workflow_history": history or [run],
    }
    from mlcookbook.experiments import runtime_versions

    report["provenance"] = {"versions": runtime_versions(), "created_utc": run["created_utc"],
                             "random_state": run["config"]["settings"].get("random_state", 0)}
    if metrics["reference_agreement"]:
        report["tables"].append({"title": "Reference agreement on valid, non-null pairs only",
                                 "headers": ["FIELD", "REFERENCE", "EVALUATED ROWS", "EXACT AGREEMENT"],
                                 "rows": [[field, value["reference_column"], str(value["evaluated_rows"]),
                                           shown(value["agreement"])] for field, value in metrics["reference_agreement"].items()]})
    if compare:
        compared = compare_runs(dataset, history)
        report["tables"].insert(0, {"title": "Saved run comparison · same source rows and output schema",
                                    "headers": ["RUN", "MODEL", "VALID / ALL", "WALL s", "TOKENS / WALL s", "LOAD s", "DECODE s", "CHARACTERS"],
                                    "rows": [[value["run_id"][:8], saved["config"]["model"],
                                              f"{value['valid']} / {value['rows']}", shown(value["wall_seconds"]),
                                              shown(value["tokens_per_second"]), shown(value["load_seconds"]),
                                              shown(value["generation_seconds"]), str(value["output_characters"])]
                                             for saved, value in zip(history, compared)]})
        report["plots"] = [{"title": title + " · run order matches comparison table", "x_label": "Saved run number", "y_label": label,
                            "series": [{"kind": "points", "color": "accent", "label": title,
                                        "points": [(index + 1, value[key]) for index, value in enumerate(compared) if value[key] is not None]}]}
                           for title, label, key in (("Output validation", "Valid rows / all source rows", "validation_rate"),
                                                     ("Generation speed", "Tokens / wall second", "tokens_per_second"),
                                                     ("Request duration", "Wall seconds", "wall_seconds"),
                                                     ("Server model load", "Reported seconds", "load_seconds"),
                                                     ("Server decode time", "Reported seconds", "generation_seconds"),
                                                     ("Output length", "Characters", "output_characters"))]
        report["warnings"].append("Timing includes runtime cache/load effects. Faster or schema-valid output does not establish better semantic quality.")
    return report


def _ollama_report(result, dataset, column, name):
    """Format server-reported usage separately from locally measured wall time."""
    def available(value, digits=None):
        return "Unavailable" if value is None else _num(value, digits) if digits is not None else str(value)

    tokens = [output.generated_tokens for output in result.outputs]
    count = sum(tokens) if all(value is not None for value in tokens) else None
    seconds = sum(output.seconds for output in result.outputs)
    sections = []
    for output in result.outputs:
        suffix = f" · source row {output.row_position}" if column else ""
        sections.append({"title": "Generated text" + suffix, "text": output.text or "[No visible answer returned.]"})
        if output.thinking:
            sections.append({"title": "Thinking" + suffix, "text": output.thinking})
        sections.append({"title": "Submitted prompt" + suffix, "text": output.submitted_prompt})
    details = result.model.details
    return {
        "dataset": f"{name} · {len(dataset.df)} rows" if column else "Typed prompt",
        "features": column or "Prompt text",
        "settings": f"Ollama · {result.model.name} · {result.prompt_format} · "
                    f"temperature {result.settings['requested_options']['temperature']:g} · "
                    f"seed {result.settings['requested_options']['seed']}",
        "summary": f"Completed {len(result.outputs)} output(s) using a local Ollama model. "
                   "Generated text has not been evaluated for accuracy.",
        "text_sections": sections,
        "metric_groups": [{"title": "Generation", "items": [
            ("OUTPUT TOKENS", available(count)),
            ("WALL TIME", f"{seconds:.2f} s"),
            ("TOKENS / WALL SECOND", available(count / seconds if count is not None and seconds else None, 1)),
            ("QUANTIZATION", str(details.get("quantization_level", "Unavailable")))]}],
        "tables": [
            {"title": "Output details", "headers": ["SOURCE ROW", "INPUT TOKENS", "OUTPUT TOKENS", "WALL (S)", "STOP"],
             "rows": [[str(output.row_position) if column else "Prompt", available(output.input_tokens),
                       available(output.generated_tokens), _num(output.seconds, 2), available(output.finish_reason)]
                      for output in result.outputs]},
            {"title": "Server timing", "headers": ["OUTPUT", "TOTAL (S)", "LOAD (S)", "DECODE (S)", "DECODE TOKENS/S"],
             "rows": [[str(i + 1), available(output.server_seconds, 2), available(output.load_seconds, 2),
                       available(output.generation_seconds, 2),
                       available(output.generated_tokens / output.generation_seconds
                                 if output.generated_tokens is not None and output.generation_seconds else None, 1)]
                      for i, output in enumerate(result.outputs)]},
            {"title": "Model and runtime", "headers": ["SETTING", "VALUE"], "rows": [
                ["Backend", "Ollama"], ["Address", result.base_url], ["Model", result.model.name],
                ["Model digest", available(result.model.digest)], ["Size (bytes)", available(result.model.size_bytes)],
                ["Parameters", str(details.get("parameter_size", "Unavailable"))],
                ["Requested context budget", str(result.settings["requested_options"]["num_ctx"])],
                ["Thinking", result.settings["thinking"]], ["Ollama version", available(result.versions["ollama"])]]}],
        "warnings": result.warnings,
        "generation": result.to_dict(),
    }


def _llamacpp_report(result, dataset, column, name):
    from mlcookbook.llamacpp import timing_seconds

    def available(value, digits=None):
        return "Unavailable" if value is None else _num(value, digits) if digits is not None else str(value)

    tokens = [output.generated_tokens for output in result.outputs]
    count = sum(tokens) if all(value is not None for value in tokens) else None
    seconds = sum(output.seconds for output in result.outputs)
    meta = result.model.metadata.get("meta") or {}
    options = result.settings["requested_options"]
    sections = []
    for output in result.outputs:
        suffix = f" · source row {output.row_position}" if column else ""
        sections.append({"title": "Generated text" + suffix, "text": output.text or "[No visible answer returned.]"})
        if output.thinking:
            sections.append({"title": "Reasoning" + suffix, "text": output.thinking})
        sections.append({"title": "Submitted user message" + suffix, "text": output.submitted_prompt})
    identity = [["Backend", "llama.cpp"], ["Address", result.base_url], ["Model reference (unverified)", result.model.name],
                ["Server model path (reference)", available(result.server_properties.get("model_path"))],
                ["Runtime version (reported)", available(result.versions["llama.cpp"])],
                ["GGUF type (reported)", available(meta.get("ftype"))],
                ["Parameters (reported)", available(meta.get("n_params"))],
                ["Context (server configured)", available(meta.get("n_ctx"))]]
    if result.local_file:
        identity += [["Selected local file", result.local_file["path"]], ["Selected file SHA-256", result.local_file["sha256"]],
                     ["Fingerprint scope", result.local_file["scope"]]]
    return {
        "dataset": f"{name} · {len(dataset.df)} rows" if column else "Typed prompt", "features": column or "Prompt text",
        "settings": f"llama.cpp · {result.model.name} · server chat template · temperature {options['temperature']:g} · seed {options['seed']}",
        "summary": f"Completed {len(result.outputs)} output(s) using llama.cpp. Generated text has not been evaluated for accuracy.",
        "text_sections": sections,
        "metric_groups": [{"title": "Generation", "items": [
            ("OUTPUT TOKENS", available(count)), ("WALL TIME", f"{seconds:.2f} s"),
            ("TOKENS / WALL SECOND", available(count / seconds if count is not None and seconds else None, 1))]}],
        "tables": [
            {"title": "Output details", "headers": ["SOURCE ROW", "INPUT TOKENS", "OUTPUT TOKENS", "WALL (S)", "STOP"],
             "rows": [[str(output.row_position) if column else "Prompt", available(output.input_tokens),
                       available(output.generated_tokens), _num(output.seconds, 2), output.finish_reason] for output in result.outputs]},
            {"title": "Server timing", "headers": ["OUTPUT", "PROMPT (S)", "DECODE (S)", "DECODE TOKENS/S"],
             "rows": [[str(i + 1), available(timing_seconds(output.timings.get("prompt_ms")), 3),
                       available(timing_seconds(output.timings.get("predicted_ms")), 3),
                       available(output.timings.get("predicted_per_second"), 1)] for i, output in enumerate(result.outputs)]},
            {"title": "Model and runtime", "headers": ["SETTING", "VALUE"], "rows": identity}],
        "warnings": result.warnings, "generation": result.to_dict(),
    }


REPORT_BUILDERS = {
    "llm": _llm_report,
    "model-comparison": _comparison_report,
    "linear-regression": _linear_regression_report,
    "logistic-regression": _logistic_regression_report,
    "decision-tree": _decision_tree_report,
    "random-forest": _random_forest_report,
    "svm": _svm_report,
    "neural-network": _neural_network_report,
    "kmeans": _kmeans_report,
    "pca": _pca_report,
    "association-scan": _association_scan_report,
}


# Recipes whose Task setting switches to a regression twin of the classifier.
REGRESSION_TWINS = {"decision-tree": "decision_tree_regression", "random-forest": "random_forest_regression",
                    "svm": "svm_regression", "neural-network": "neural_network_regression"}


def _wearable_report(dataset, req, name):
    from mlcookbook import evaluate_wearable

    result = evaluate_wearable(dataset, **req["params"])
    data = result.to_dict()
    selected = data["candidates"][data["selected_model"]]
    groups = {split: sum(value == split for value in result.group_splits.values())
              for split in ("train", "validation", "test")}
    metrics = [{"title": "Fixed validation groups", "items": [
        ("Macro F1", _num(selected["validation"]["macro_f1"])),
        ("Balanced accuracy", _num(selected["validation"]["balanced_accuracy"])),
        ("Accuracy", _num(selected["validation"]["accuracy"]))]}]
    if result.final_test_evaluated:
        metrics.append({"title": "Frozen final test groups", "items": [
            ("Macro F1", _num(selected["test"]["macro_f1"])),
            ("Balanced accuracy", _num(selected["test"]["balanced_accuracy"]))]})
    validation = selected["validation"]
    return {"dataset": name, "target": "Human movement label", "features": f"{len(result.features)} fixed window summaries; IDs and times excluded",
            "settings": f"Whole-{result.group_by} splits · seed {result.seed} · test {'evaluated' if result.final_test_evaluated else 'sealed'}",
            "summary": ("Synthetic integration fixture. " if any(w.startswith("Synthetic fixture") for w in result.warnings) else "") + f"Selected {result.selected_model.replace('_', ' ')} using macro F1 on separate validation groups. "
                       f"Groups: {groups['train']} training, {groups['validation']} validation, {groups['test']} test. "
                       "These classical baselines do not train the board CNN or establish fall-event performance.",
            "metric_groups": metrics,
            "tables": [{"title": "Validation comparison", "headers": ["Candidate", "Macro F1", "Balanced accuracy", "Group folds"],
                        "rows": [[key.replace('_', ' '), _num(value["validation"]["macro_f1"]),
                                  _num(value["validation"]["balanced_accuracy"]), str(len(value["grouped_cv"]))]
                                 for key, value in result.candidates.items()], "stretch_column": 0},
                       {"title": "Validation confusion matrix · actual rows / predicted columns",
                        "headers": ["Actual / predicted", *result.labels],
                        "rows": [[label, *map(str, row)] for label, row in zip(result.labels, validation["confusion_matrix"])],
                        "stretch_column": 0},
                       {"title": "Fixed group assignments", "headers": ["Group", "Split"],
                        "rows": [[key, value] for key, value in sorted(result.group_splits.items())], "stretch_column": 0}],
            "warnings": [*result.warnings, *result.limitations], "wearable_evaluation": data}


REPORT_BUILDERS["wearable-study"] = _wearable_report


def _screens():
    from mlcookbook import (
        association_scan,
        compare_regressors,
        decision_tree,
        evaluate_wearable,
        generate_text,
        kmeans,
        linear_regression,
        logistic_regression,
        neural_network,
        pca,
        random_forest,
        svm,
    )

    return {
        "wearable-study": (evaluate_wearable, WearableScreen),
        "llm": (generate_text, LLMScreen),
        "model-comparison": (compare_regressors, ModelComparisonScreen),
        "linear-regression": (linear_regression, LinearRegressionScreen),
        "logistic-regression": (logistic_regression, LogisticRegressionScreen),
        "decision-tree": (decision_tree, DecisionTreeScreen),
        "random-forest": (random_forest, RandomForestScreen),
        "svm": (svm, SVMScreen),
        "neural-network": (neural_network, NeuralNetworkScreen),
        "kmeans": (kmeans, KMeansScreen),
        "pca": (pca, PCAScreen),
        "association-scan": (association_scan, AssociationScanScreen),
    }


def _prepare_runtime():
    """Resolve the existing analysis functions in a worker, without running them."""
    _screens()


def _default_screen(key, demo=True, setup=False):
    from mlcookbook import Dataset

    func, screen_class = _screens()[key]
    engine = _engine_info(func)
    engine["description"] = next(recipe.description for recipe in RECIPES if recipe.key == key)
    if key == "model-comparison":
        engine["evaluation"] = "Shared training folds select a model before its test holdout is evaluated."
    elif key in ("linear-regression", "logistic-regression", *REGRESSION_TWINS):
        defaults = inspect.signature(func).parameters
        engine["evaluation"] = (f"Test holdout: {defaults['test_size'].default:.0%} · "
                                f"fixed split/model seed {defaults['random_state'].default}.\n"
                                "Cross-validation uses training rows only.")
    elif key in ("kmeans", "pca"):
        engine["evaluation"] = "Fitted on all selected rows; no test holdout or training cross-validation."
        if key == "kmeans":
            engine["evaluation"] += f" Fixed clustering seed {inspect.signature(func).parameters['random_state'].default}."
    elif key == "association-scan":
        engine["evaluation"] = "Each pair uses its complete rows. Adjusted p-values control the false discovery rate."
    elif key == "wearable-study":
        engine["evaluation"] = "Fixed training, validation and test groups. Final-test evaluation must be selected explicitly."
    else:
        engine["evaluation"] = ("Generated text is not scored for factual accuracy. Structured tasks check output fields; "
                                "reference agreement uses available labelled rows.")
    if key in REGRESSION_TWINS:
        engine["summary"] = (engine["summary"].replace("Fit a classification", "Fit a")
                             + f" The Regression task runs mlcookbook.{REGRESSION_TWINS[key]} instead.")
    screen = screen_class(engine, {})
    if setup:
        import pandas as pd

        # Empty typed-prompt storage is deliberately not a preset dataset.
        frame = pd.DataFrame({"prompt": pd.Series(dtype="string")}) if key == "llm" else pd.DataFrame()
        screen.dataset = Dataset(frame, overrides={"prompt": "text"} if key == "llm" else None)
        screen.dataset_name = "No dataset selected"
        screen.source = {"kind": "empty"}
        screen.set_dataset(_dataset_view(screen.dataset, screen.dataset_name))
        screen.set_status("Choose a local model and write a prompt." if key == "llm" else
                          "Open a movement study export." if key == "wearable-study" else
                          "")
        return screen
    make_frame, name, target, positive = DEMOS[key]
    dataset = Dataset(make_frame())
    screen.is_example = key not in ("llm", "wearable-study")
    screen.set_dataset(_dataset_view(dataset, name), target=target)
    if positive is not None:
        screen.positive.setCurrentText(positive)
    screen.dataset = dataset
    screen.dataset_name = name
    screen.source = {"kind": "demo", "recipe": key}
    if demo and getattr(screen, "auto_run_demo", True):
        screen.show_report(_run_report(key, dataset, name, screen.request()))
    screen.set_status("Open the export.json from the wearable pipeline; no recording is loaded automatically." if key == "wearable-study" else
                      "Choose a local model folder, write a prompt, and generate text." if key == "llm" else
                      "Built-in example. Open your own file or adjust the selections and run.")
    return screen


SCREEN_FACTORIES = {key: (lambda demo=True, setup=False, key=key: _default_screen(key, demo, setup)) for key in DEMOS}


def _dataset_view(dataset, name, import_report=None):
    import pandas as pd

    frame = dataset.df
    issues = dataset.describe().issues
    by_column: dict[str, list] = {}
    for issue in issues:
        by_column.setdefault(issue.column, []).append((issue.kind, issue.message))

    def classes(series):
        values = series.dropna().unique().tolist()
        if not 2 <= len(values) <= MAX_CLASSES:
            return []
        try:
            return sorted(values)
        except TypeError:
            return values

    return {
        "name": name,
        "rows": len(frame),
        "columns": [
            {"name": col.name, "type": col.var_type.value,
             "missing": int(frame[col.name].isna().sum()),
             "classes": classes(frame[col.name]),
             "issues": by_column.get(col.name, [])}
            for col in dataset.schema().columns
        ],
        "preview": [["" if pd.isna(value) else str(value) for value in row]
                    for row in frame.head(20).itertuples(index=False, name=None)],
        "notes": list(import_report.notes) if import_report is not None else [],
        "issues": [issue.message for issue in issues],
    }


def _gathering_source_lines(source):
    """Format recorded API provenance, without making another request."""
    gathered = source["gathering"]
    details = ["Source: " + gathered["title"] + " · " + gathered["source_url"],
               "Fetched (UTC): " + gathered["retrieved_at"],
               "Query: " + json.dumps(gathered["query"], ensure_ascii=False, sort_keys=True)]
    units = gathered.get("units", {})
    if units:
        details.append("Units: " + " · ".join(f"{column}: {unit}" for column, unit in units.items() if unit))
    if source.get("dataset_fingerprint"):
        details.append("Fetched snapshot · SHA-256 " + source["dataset_fingerprint"])
    details.append("The workspace holds fetched values; returning here does not contact the source.")
    return details


def _playground_query(query):
    """Credentials are transient and never identify a fetched dataset."""
    return {key: deepcopy(value) for key, value in query.items() if key != "api_key"}


def _run_report(key, dataset, name, request, *, stop_requested=None, on_progress=None):
    import pandas as pd

    from mlcookbook import Dataset
    from mlcookbook.experiments import runtime_versions

    # Overrides describe roles; they never convert strings into numbers silently.
    for column, role in request["overrides"].items():
        values = dataset.df[column]
        if role == "numeric" and not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"{column!r} contains non-numeric data. Clean the source column before choosing numeric.")
        if role == "boolean" and not pd.api.types.is_bool_dtype(values):
            raise ValueError(f"{column!r} is not stored as boolean. Use categorical for coded labels.")
        if role == "datetime" and not pd.api.types.is_datetime64_any_dtype(values):
            raise ValueError(f"{column!r} is not stored as dates. ISO-8601 dates (2024-05-31) are parsed on import.")
    configured = Dataset(dataset.df, overrides=request["overrides"])
    report = (REPORT_BUILDERS[key](configured, request, name, stop_requested=stop_requested, on_progress=on_progress) if key == "llm" else
              REPORT_BUILDERS[key](configured, request, name))
    report["provenance"] = {"versions": runtime_versions(include_llm=key == "llm" and request["params"].get("backend", "pytorch") == "pytorch"),
                            "created_utc": datetime.now(UTC).isoformat(),
                            "random_state": request.get("params", {}).get("random_state", 0)}
    report["run_details"] = [
        "Created (UTC): " + report["provenance"]["created_utc"],
        "Runtime: " + " · ".join(f"{package} {version}" for package, version in report["provenance"]["versions"].items()),
    ]
    return report


class _Job(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    progress = Signal(object)

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def run(self):
        try:
            self.succeeded.emit(self.operation())
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)


class MLCookbookApp:
    """Own navigation, imported data, and background work; GUI only renders data."""

    def __init__(self) -> None:
        self.window = MainWindow()
        self.imported = None
        self.job = None
        self._job_continuation = None
        # An embedded/test host may already have loaded the scientific stack.
        # Ordinary startup checks module names only and imports nothing here.
        self._runtime_initialized = all(name in sys.modules for name in (
            "mlcookbook.classification.neural_network", "mlcookbook.regression.neural_network", "mlcookbook.wearable"))
        self.screen = None
        self.key = "recipes"
        self.data_options = {}
        self.data_status = ""
        self.playground_specs = None
        self.playground_state = {}
        self.playground_preview = None
        self.playground_status = ""
        self.llm_lab_document = None
        self.llm_lab_mode = "generate"
        self.sessions: dict[str, dict] = {}
        self.workspace_sessions: dict[str, dict] = {}
        self.example_sessions: dict[str, dict] = {}
        self.kitchen = None
        self.kitchen_hashes, self.kitchen_views = {}, {}
        self.kitchen_history = ([], [])
        self.kitchen_output_ports = {}
        self.window.experiment_open_requested.connect(self.open_experiment)
        self.window.experiment_save_requested.connect(self.save_experiment)
        self.window.navigation_requested.connect(self.navigate)
        QApplication.instance().aboutToQuit.connect(self.wait_for_job)
        self.show_data_page()

    def navigate(self, key):
        if self.job is not None:
            return
        if key == "data":
            self.show_data_page()
        elif key == "playground":
            self.show_playground()
        elif key == "recipes":
            self.show_front_page()
        elif key == "kitchen":
            self.show_kitchen()
        elif key in ("llm", "wearable-study"):
            self.open_recipe(next(recipe for recipe in RECIPES if recipe.key == key))

    def _refresh_shell(self):
        busy = self.job is not None
        self.window.set_navigation(self.key, busy=busy)
        can_save = isinstance(self.screen, ReportScreen) and bool(self.screen.view["columns"])
        if isinstance(self.screen, LLMScreen) and self.screen.workspace_mode.currentData() == "research":
            can_save = False  # Research artifacts use their explicit, independent lab archive.
        self.window.set_experiment_actions(can_save=can_save, busy=busy)
        if isinstance(self.screen, ReportScreen):
            self.screen.set_example_actions(available=self.key not in ("llm", "wearable-study"),
                                            using_example=self.screen.is_example,
                                            has_workspace=self.imported is not None,
                                            has_example=self.key in self.example_sessions)
        elif isinstance(self.screen, KitchenScreen):
            self.screen.set_active_available(self.imported is not None)
        if self.imported is None:
            self.window.set_dataset_context("", "")
        else:
            _, name, view = self.imported
            self.window.set_dataset_context(name, f"{view['rows']} rows · {len(view['columns'])} columns")

    def _workspace_view(self):
        if self.imported is None:
            return None
        view = deepcopy(self.imported[2])
        source = view.get("source", {})
        details = []
        if view.get("snapshot_fingerprint"):
            details.append("Saved dataset snapshot · SHA-256 " + view["snapshot_fingerprint"])
        if source.get("kind") == "file":
            details.append("Source at import: " + source.get("path", view["name"]))
            if Path(source.get("path", "")).suffix.lower() == ".csv":
                delimiter = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}.get(source.get("delimiter"), "unspecified")
                details.append(f"CSV at import: {delimiter} delimiter · {source.get('encoding', 'encoding unavailable')} encoding")
                if source.get("missing_markers"):
                    details.append("Extra missing markers: " + ", ".join(source["missing_markers"]))
            details.append("The workspace holds the imported values; opening Data does not reread the file.")
        elif source.get("kind") == "api":
            details.extend(_gathering_source_lines(source))
        elif source.get("kind") == "kitchen_output":
            details.append(f"Kitchen output: {source['node_id']} · {source['port']}")
            details.append("Completed (UTC): " + source["completed_utc"])
            details.append("Artifact: " + source["artifact_id"])
            details.append("Configuration/input fingerprint: " + source["configuration_fingerprint"])
            details.append("This is a completed output snapshot; editing the Kitchen does not replace it.")
        elif source.get("run_id") or source.get("original_source"):
            details.append("Dataset with accepted LLM-generated columns. Review generated values and lineage in Local LLMs.")
            if source.get("run_id"):
                details.append("Source run: " + source["run_id"])
        elif not details:
            details.append("Dataset snapshot: " + view["name"])
        view["source_summary"] = "\n".join(details)
        return view

    def show_data_page(self):
        if self.job is not None:
            return
        self._remember_screen()
        page = DataScreen()
        page.restore_import_options(self.data_options)
        page.set_dataset(self._workspace_view())
        page.set_status(self.data_status)
        page.load_requested.connect(self.load_file)
        self.screen, self.key = page, "data"
        self.window.show_screen(page)
        self.window.set_status("")
        self._refresh_shell()

    def wait_for_job(self):
        if self.job is not None:
            self.job.wait()

    def show_playground(self):
        if self.job is not None:
            return
        if self.playground_specs is None:
            previous = (self.screen._status_message if isinstance(self.screen, (ReportScreen, KitchenScreen)) else
                        self.data_status if isinstance(self.screen, DataScreen) else "")
            prepared = []

            def prepare():
                _prepare_runtime()
                from mlcookbook.gathering import source_specs

                return source_specs()

            def accept(specs):
                self._runtime_initialized = True
                self.playground_specs = specs
                self._set_status(previous)
                prepared.append(True)

            def continued():
                if prepared:
                    self.show_playground()
                return False

            self._start_job(prepare, accept, "Preparing data sources…", continuation=continued)
            return
        self._remember_screen()
        page = DataPlaygroundScreen(self.playground_specs)
        page.set_download_directory(str(_playground_data_dir()))
        page.restore_session(self.playground_state)
        if self.playground_preview is not None:
            page.set_preview(self.playground_preview[2])
        page.set_status(self.playground_status)
        page.fetch_requested.connect(self.fetch_playground)
        page.use_requested.connect(self.accept_playground_dataset)
        page.download_requested.connect(self.download_playground_dataset)
        page.cancel_requested.connect(self.cancel_analysis)
        self.screen, self.key = page, "playground"
        self.window.show_screen(page)
        self.window.set_status("")
        self._refresh_shell()

    def fetch_playground(self, source, query):
        if self.job is not None or not isinstance(self.screen, DataPlaygroundScreen):
            return
        screen = self.screen
        query = deepcopy(query)
        requested = {"source": source, "query": _playground_query(query)}

        def fetch():
            from mlcookbook.core.fingerprint import dataset_fingerprint
            from mlcookbook.gathering import fetch_dataset

            worker = QThread.currentThread()
            gathered = fetch_dataset(source, query, cancelled=worker.isInterruptionRequested,
                                     progress=worker.progress.emit)
            if worker.isInterruptionRequested():
                raise ValueError("Fetch cancelled; previous preview kept.")
            metadata = gathered.result.to_dict()
            name = metadata["title"]
            view = _dataset_view(gathered.dataset, name)
            view["source"] = {"kind": "api", "gathering": metadata,
                              "dataset_fingerprint": dataset_fingerprint(gathered.dataset)}
            view["notes"] = list(metadata["warnings"])
            for column in view["columns"]:
                column["unit"] = metadata["units"].get(column["name"], "")
            view["provenance"] = _gathering_source_lines(view["source"])
            view["source_summary"] = "\n".join(view["provenance"])
            view["playground_request"] = requested
            return gathered.dataset, name, view

        def accept(loaded):
            self.playground_preview = loaded
            screen.set_preview(loaded[2])
            rows = loaded[2]["rows"]
            self._set_status(f"Fetched {rows} rows. Review the preview, then choose Use dataset." if rows else
                             "No rows matched this query. Adjust the filters and fetch again.")

        def progress(event):
            if isinstance(event, str):
                self._set_status(event)
            elif isinstance(event, dict) and event.get("message"):
                self._set_status(str(event["message"]))

        self._start_job(fetch, accept, "Fetching data…", cancellable=True, on_progress=progress)

    def download_playground_dataset(self, file_format):
        if self.job is not None or not isinstance(self.screen, DataPlaygroundScreen) or self.playground_preview is None:
            return
        screen = self.screen
        try:
            source, query = screen.request()
            requested = {"source": source, "query": _playground_query(query)}
            if screen.preview_stale or requested != self.playground_preview[2]["playground_request"]:
                raise ValueError("Query changed. Fetch a matching preview before downloading this dataset.")
            if not self.playground_preview[2]["rows"]:
                raise ValueError("This query returned no rows. Adjust the filters and fetch again.")
        except ValueError as exc:
            self._set_status(str(exc))
            return
        dataset, _, view = self.playground_preview
        metadata = deepcopy(view["source"])
        directory = _playground_data_dir()

        def write():
            from mlcookbook.gathering_io import save_gathered_dataset

            return save_gathered_dataset(dataset, metadata, directory, format=file_format)

        self._start_job(write, lambda path: self._set_status(f"Saved all {view['rows']} rows to {path}"),
                        "Saving fetched dataset…")

    def accept_playground_dataset(self):
        if self.job is not None or not isinstance(self.screen, DataPlaygroundScreen) or self.playground_preview is None:
            return
        screen = self.screen
        try:
            source, query = screen.request()
            requested = {"source": source, "query": _playground_query(query)}
            if screen.preview_stale or requested != self.playground_preview[2]["playground_request"]:
                raise ValueError("Query changed. Fetch a matching preview before using this dataset.")
            if not self.playground_preview[2]["rows"]:
                raise ValueError("This query returned no rows. Adjust the filters and fetch again.")
        except ValueError as exc:
            self._set_status(str(exc))
            return
        dataset, name, view = self.playground_preview

        def prepare():
            from mlcookbook.core.dataset import Dataset

            copied = Dataset(dataset.df, overrides={column.name: column.var_type for column in dataset.schema().columns})
            return copied, name, deepcopy(view)

        def accept(loaded):
            self._use_import(loaded)
            self.data_status = f"{name} is active. Choose a recipe to analyze the fetched snapshot."
            self._set_status(self.data_status)

        self._start_job(prepare, accept, "Preparing fetched dataset…")

    def show_kitchen(self, *, remember=True):
        if self.job is not None:
            return
        if not self._runtime_initialized:
            previous = (self.screen._status_message if isinstance(self.screen, (ReportScreen, KitchenScreen)) else
                        self.data_status if isinstance(self.screen, DataScreen) else "")

            def accept(_):
                self._runtime_initialized = True
                self._set_status(previous)

            def continued():
                if self._runtime_initialized:
                    self.show_kitchen()
                return False

            self._start_job(_prepare_runtime, accept, "Preparing Kitchen runtime…", continuation=continued)
            return
        from mlcookbook.kitchen import PORTS, KitchenDocument, default_settings

        if remember:
            self._remember_screen()
        if self.kitchen is None:
            self.kitchen = KitchenDocument()
        screen = KitchenScreen({kind: default_settings(kind) for kind in PORTS})
        screen.set_editor({"graph": self.kitchen.graph.to_dict(), "layout": self.kitchen.layout, "selected": self.kitchen.selected})
        screen._undo, screen._redo = deepcopy(self.kitchen_history)
        screen._update_actions()
        self._connect_kitchen(screen)
        self.screen, self.key = screen, "kitchen"
        self.window.show_screen(screen)
        self.window.set_status("")
        self._refresh_kitchen()
        self._refresh_shell()
        QTimer.singleShot(0, screen.canvas, screen.canvas.fit_graph)

    def _connect_kitchen(self, screen):
        screen.changed.connect(self._kitchen_changed)
        screen.selection_changed.connect(self._kitchen_selected)
        screen.capture_requested.connect(self.capture_kitchen_snapshot)
        screen.run_requested.connect(self.run_kitchen)
        screen.cancel_requested.connect(self.cancel_analysis)
        screen.save_requested.connect(self.save_kitchen)
        screen.open_requested.connect(self.open_kitchen)
        screen.export_requested.connect(self.export_kitchen_output)
        screen.accept_requested.connect(self.accept_kitchen_dataset)
        screen.inspect_requested.connect(self.inspect_kitchen_output)

    def _kitchen_changed(self, state):
        from mlcookbook.kitchen import graph_from_dict

        graph = graph_from_dict(state["graph"])
        self.kitchen.graph, self.kitchen.layout, self.kitchen.selected = graph, deepcopy(state["layout"]), state["selected"]
        self._refresh_kitchen()
        if self.job is None:
            self.screen.set_status("")

    def _kitchen_selected(self, key):
        self.kitchen.selected = key
        self._refresh_kitchen()

    @staticmethod
    def _kitchen_columns(document, views, key):
        from functools import cache

        nodes = {n.id: n for n in document.graph.nodes}
        parents = {(e.target, e.target_port): (e.source, e.source_port) for e in document.graph.edges}

        def input_meta(identifier, port):
            source = parents.get((identifier, port))
            return output(*source) if source else {}

        @cache
        def output(identifier, port):
            node = nodes.get(identifier)
            if node is None:
                return {}
            s = node.settings
            if node.kind == "dataset":
                return {"columns": deepcopy(views.get(s["snapshot"], {}).get("columns", []))}
            if node.kind == "select":
                available = {c["name"]: c for c in input_meta(identifier, "data").get("columns", [])}
                return {"columns": [dict(available[name], type=s["roles"].get(name, available[name]["type"]))
                                    for name in s["columns"] if name in available]}
            if node.kind == "column":
                col = next((c for c in input_meta(identifier, "data").get("columns", []) if c["name"] == s["column"]), None)
                return {"name": (s["column"] + " missing") if port == "missing" else s["column"],
                        "role": "boolean" if port == "missing" else col["type"] if col else "unknown"}
            if node.kind in ("map", "combine"):
                first = input_meta(identifier, "vector" if node.kind == "map" else "left")
                return {"name": f"{s['operation']}({first.get('name', 'vector')})",
                        "role": "boolean" if s["operation"] in ("greater_than", "less_than", "equal") else "numeric"}
            if node.kind == "build":
                return {"columns": [{"name": s["name"], "type": input_meta(identifier, "vector").get("role", "unsupported")}]}
            if node.kind == "add_column":
                columns = list(input_meta(identifier, "data").get("columns", []))
                if s["name"] not in {c["name"] for c in columns}:
                    columns.append({"name": s["name"], "type": input_meta(identifier, "vector").get("role", "unsupported")})
                return {"columns": columns}
            if node.kind == "filter":
                return input_meta(identifier, "data")
            return {}

        node = nodes.get(key)
        if node is None:
            return {}
        snapshot = node.settings.get("snapshot")
        info = document.snapshot_info.get(snapshot, {})
        view = views.get(snapshot, {})
        from mlcookbook.kitchen import PORTS
        from mlcookbook.kitchen_artifacts import TableArtifact

        record = document.run.records.get(key)
        completed = []
        if record:
            for port, identifier in record.payload.get("outputs", {}).items():
                artifact = document.artifacts.get(identifier)
                if artifact is not None:
                    completed.append({"port": port, "kind": "dataset" if isinstance(artifact, TableArtifact) else "vector"})
            if record.kind in ("comparison", "linear", "summary", "correlation", "report"):
                completed.append({"port": "result", "kind": "analysis"})
        vector_inputs = {port: input_meta(key, port) for port, kind in PORTS[node.kind][0].items() if kind == "vector"}
        return {"columns": input_meta(key, "data").get("columns", []), "vectors": vector_inputs, "completed_outputs": completed,
                "error": document.run.attempts.get(key, {}).get("error", ""),
                "identity": f"{info['name']} · {view['rows']} rows · {len(view['columns'])} columns"
                            if info and view else "No snapshot selected"}

    @staticmethod
    def _kitchen_report(document, key, port=None):
        record = document.run.records.get(key)
        if record is None:
            return {}
        payload = record.payload
        name = document.snapshot_info.get(payload.get("snapshot_id"), {}).get("name", "Kitchen snapshot")
        kind = payload.get("analysis_kind", "comparison" if record.kind == "report" else record.kind)
        if kind == "comparison":
            report = _comparison_data_report(payload, name, payload["input_rows"])
        elif kind == "linear":
            report = _kitchen_linear_report(payload, name)
        elif kind == "summary":
            report = {"dataset": name, "features": payload["name"], "summary": f"{payload['n_obs']} observed · "
                      f"{payload['n_missing']} missing · {payload['n_unique']} distinct values.",
                      "metric_groups": [{"title": "Descriptive statistics", "items": [
                          (label.replace("_", " ").capitalize(), _num(value)) for label, value in payload["statistics"].items()]}]
                      if payload["statistics"] else [],
                      "tables": [{"title": "Observed value counts · up to 50 levels", "headers": ["Value", "Count"],
                                  "rows": [[value, str(count)] for value, count in payload["value_counts"]]}],
                      "warnings": list(record.warnings)}
        elif kind == "correlation":
            report = {"dataset": name, "features": payload["left_name"] + ", " + payload["right_name"],
                      "summary": f"{payload['method'].capitalize()} correlation: {_num(payload['statistic'])} "
                                 f"across {payload['n_obs']} complete pairs.",
                      "plots": [{"title": "Paired observations", "x_label": payload["left_name"], "y_label": payload["right_name"],
                                 "series": [{"kind": "points", "color": "accent", "label": "pairs", "points": payload["points"]}]}],
                      "metric_groups": [{"title": "Association", "items": [("Coefficient", _num(payload["statistic"])),
                                            ("p-value", _p(payload["p_value"])), ("CI low", _num(payload["ci_low"])),
                                            ("CI high", _num(payload["ci_high"]))]}], "warnings": list(record.warnings)}
        elif "vectors" in payload:
            tables = []
            shown = {name: meta for name, meta in payload["vectors"].items() if port is None or name == port}
            for output_name, meta in shown.items():
                tables.append({"title": f"{output_name} · {meta['name']} · {meta['role']} · {meta['rows']} rows",
                               "headers": ["Preview row", "Value"], "rows": [[str(i), value] for i, value in enumerate(meta["preview"])]})
            report = {"dataset": name, "features": ", ".join(v["name"] for v in shown.values()),
                      "tables": tables, "warnings": list(record.warnings)}
        else:
            report = {"dataset": name, "features": ", ".join(c["name"] for c in payload["columns"]),
                      "summary": f"{payload['rows']} rows · {len(payload['columns'])} columns.",
                      "tables": [{"title": "Columns and roles", "headers": ["Column", "Role", "Dtype"],
                                  "rows": [[c["name"], c["type"], c["dtype"]] for c in payload["columns"]]},
                                 {"title": "Preview · first 20 rows", "headers": [c["name"] for c in payload["columns"]],
                                  "rows": payload["preview"]}], "warnings": list(record.warnings)}
        report["run_details"] = [f"Node: {record.node_id} · {record.kind}", "Completed (UTC): " + record.completed_utc,
                                 "Configuration/input fingerprint: " + record.fingerprint,
                                 "Recorded settings: " + json.dumps(record.settings, ensure_ascii=False),
                                 "Completed input ports: " + json.dumps(record.inputs, ensure_ascii=False),
                                 "Runtime: " + " · ".join(f"{k} {v}" for k, v in record.versions.items())]
        report["warnings"] = list(dict.fromkeys([*report.get("warnings", []), *document.warnings, *document.run.warnings]))
        error = document.run.attempts.get(key, {}).get("error")
        if error:
            report["warnings"].append("Last attempt: " + error)
        return report

    def _refresh_kitchen(self):
        if not isinstance(self.screen, KitchenScreen):
            return
        from mlcookbook.kitchen import graph_fingerprints, node_states

        states = node_states(self.kitchen.graph, self.kitchen_hashes, self.kitchen.run)
        self.screen.set_states(states)
        key = self.kitchen.selected
        context = self._kitchen_columns(self.kitchen, self.kitchen_views, key)
        record = self.kitchen.run.records.get(key)
        context["current_output"] = bool(record and graph_fingerprints(self.kitchen.graph, self.kitchen_hashes).get(key) == record.fingerprint)
        ports = [item["port"] for item in context.get("completed_outputs", [])]
        selected_port = self.kitchen_output_ports.get(key)
        if selected_port not in ports:
            selected_port = ports[0] if ports else None
        context["selected_output"] = selected_port
        context["node_captions"] = {node.id: self.kitchen.snapshot_info.get(node.settings.get("snapshot"), {}).get("name", "")
                                    for node in self.kitchen.graph.nodes if node.kind == "dataset"}
        self.screen.set_context(context, report=self._kitchen_report(self.kitchen, key, selected_port), state=states.get(key, "unrun"))

    def inspect_kitchen_output(self, node_id, port):
        if not isinstance(self.screen, KitchenScreen) or node_id != self.kitchen.selected:
            return
        self.kitchen_output_ports[node_id] = port
        self._refresh_kitchen()

    def _completed_kitchen_output(self, node_id, port):
        if self.kitchen is None or node_id not in self.kitchen.run.records:
            raise ValueError("Run this node before using its output.")
        record = self.kitchen.run.records[node_id]
        if port == "result" and record.kind in ("comparison", "linear", "summary", "correlation", "report"):
            return record, None
        identifier = record.payload.get("outputs", {}).get(port)
        artifact = self.kitchen.artifacts.get(identifier)
        if artifact is None:
            raise ValueError("This saved output contains a preview. Run its path to produce a complete output.")
        return record, artifact

    def export_kitchen_output(self, node_id, port, path):
        if self.job is not None or not isinstance(self.screen, KitchenScreen):
            return
        try:
            record, artifact = self._completed_kitchen_output(node_id, port)
        except ValueError as exc:
            self._set_status(str(exc))
            return
        from mlcookbook.kitchen import graph_fingerprints

        matches = graph_fingerprints(self.kitchen.graph, self.kitchen_hashes).get(node_id) == record.fingerprint

        def write():
            import os
            import tempfile

            import pandas as pd

            from mlcookbook.kitchen_artifacts import TableArtifact

            destination = Path(path)
            suffix = destination.suffix.lower()
            if artifact is None and suffix != ".json":
                raise ValueError("Export an analysis result as JSON.")
            if artifact is not None and suffix not in (".csv", ".parquet"):
                raise ValueError("Export values as CSV or Parquet.")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                                 suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                if artifact is None:
                    payload = {"format": "mlcookbook-kitchen-output", "port": port,
                               "current_settings_match": matches, "completed": record.to_dict()}
                    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
                else:
                    frame = artifact.dataset.df if isinstance(artifact, TableArtifact) else pd.DataFrame(
                        {artifact.name: artifact.values.array}, index=artifact.index)
                    if suffix == ".csv":
                        frame.to_csv(temporary, index=False)
                    else:
                        frame.to_parquet(temporary, index=True, engine="pyarrow")
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

        state = "" if matches else " · last completed output; current settings differ"
        self._start_job(write, lambda _: self._set_status(f"Exported {Path(path).name}{state}."), "Exporting completed output…")

    def accept_kitchen_dataset(self, node_id, port):
        if self.job is not None or not isinstance(self.screen, KitchenScreen):
            return
        from mlcookbook.kitchen import graph_fingerprints
        from mlcookbook.kitchen_artifacts import TableArtifact

        try:
            record, artifact = self._completed_kitchen_output(node_id, port)
            if not isinstance(artifact, TableArtifact):
                raise ValueError("Choose a completed dataset output.")
            if graph_fingerprints(self.kitchen.graph, self.kitchen_hashes).get(node_id) != record.fingerprint:
                raise ValueError("This output uses older settings. Run its path before using it as active data.")
        except ValueError as exc:
            self._set_status(str(exc))
            return

        def prepare():
            from mlcookbook.core.dataset import Dataset

            dataset = Dataset(artifact.dataset.df, overrides={c.name: c.var_type for c in artifact.dataset.schema().columns})
            name = f"Kitchen · {node_id}"
            view = _dataset_view(dataset, name)
            view["source"] = {"kind": "kitchen_output", "node_id": node_id, "port": port, "artifact_id": artifact.id,
                              "completed_utc": record.completed_utc, "configuration_fingerprint": record.fingerprint,
                              "settings": deepcopy(record.settings), "source_snapshot_ids": record.payload.get("source_snapshot_ids", [])}
            view["notes"] = list(dict.fromkeys([*view.get("notes", []), *artifact.warnings, *record.warnings]))
            return dataset, name, view

        def accept(loaded):
            self._use_import(loaded)
            self.data_status = "Completed Kitchen output is active."
            self._set_status(self.data_status)

        self._start_job(prepare, accept, "Preparing completed dataset…")

    def capture_kitchen_snapshot(self, node_id):
        if self.imported is None or self.job is not None or not isinstance(self.screen, KitchenScreen):
            return
        dataset, name, view = self.imported

        def copy_snapshot():
            from uuid import uuid4

            from mlcookbook.core.dataset import Dataset
            from mlcookbook.core.fingerprint import dataset_fingerprint

            copied = Dataset(dataset.df, overrides={c.name: c.var_type for c in dataset.schema().columns})
            return uuid4().hex, copied, dataset_fingerprint(copied), _dataset_view(copied, name)

        def accept(value):
            key, copied, fingerprint, copied_view = value
            self.kitchen.snapshots[key] = copied
            self.kitchen.snapshot_info[key] = {"name": name, "source": deepcopy(view.get("source", {}))}
            self.kitchen_hashes[key], self.kitchen_views[key] = fingerprint, copied_view
            self.screen.install_snapshot(node_id, key)
            self._set_status("Dataset snapshot captured. Changing active data later will not replace it.")

        self._start_job(copy_snapshot, accept, "Capturing dataset snapshot…")

    def run_kitchen(self, to_node=None):
        if self.job is not None or not isinstance(self.screen, KitchenScreen):
            return
        from mlcookbook.kitchen import run_graph

        document = self.kitchen

        def execute():
            return run_graph(document.graph, document.snapshots, previous=document.run, to_node=to_node,
                             stop_requested=self.job.isInterruptionRequested, on_progress=self.job.progress.emit)

        def progress(event):
            states = dict(self.screen._states)
            states[event["node"]] = event["state"]
            self.screen.set_states(states)
            self.screen.result_state.setText(f"{event['node']} · {event['state']}" +
                                            (" · previous completed output retained" if document.selected in document.run.records else ""))
            self.screen.result_state.show()
            self._set_status(f"{event['node']} · {event['state']}" + (": " + event["error"] if event["error"] else ""))

        def accept(execution):
            document.accept_execution(execution)
            self._refresh_kitchen()
            self.screen.inspector.setCurrentIndex(1)
            failed = [f"{key}: {attempt['error']}" for key, attempt in execution.run.attempts.items()
                      if key in execution.requested_nodes and attempt["state"] == "failed"]
            self._set_status("; ".join(failed) if failed else "Cancelled between operations; completed results kept."
                             if execution.run.warnings else "Workflow complete.")

        self._start_job(execute, accept, "Validating workflow…", cancellable=True, on_progress=progress)

    def save_kitchen(self, path):
        if self.job is not None or not isinstance(self.screen, KitchenScreen):
            return
        from mlcookbook.kitchen import KitchenDocument, KitchenRun
        from mlcookbook.kitchen_io import save_kitchen

        self._remember_screen()
        doc = self.kitchen
        ids = {n.id for n in doc.graph.nodes}
        run = KitchenRun({k: v for k, v in doc.run.records.items() if k in ids},
                         {k: v for k, v in doc.run.attempts.items() if k in ids},
                         [k for k in doc.run.execution_order if k in ids], doc.run.warnings)
        saved = KitchenDocument(doc.graph, doc.snapshots, doc.snapshot_info, run, doc.layout, doc.selected, doc.warnings, doc.artifacts)
        self._start_job(lambda: save_kitchen(path, saved), lambda _: self._set_status(f"Saved {Path(path).name}."), "Saving Kitchen…")

    def open_kitchen(self, path):
        if self.job is not None:
            return

        def read():
            _prepare_runtime()
            from mlcookbook.core.fingerprint import dataset_fingerprint
            from mlcookbook.kitchen_io import load_kitchen

            doc = load_kitchen(path)
            KitchenScreen.validate_editor({"graph": doc.graph.to_dict()})
            # Validate every renderer input before replacing the current document.
            for key in doc.run.records:
                self._kitchen_report(doc, key)
            hashes = {key: dataset_fingerprint(data) for key, data in doc.snapshots.items()}
            views = {key: _dataset_view(data, doc.snapshot_info[key]["name"]) for key, data in doc.snapshots.items()}
            return doc, hashes, views

        opened = []

        def accept(value):
            self._runtime_initialized = True
            self._remember_screen()
            self.kitchen, self.kitchen_hashes, self.kitchen_views = value
            self.kitchen_history = ([], [])
            self.kitchen_output_ports = {}
            opened.append(True)

        def continued():
            if opened:
                # The worker has retired; do not overwrite the new graph with the old screen.
                self.show_kitchen(remember=False)
                self._set_status(f"Opened {Path(path).name}; results restored without execution.")
            return False

        self._start_job(read, accept, "Opening Kitchen…", continuation=continued)

    def show_front_page(self) -> None:
        if self.job is not None:
            return
        self._remember_screen()
        self.screen = None
        self.key = "recipes"
        page = FrontPage()
        page.recipe_selected.connect(self.open_recipe)
        self.window.show_screen(page)
        self.window.set_status("")
        self._refresh_shell()

    def open_recipe(self, recipe: Recipe, *, remember=True, on_ready=None) -> None:
        if self.job is not None:
            return
        factory = SCREEN_FACTORIES.get(recipe.key)
        if factory is None:
            return
        if not self._runtime_initialized:
            prepared = []
            previous_status = (self.screen._status_message if isinstance(self.screen, ReportScreen) else
                               self.data_status if isinstance(self.screen, DataScreen) else
                               self.window.statusBar().currentMessage())

            def accept(_):
                self._runtime_initialized = True
                self._set_status(previous_status)
                prepared.append(True)

            def continue_open():
                if prepared:
                    self.open_recipe(recipe, remember=remember, on_ready=on_ready)
                return self.job is not None

            self._start_job(_prepare_runtime, accept, "Preparing analysis runtime…", continuation=continue_open)
            return
        if remember:
            self._remember_screen()
        saved = self.sessions.get(recipe.key)
        screen = factory(demo=False, setup=True)
        self.screen = screen
        self.key = recipe.key
        if saved is not None:
            screen.dataset = saved["dataset"]
            screen.dataset_name = saved["name"]
            screen.set_dataset(saved["view"])
            screen.source = saved["source"]
        elif self.imported is not None and recipe.key != "wearable-study":
            self._display_import(self.imported)
        self._connect_screen(screen)
        self.window.show_screen(screen)
        if saved is not None:
            screen.restore_session(saved["state"])
        self._update_input_context()
        self._refresh_shell()
        if on_ready is not None:
            on_ready()

    def run_example(self):
        """Restore a cached example, or load/run its first result before switching."""
        if self.job is not None or not isinstance(self.screen, ReportScreen) or self.key in ("llm", "wearable-study"):
            return
        self._remember_screen()
        key = self.key
        saved = self.example_sessions.get(key)
        if saved is not None and not self.screen.is_example:
            # Restore even invalid/unrun edits so they can be repaired. Never
            # make access to the cached example depend on a successful rerun.
            self.sessions[key] = saved
            self.open_recipe(next(recipe for recipe in RECIPES if recipe.key == key), remember=False)
            return
        loaded = {}

        def prepare(value):
            loaded["value"] = value

        def start_analysis():
            if "value" not in loaded:
                return False
            dataset, name, view, target, positive = loaded["value"]
            candidate = SCREEN_FACTORIES[key](demo=False, setup=True)
            try:
                candidate.dataset, candidate.dataset_name = dataset, name
                candidate.source, candidate.is_example = {"kind": "demo", "recipe": key}, True
                candidate.set_dataset(view, target=target)
                if saved is not None:
                    candidate.restore_session(saved["state"])
                elif positive is not None:
                    candidate.positive.setCurrentText(positive)
                request = candidate.request()
            except Exception:
                candidate.deleteLater()
                raise

            def accept(report):
                candidate.show_report(report, request=request)
                candidate.set_status("Analysis complete.")
                self.screen = candidate
                self._connect_screen(candidate)
                self.window.show_screen(candidate)
                self._update_input_context()

            def cleanup():
                if self.screen is not candidate:
                    candidate.deleteLater()
                return False

            self._start_job(lambda: _run_report(key, dataset, name, request), accept,
                            "Running built-in example…", continuation=cleanup)
            return True

        def load():
            if saved is not None:
                return saved["dataset"], saved["name"], saved["view"], None, None
            from mlcookbook import Dataset

            make_frame, name, target, positive = DEMOS[key]
            dataset = Dataset(make_frame())
            return dataset, name, _dataset_view(dataset, name), target, positive

        self._start_job(load, prepare, "Loading built-in example…", continuation=start_analysis)

    def use_workspace_data(self):
        if self.job is not None or not isinstance(self.screen, ReportScreen) or not self.screen.is_example:
            return
        self._remember_screen()
        saved = self.workspace_sessions.get(self.key)
        if saved is not None:
            self.sessions[self.key] = saved
        else:
            self.sessions.pop(self.key, None)
        self.open_recipe(next(recipe for recipe in RECIPES if recipe.key == self.key), remember=False)

    def _connect_screen(self, screen):
        screen.back_requested.connect(self.show_front_page)
        screen.load_requested.connect(self.load_file)
        screen.run_requested.connect(self.run_analysis)
        screen.cancel_requested.connect(self.cancel_analysis)
        screen.configuration_changed.connect(self._update_input_context)
        screen.example_requested.connect(self.run_example)
        screen.workspace_requested.connect(self.use_workspace_data)
        if isinstance(screen, WearableScreen):
            screen.study_requested.connect(self.load_wearable_study)
            screen.evaluation_export_requested.connect(self.export_wearable_evaluation)
        if isinstance(screen, LLMScreen):
            screen.connection_requested.connect(self.connect_local_llm)
            screen.workflow_requested.connect(self.workflow_action)
            screen.lab_requested.connect(self.llm_lab_action)
            screen.restore_research(self.llm_lab_document, self.llm_lab_mode)
            screen.lab_changed.connect(self._refresh_shell)
        self._refresh_shell()
        self.window.set_status("")

    def _update_input_context(self):
        screen = self.screen
        if not isinstance(screen, ReportScreen):
            return
        params = screen.request()["params"] if isinstance(screen, LLMScreen) else {}
        typed_prompt = params.get("workflow") == "generate" and params.get("input_mode") == "prompt"
        if typed_prompt:
            text = "Input: typed prompt"
        elif isinstance(screen, WearableScreen):
            text = ("Input: Movement Study · " + screen.dataset_name + " · fixed group holdouts"
                    if screen.source.get("kind") == "wearable_study" else "Input: no Movement Study selected")
        elif screen.is_example:
            text = "Input: built-in example · " + screen.dataset_name
        elif screen.source.get("kind") == "empty":
            text = "Input: no dataset selected"
        else:
            prefix = "Input: saved dataset snapshot · " if screen.view.get("snapshot_fingerprint") else "Input: "
            text = prefix + screen.dataset_name
        screen.set_input_context(text, include_shape=not typed_prompt)

    def _remember_screen(self):
        if isinstance(self.screen, DataScreen):
            self.data_options = self.screen.import_options()
        elif isinstance(self.screen, DataPlaygroundScreen):
            self.playground_state = self.screen.session_state()
            self.playground_status = self.screen._status_message
        elif isinstance(self.screen, KitchenScreen):
            state = self.screen.state()
            self.kitchen.layout, self.kitchen.selected = state["layout"], state["selected"]
            self.kitchen_history = deepcopy((self.screen._undo, self.screen._redo))
        elif isinstance(self.screen, ReportScreen):
            if isinstance(self.screen, LLMScreen):
                self.llm_lab_document = self.screen.research.lab_document()
                self.llm_lab_mode = self.screen.workspace_mode.currentData()
            snapshot = {
                "dataset": self.screen.dataset, "name": self.screen.dataset_name,
                "view": self.screen.view, "state": self.screen.session_state(),
                "source": deepcopy(self.screen.source),
            }
            self.sessions[self.key] = snapshot
            (self.example_sessions if self.screen.is_example else self.workspace_sessions)[self.key] = snapshot

    def _clear_sessions(self):
        self.sessions.clear()
        self.workspace_sessions.clear()
        self.example_sessions.clear()

    def _start_job(self, operation, accept, message, *, cancellable=False, on_progress=None, continuation=None):
        if self.job is not None:
            return
        screen = self.screen
        if screen is not None:
            screen.set_busy(True)
            if isinstance(screen, (ReportScreen, KitchenScreen, DataPlaygroundScreen)):
                screen.set_cancellable(cancellable)
        else:
            self.window._screen.setEnabled(False)
        self._set_status(message)
        self.window.set_experiment_actions(can_save=isinstance(screen, ReportScreen), busy=True)
        self.window.set_navigation(self.key, busy=True)

        def accepted(value):
            try:
                accept(value)
            except Exception as exc:
                self._set_status("Unable to complete: " + (str(exc) or type(exc).__name__))

        self.job = _Job(operation)
        self._job_continuation = continuation
        self.job.succeeded.connect(accepted)
        self.job.failed.connect(lambda error: self._set_status("Unable to complete: " + error))
        if on_progress is not None:
            self.job.progress.connect(on_progress)
        self.job.finished.connect(self._job_finished)
        self.job.start()

    def _job_finished(self):
        self.job.deleteLater()
        self.job = None
        continuation, self._job_continuation = self._job_continuation, None
        if continuation is not None:
            try:
                if continuation():
                    return
            except Exception as exc:
                self._set_status("Unable to complete: " + (str(exc) or type(exc).__name__))
        if self.screen is not None:
            self.screen.set_busy(False)
        self.window._screen.setEnabled(True)
        self._update_input_context()
        self._refresh_shell()

    def _set_status(self, message):
        if self.screen is not None:
            self.screen.set_status(message)
            if isinstance(self.screen, DataScreen):
                self.data_status = message
            elif isinstance(self.screen, DataPlaygroundScreen):
                self.playground_status = message
        else:
            self.window.set_status(message)

    def save_experiment(self, path):
        from mlcookbook.experiments import save_experiment

        if not isinstance(self.screen, ReportScreen) or not self.screen.view["columns"] or self.job is not None:
            return
        screen = self.screen
        state = screen.session_state()
        try:
            screen.validate_session(state)
        except ValueError as exc:
            self._set_status("Unable to save experiment: " + str(exc))
            return
        source = deepcopy(screen.source)
        source.setdefault("import_notes", list(screen.view.get("notes", [])))
        self._start_job(lambda: save_experiment(path, screen.dataset, recipe=self.key,
                                               name=screen.dataset_name, state=state, source=source),
                        lambda _: self._set_status(f"Saved {Path(path).name} · includes dataset, settings, and last report."),
                        "Saving experiment…")

    def open_experiment(self, path):
        previous_status = (self.screen._status_message if isinstance(self.screen, ReportScreen) else
                           self.data_status if isinstance(self.screen, DataScreen) else
                           self.window.statusBar().currentMessage())

        def read():
            _prepare_runtime()
            from mlcookbook.experiments import load_experiment

            loaded = load_experiment(path)
            view = _dataset_view(loaded.dataset, loaded.name)
            view["source"] = deepcopy(loaded.source)
            view["notes"] = list(loaded.source.get("import_notes", []))
            view["notes"] += loaded.warnings
            view["notes"].append(f"Saved dataset snapshot · SHA-256 {loaded.fingerprint}")
            view["snapshot_fingerprint"] = loaded.fingerprint
            return loaded, view

        def accept(opened):
            self._runtime_initialized = True
            # An independent study preserves the ordinary workspace/session. Do
            # not snapshot the transient "Opening experiment" job status with it.
            self._set_status(previous_status)
            self._use_experiment(opened)

        self._start_job(read, accept, "Opening experiment…")

    def _use_experiment(self, opened):
        loaded, view = opened
        factory = SCREEN_FACTORIES.get(loaded.recipe)
        if factory is None:
            raise ValueError(f"Unknown experiment recipe: {loaded.recipe!r}.")
        screen = factory(demo=False, setup=True)
        try:
            screen.dataset, screen.dataset_name = loaded.dataset, loaded.name
            screen.source = loaded.source
            screen.set_dataset(view)
            state = screen.prepare_session(loaded.state)
            screen.validate_session(state)
            if loaded.recipe == "llm":
                from mlcookbook.core.dataset import Dataset
                from mlcookbook.llm_workflow import check_source

                runs = list(state["report"].get("workflow_history", []))
                if state["report"].get("workflow_run"):
                    runs.append(state["report"]["workflow_run"])
                for run in runs:
                    # Role overrides belong to the original run, not to unrun edits.
                    roles = state["report_request"]["overrides"]
                    check_source(Dataset(loaded.dataset.df, overrides=roles), run)
            if loaded.recipe == "wearable-study" and loaded.source.get("kind") == "wearable_study":
                from mlcookbook.wearable import check_wearable_study

                check_wearable_study(loaded.dataset)
            screen.restore_session(state)
            if screen.request() != state["request"]:
                raise ValueError("Saved selections cannot be restored exactly with this recipe.")
        except Exception:
            screen.deleteLater()
            raise
        # Commit navigation only after the entire snapshot has been read and validated.
        self._remember_screen()
        is_example = loaded.recipe not in ("llm", "wearable-study") and state["is_example"]
        independent = loaded.recipe == "wearable-study" or is_example or (
            loaded.recipe == "llm" and loaded.source.get("kind") in ("empty", "demo"))
        if independent:
            self.sessions.pop(loaded.recipe, None)
        else:
            self._clear_sessions()
            self.imported = loaded.dataset, loaded.name, view
            self.data_status = f"Opened {loaded.name} from its saved dataset snapshot."
        self.screen, self.key = screen, loaded.recipe
        if loaded.recipe == "llm":
            self.llm_lab_mode = "generate"
        if is_example:
            self.example_sessions[self.key] = {"dataset": loaded.dataset, "name": loaded.name,
                "view": view, "state": state, "source": deepcopy(loaded.source)}
        self._connect_screen(screen)
        self.window.show_screen(screen)
        screen.restore_session(state)  # scroll positions require the attached layout
        message = f"Opened {loaded.name} from its saved snapshot. Results restored without retraining."
        if loaded.warnings:
            message += " Package versions differ; see Data checks before rerunning."
        screen.set_status(message)
        self.window.set_status("")
        self._update_input_context()
        self._refresh_shell()

    def load_wearable_study(self, path):
        if self.job is not None or not isinstance(self.screen, WearableScreen):
            return
        def read():
            from mlcookbook import load_wearable_study

            dataset, imported = load_wearable_study(path)
            name = Path(path).parent.name if Path(path).is_file() else Path(path).name
            view = _dataset_view(dataset, name)
            view["notes"] = [*imported.warnings, "Use Movement Study for grouped scores. Ordinary recipe row splits are exploratory."]
            view["source"] = {"kind": "wearable_study", "path": str(Path(path).resolve()),
                              "wearable_import": imported.to_dict()}
            return dataset, name, view

        def accept(loaded):
            self.sessions.pop("wearable-study", None)
            self._display_import(loaded)
            self.screen.set_status("Study verified. Fixed groups and human labels retained; final test remains sealed.")

        self._start_job(read, accept, "Verifying movement study and fixed groups…")

    def export_wearable_evaluation(self, path):
        import json
        import os
        import tempfile

        if self.job is not None or not isinstance(self.screen, WearableScreen):
            return
        value = deepcopy(self.screen._report.get("wearable_evaluation"))
        if value is None:
            return

        def write():
            destination = Path(path)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                                  prefix=".movement-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

        self._start_job(write, lambda _: self._set_status("Evaluation exported with predictions, probabilities and source-window lineage."),
                        "Exporting movement evaluation…")

    def load_file(self, path, delimiter, encoding, markers=""):
        missing_markers = [m.strip() for m in markers.split(",") if m.strip()]

        def read():
            _prepare_runtime()
            from mlcookbook.io import load_dataset_with_report

            dataset, report = load_dataset_with_report(path, delimiter=delimiter, encoding=encoding,
                                                       missing_markers=missing_markers)
            view = _dataset_view(dataset, Path(path).name, report)
            view["source"] = {"kind": "file", "path": str(Path(path).resolve()), "delimiter": delimiter,
                              "encoding": encoding, "missing_markers": missing_markers,
                              "parse_dates": True, "keep_leading_zeros": True}
            return dataset, Path(path).name, view
        self._start_job(read, self._use_import, "Loading file...")

    def _use_import(self, loaded):
        self._runtime_initialized = True
        self.imported = loaded
        self.data_status = "Dataset loaded. Review the preview and column roles, then choose a recipe."
        # A successful replacement invalidates every recipe's results, even for the same path.
        self._clear_sessions()
        if isinstance(self.screen, DataScreen):
            self.screen.set_dataset(self._workspace_view())
            self._set_status(self.data_status)
        elif isinstance(self.screen, ReportScreen):
            self._display_import(loaded)
        self._refresh_shell()

    def _display_import(self, loaded):
        dataset, name, view = loaded
        self.screen.is_example = False
        self.screen.source = deepcopy(view.get("source", {"kind": "snapshot"}))
        self.screen.dataset = dataset
        self.screen.dataset_name = name
        self.screen.set_dataset(view)
        self.screen.show_report({})
        self.screen.set_status("File loaded.")
        self._update_input_context()

    def connect_local_llm(self, backend, base_url, selected):
        if backend == "ollama":
            return self.connect_ollama(base_url, selected)
        if backend != "llamacpp":
            return
        from mlcookbook.llamacpp import llamacpp_models

        def connect():
            return llamacpp_models(base_url, stop_requested=QThread.currentThread().isInterruptionRequested).to_dict()

        def accept(result):
            self.screen.show_llamacpp_connection(result)
            self.screen.research.set_connection(base_url, result["models"][0]["name"] if result["models"] else "")
            self._set_status("llama.cpp is loading its model. Refresh when ready." if result["status"] == "loading" else
                             "llama.cpp connection checked. Review the loaded model, then generate.")

        self._start_job(connect, accept, "Checking the local llama.cpp server…", cancellable=True)

    def connect_ollama(self, base_url, selected):
        from mlcookbook.ollama import describe_ollama_model, ollama_models

        def connect():
            stopped = QThread.currentThread().isInterruptionRequested
            connection = ollama_models(base_url, stop_requested=stopped)
            chosen = selected or (connection.models[0].name if connection.models else "")
            result = connection.to_dict()
            result["selected"] = None
            if any(model.name == chosen for model in connection.models):
                result["selected"] = describe_ollama_model(base_url, chosen, connection=connection,
                                                           stop_requested=stopped).to_dict()
            elif chosen:
                result["warnings"].append(f"Previously selected model {chosen!r} is no longer installed. Choose another model.")
            return result

        def accept(result):
            self.screen.show_connection(result)
            self._set_status("Ollama connection checked. Choose an installed local model, then generate.")

        self._start_job(connect, accept, "Checking the local Ollama server…", cancellable=True)

    def llm_lab_action(self, action, payload):
        """Research is independent of recipe reports, active data and .mlcb v1."""
        if self.job is not None or not isinstance(self.screen, LLMScreen):
            return
        screen = self.screen
        values = deepcopy(payload or {})

        def stopped():
            return QThread.currentThread().isInterruptionRequested()

        def progress(event):
            QThread.currentThread().progress.emit(event)

        def show_progress(event):
            if isinstance(event, str):
                text = event
            elif "trial" in event:
                outcome = event.get("outcome", {}).get("status", "running")
                text = f"Trial {event['trial']} / {event.get('total', '?')} · {outcome}"
            elif "stage" in event:
                text = str(event["stage"]).replace("_", " ")
                if "completed" in event:
                    text += f" · {event['completed']:,} / {event.get('total', 0):,}"
            else:
                text = str(event.get("message", "Research operation running…"))
            screen.research.show_status(text[:240])

        def completed(value):
            if action == "inspect":
                screen.research.show_inspection(value)
                message = "GGUF descriptors inspected. Review tensor storage and file identity in Model inspection."
            elif action == "compare_model":
                screen.research.show_model_comparison(value)
                message = value["summary"]
            elif action in ("capture", "import_graph"):
                screen.research.show_capture(value)
                message = "Actual compute graph ready. Select a phase and operation to inspect its tensor dependencies."
            elif action == "compare_graph":
                screen.research.show_graph_comparison(value)
                message = value["summary"]
            elif action == "benchmark":
                screen.research.show_benchmark(value)
                summary = value["summary"]
                message = ("Performance trials cancelled. " if summary["cancelled"] else "Performance trials finished. ")
                message += (f"{summary['completed_measured_trials']} / {summary['planned_measured_trials']} measured trials "
                            f"completed; {summary['failed_measured_trials']} failed. Review raw samples and measurement scope.")
            elif action == "probe":
                screen.research.show_capabilities(value)
                models = value.get("connection", {}).get("models", [])
                if models and not values.get("model"):
                    screen.research.set_connection(values["base_url"], models[0]["name"])
                message = "Server capabilities checked. Unsupported and untested features remain explicit."
            elif action == "open":
                screen.research.restore_document(value)
                message = "Research lab restored offline from its saved artifacts."
            else:
                message = f"{'Exported' if action == 'export' else 'Saved'} {Path(values['path']).name}."
            self.llm_lab_document = screen.research.lab_document()
            self.llm_lab_mode = screen.workspace_mode.currentData()
            screen.research.show_status("")
            self._set_status(message)

        def run():
            if action == "inspect":
                from mlcookbook.llm_lab import inspect_gguf

                return inspect_gguf(values["path"], stop_requested=stopped, on_progress=progress).to_dict()
            if action == "probe":
                from mlcookbook.llm_lab import probe_llamacpp

                return probe_llamacpp(values["base_url"], stop_requested=stopped).to_dict()
            if action == "compare_model":
                from mlcookbook.llm_lab import compare_gguf, inspect_gguf

                candidate = inspect_gguf(values["path"], stop_requested=stopped, on_progress=progress).to_dict()
                return compare_gguf(values["baseline"], candidate)
            if action == "benchmark":
                from mlcookbook.llm_lab import benchmark_llamacpp

                return benchmark_llamacpp(**values, stop_requested=stopped, on_progress=progress).to_dict()
            if action == "capture":
                from mlcookbook.llm_graph import capture_graph

                return capture_graph(**values, stop_requested=stopped, on_progress=progress)
            if action == "import_graph":
                from mlcookbook.llm_graph import load_graph_capture

                return load_graph_capture(values["path"], stop_requested=stopped)
            if action == "compare_graph":
                from mlcookbook.llm_graph import compare_graphs

                return compare_graphs(values["capture"], values["left"], values["right"])
            if action in ("save", "export", "open"):
                from mlcookbook.llm_lab_io import export_lab, load_lab, save_lab

                if action == "open":
                    return load_lab(values["path"])
                if action == "export":
                    return export_lab(values["path"], values["document"])
                return save_lab(values["path"], values["document"])
            raise ValueError("Unknown research action.")

        # Snapshot presentation data on the GUI thread, before the worker starts.
        if action in ("save", "export") and "document" not in values:
            values["document"] = screen.research.lab_document()
        self._start_job(run, completed, {
            "inspect": "Reading GGUF tensor descriptors…", "probe": "Checking llama.cpp research capabilities…",
            "compare_model": "Inspecting and comparing explicitly selected GGUF files…",
            "capture": "Capturing actual llama.cpp computation…", "import_graph": "Opening recorded compute graph…",
            "compare_graph": "Comparing recorded graph phases…",
            "benchmark": "Running bounded performance trials…", "save": "Saving research lab…",
            "export": "Exporting research artifacts…", "open": "Opening research lab offline…",
        }.get(action, "Research operation…"), cancellable=action not in ("save", "export", "open"),
                        on_progress=show_progress)

    def run_analysis(self, request):
        if self.job is not None or not isinstance(self.screen, ReportScreen):
            return
        screen = self.screen
        key = self.key

        def run():
            if key == "llm":
                return _run_report(key, screen.dataset, screen.dataset_name, request,
                                   stop_requested=QThread.currentThread().isInterruptionRequested,
                                   on_progress=lambda event: QThread.currentThread().progress.emit(event))
            return _run_report(key, screen.dataset, screen.dataset_name, request)

        self._start_job(run,
                        lambda report: self._show_result(report, request),
                        "Loading local model and generating text…" if key == "llm" else
                        "Running built-in example..." if screen.is_example else "Running analysis...",
                        cancellable=key == "llm", on_progress=screen.show_progress if key == "llm" else None)

    def cancel_analysis(self):
        if self.job is not None:
            self.job.requestInterruption()
            if isinstance(self.screen, KitchenScreen):
                self.screen.cancel_button.setEnabled(False)
            else:
                self.screen.set_cancellable(False)
            self._set_status("Cancelling data fetch…" if isinstance(self.screen, DataPlaygroundScreen) else
                             "Stopping after the current model operation…")

    def _show_result(self, report, request):
        if "workflow_run" in report:
            history = list(self.screen._report.get("workflow_history", []))
            if any(saved["dataset_fingerprint"] != report["workflow_run"]["dataset_fingerprint"] for saved in history):
                history = []
                report["warnings"].append("Dataset values or roles changed; the run history now starts from this source snapshot.")
            history.append(report["workflow_run"])
            report["workflow_history"] = history[-10:]
            if len(history) > 10:
                report["warnings"].append("Keeping the ten most recent runs in this experiment. Export runs you want to archive separately.")
        self.screen.show_report(report, request=request)
        if "workflow_run" in report:
            rows = report["workflow_run"]["rows"]
            valid = sum(row["status"] == "valid" for row in rows)
            cancelled = any(row["status"] == "cancelled" for row in rows)
            self.screen.set_status(("Batch cancelled; completed rows retained. " if cancelled else "Text workflow complete. ") +
                                   f"{valid} of {len(rows)} source rows validated. Review row outcomes before accepting columns.")
        else:
            self.screen.set_status("Analysis complete.")

    def workflow_action(self, action, payload):
        from mlcookbook.core.dataset import Dataset
        from mlcookbook.llm_workflow import derive_dataset, export_workflow

        screen = self.screen
        if self.job is not None or not isinstance(screen, LLMScreen):
            return
        report = screen._report
        run = report.get("workflow_run")
        if not run:
            return
        request = deepcopy(screen._report_request)
        dataset = Dataset(screen.dataset.df, overrides=request["overrides"])
        if action in ("retry", "derive") and screen.request() != request:
            self._set_status("Settings changed. Run the full workflow before retrying or accepting fields.")
            return
        if action == "retry":
            def retry():
                return _run_llm_workflow(dataset, request, screen.dataset_name, previous=run, positions=payload,
                                         stop_requested=QThread.currentThread().isInterruptionRequested,
                                         on_progress=lambda event: QThread.currentThread().progress.emit(event))
            self._start_job(retry, lambda result: self._show_result(result, request), "Retrying selected source rows…",
                            cancellable=True, on_progress=screen.show_progress)
        elif action == "export":
            self._start_job(lambda: export_workflow(payload, dataset, run),
                            lambda _: self._set_status(f"Exported {Path(payload).name} with source rows, outcomes and run metadata."),
                            "Exporting LLM workflow…")
        elif action == "compare":
            history = report["workflow_history"]
            selected = [saved for saved in history if payload is None or saved["run_id"] in payload]
            def compare():
                result = _workflow_report(run, dataset, screen.dataset_name, history=selected, compare=True)
                result["workflow_history"] = history
                result["comparison_run_ids"] = [saved["run_id"] for saved in selected]
                return result
            def compared(result):
                screen.show_report(result, request=request)
                self._set_status(f"Compared {len(selected)} saved runs on the same source rows. Inspect validation, reference coverage and timings.")
            self._start_job(compare,
                            compared, "Comparing saved runs…")
        elif action == "derive":
            if not any(row["status"] == "valid" for row in run["rows"]):
                self._set_status("No validated rows are available to accept.")
                return
            def derive():
                derived, lineage = derive_dataset(dataset, run)
                name = screen.dataset_name + " · generated fields"
                view = _dataset_view(derived, name)
                view["source"] = {**lineage, "original_source": deepcopy(screen.source)}
                view["notes"].append("Accepted LLM-derived columns. Invalid/skipped/unprocessed rows remain missing. Run " + run["run_id"])
                return derived, name, view
            def accept(loaded):
                self._remember_screen()
                saved = self.sessions["llm"]
                self._clear_sessions()
                self.sessions["llm"] = saved
                self.workspace_sessions["llm"] = saved
                self.imported = loaded
                self.data_status = "Accepted generated fields. Review the new columns before using them in an analysis."
                self._refresh_shell()
                self._set_status("Derived dataset ready. Click Back and open an analysis recipe to use the generated columns. "
                                 "Local LLMs retains this source run; Save experiment keeps it for later.")
            self._start_job(derive, accept, "Creating a typed derived dataset…")


def _startup_options(argv):
    parser = argparse.ArgumentParser(description="MLcookbook offline desktop workbench")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--study", metavar="EXPORT_JSON", help="open a verified Movement Study export without training")
    start.add_argument("--experiment", metavar="FILE_MLCB", help="restore a saved experiment without retraining")
    start.add_argument("--movement-study", action="store_true", help="open the Movement Study setup card")
    return parser.parse_known_args(argv)


def _open_startup(controller, options):
    if options.study or options.movement_study:
        controller.open_recipe(next(recipe for recipe in RECIPES if recipe.key == "wearable-study"),
                               on_ready=(lambda: controller.load_wearable_study(str(Path(options.study).resolve())))
                               if options.study else None)
    elif options.experiment:
        controller.open_experiment(str(Path(options.experiment).resolve()))


def main(argv=None) -> int:
    options, qt_arguments = _startup_options(sys.argv[1:] if argv is None else argv)
    app = QApplication([sys.argv[0], *qt_arguments])
    app.setApplicationName("MLcookbook")
    app.setFont(QFont("Segoe UI", theme.BODY_POINT_SIZE))
    app.setStyleSheet(theme.stylesheet())
    icon_path = _assets_dir() / "mlcookbook.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    controller = MLCookbookApp()
    controller.window.resize(1280, 860)
    controller.window.show()
    QTimer.singleShot(0, lambda: _open_startup(controller, options))
    try:
        return app.exec()
    finally:
        # Retire widgets and workers on the GUI thread while QApplication is
        # still alive, rather than leaving their destruction to interpreter exit.
        controller.wait_for_job()
        app.aboutToQuit.disconnect(controller.wait_for_job)
        if controller.job is not None:
            controller.job.deleteLater()
        # Also include hidden candidate screens/dialogs that have not yet been
        # attached to the main window when a background operation is interrupted.
        for widget in app.topLevelWidgets():
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


if __name__ == "__main__":
    raise SystemExit(main())
