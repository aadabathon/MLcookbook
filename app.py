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
gui/recipes.py, give it a key, and register a factory here.

Launch with:  python -m mlcookbook
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if not getattr(sys, "frozen", False):  # a PyInstaller bundle manages its own paths
    for _p in (str(_ROOT), str(_ROOT / "src")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication, QWidget

from gui import theme
from gui.front_page import FrontPage
from gui.main_window import MainWindow
from gui.recipes import Recipe
from gui.screens.decision_tree import DecisionTreeScreen
from gui.screens.random_forest import RandomForestScreen
from gui.screens.linear_regression import LinearRegressionScreen
from gui.screens.logistic_regression import LogisticRegressionScreen


def _assets_dir() -> Path:
    # Frozen builds unpack data files under sys._MEIPASS.
    return Path(getattr(sys, "_MEIPASS", _ROOT)) / "gui" / "assets"


# --- Default datasets ---------------------------------------------------------
# Each recipe screen opens by running its core function on a small built-in
# dataset (bundled with scikit-learn, so no new dependency) to show a real result.

def _diabetes_frame():
    from sklearn.datasets import load_diabetes

    return load_diabetes(as_frame=True, scaled=False).frame.rename(columns={"target": "progression"})


def _breast_cancer_frame():
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame[[c for c in bunch.frame.columns if c.startswith("mean ")]].copy()
    frame["diagnosis"] = bunch.target.map(dict(enumerate(bunch.target_names)))  # 0 = malignant
    return frame


# --- Result formatting (core result -> plain strings for the dumb screens) ----

def _num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None or value != value else f"{value:.{digits}f}"


def _engine_info(func) -> dict[str, str]:
    doc = (inspect.getdoc(func) or "").splitlines()
    return {
        "callable": f"{func.__module__}.{func.__qualname__}",
        "signature": str(inspect.signature(func)),
        "summary": (doc[0] if doc else "").replace("*", ""),
    }


def _coefficient_rows(coefficients: dict[str, float], extra: dict[str, float] | None = None):
    rows = []
    for name, value in sorted(coefficients.items(), key=lambda kv: abs(kv[1]), reverse=True):
        row = [name, _num(value)]
        if extra is not None:
            row.append(_num(extra[name]))
        rows.append(row)
    return rows


def _sigmoid_plot(result) -> dict:
    """Logistic fit: held-out rows on the log-odds axis, with the sigmoid they follow."""
    import math

    eps = 1e-6
    z = [math.log(max(eps, min(1 - eps, p)) / (1 - max(eps, min(1 - eps, p)))) for p in result.test_probability]
    lo, hi = min(z) - 0.5, max(z) + 0.5
    curve = [(lo + (hi - lo) * i / 100, 1 / (1 + math.exp(-(lo + (hi - lo) * i / 100)))) for i in range(101)]
    negative, positive = result.classes
    # Deterministic vertical jitter so overlapping points stay readable.
    jitter = lambda i: ((i * 0.618034) % 1 - 0.5) * 0.05
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


# --- Screen factories: where a recipe meets its core module -------------------

def _linear_regression_report(dataset, target, features=None, positive_class=None, dataset_name="Dataset") -> dict:
    from mlcookbook.regression import linear_regression

    frame = dataset.df
    result = linear_regression(dataset, target=target, features=features)

    def metrics(title: str, m) -> dict:
        return {
            "title": f"{title} · {m.n_obs} rows",
            "items": [("R²", _num(m.r2)), ("RMSE", _num(m.rmse, 1)), ("MAE", _num(m.mae, 1))],
        }

    lo, hi = min(result.test_actual + result.test_predicted), max(result.test_actual + result.test_predicted)
    report = {
        "plots": [
            {
                "title": "Actual vs predicted · held-out rows",
                "x_label": f"predicted {result.target}",
                "y_label": f"actual {result.target}",
                "series": [
                    {"kind": "line", "color": "muted", "label": "perfect fit", "points": [(lo, lo), (hi, hi)]},
                    {
                        "kind": "points",
                        "color": "accent",
                        "label": "test rows",
                        "points": list(zip(result.test_predicted, result.test_actual)),
                    },
                ],
            }
        ],
        "dataset": f"{dataset_name} · {len(frame)} rows",
        "target": result.target,
        "features": ", ".join(result.features),
        "metric_groups": [metrics("Test set", result.test_metrics), metrics("Train set", result.train_metrics)],
        "tables": [
            {
                "title": f"Coefficients (per SD of feature) · intercept {_num(result.intercept, 1)}",
                "headers": ["FEATURE", "COEFFICIENT"],
                "rows": _coefficient_rows(result.coefficients),
            }
        ],
        "warnings": result.warnings,
    }
    return report


def _logistic_regression_report(dataset, target, features=None, positive_class=None, dataset_name="Dataset") -> dict:
    from mlcookbook.classification import logistic_regression

    frame = dataset.df
    result = logistic_regression(dataset, target=target, features=features, positive_class=positive_class)

    def metrics(title: str, m) -> dict:
        return {
            "title": f"{title} · {m.n_obs} rows",
            "items": [
                ("ACCURACY", _num(m.accuracy)),
                ("PRECISION", _num(m.precision)),
                ("RECALL", _num(m.recall)),
                ("F1", _num(m.f1)),
                ("ROC AUC", _num(m.roc_auc)),
            ],
        }

    negative, positive = result.classes
    matrix = result.test_metrics.confusion_matrix
    report = {
        "plots": [_sigmoid_plot(result)],
        "dataset": f"{dataset_name} · {len(frame)} rows",
        "target": f"{result.target}  (positive = {result.positive_class})",
        "features": ", ".join(result.features),
        "metric_groups": [metrics("Test set", result.test_metrics), metrics("Train set", result.train_metrics)],
        "tables": [
            {
                "title": "Confusion matrix (test set)",
                "headers": ["ACTUAL / PREDICTED", str(negative).upper(), str(positive).upper()],
                "rows": [[str(negative), *map(str, matrix[0])], [str(positive), *map(str, matrix[1])]],
            },
            {
                "title": f"Coefficients (log-odds of {positive}, per SD) · intercept {_num(result.intercept)}",
                "headers": ["FEATURE", "COEFFICIENT", "ODDS RATIO"],
                "rows": _coefficient_rows(result.coefficients, result.odds_ratios),
            },
        ],
        "warnings": result.warnings,
    }
    return report


def _decision_tree_report(dataset, target, features=None, positive_class=None, dataset_name="Dataset") -> dict:
    from mlcookbook.classification import decision_tree

    frame = dataset.df
    result = decision_tree(dataset, target=target, features=features, positive_class=positive_class)
    negative, positive = result.classes

    def metrics(title: str, m) -> dict:
        return {
            "title": f"{title} · {m.n_obs} rows",
            "items": [
                ("ACCURACY", _num(m.accuracy)),
                ("PRECISION", _num(m.precision)),
                ("RECALL", _num(m.recall)),
                ("F1", _num(m.f1)),
                ("ROC AUC", _num(m.roc_auc)),
            ],
        }

    tree_nodes = [
        {
            "title": f"{n.feature} ≤ {n.threshold:.4g}" if n.left is not None else f"→ {n.prediction}",
            "subtitle": f"n={n.n_samples} · gini {n.impurity:.2f}",
            "fraction": n.class_counts[1] / sum(n.class_counts),
            "positive": n.prediction == positive,
            "left": n.left,
            "right": n.right,
        }
        for n in result.nodes
    ]

    curve = result.depth_curve
    depth_series = []
    if curve:
        train = [(d.max_depth, d.train_accuracy) for d in curve]
        cv = [(d.max_depth, d.cv_accuracy) for d in curve]
        low = min(y for _, y in train + cv)
        depth_series = [
            {"kind": "line", "color": "muted", "label": f"chosen depth ({result.max_depth})",
             "points": [(result.max_depth, low), (result.max_depth, 1.0)]},
            {"kind": "line", "color": "alt", "label": "train", "points": train},
            {"kind": "points", "color": "alt", "label": "", "points": train},
            {"kind": "line", "color": "accent", "label": "cross-validated", "points": cv},
            {"kind": "points", "color": "accent", "label": "", "points": cv},
        ]

    matrix = result.test_metrics.confusion_matrix
    report = {
        "tree": {
            "nodes": tree_nodes,
            "negative_label": str(negative),
            "positive_label": str(positive),
        },
        "plots": [
            {
                "title": "Accuracy vs tree depth · the overfitting curve",
                "x_label": "max_depth",
                "y_label": "accuracy",
                "series": depth_series,
            }
        ]
        if depth_series
        else [],
        "dataset": f"{dataset_name} · {len(frame)} rows",
        "target": f"{result.target}  (positive = {result.positive_class})",
        "features": ", ".join(result.features),
        "settings": f"max_depth={result.max_depth} · min_samples_leaf={result.min_samples_leaf}"
        f" · fitted depth {result.depth}, {result.n_leaves} leaves",
        "metric_groups": [metrics("Test set", result.test_metrics), metrics("Train set", result.train_metrics)],
        "bars": [
            {
                "title": "Feature importance (share of impurity reduction)",
                "items": [
                    (name, value, f"{value:.1%}")
                    for name, value in sorted(result.feature_importances.items(), key=lambda kv: -kv[1])
                    if value > 0
                ],
            }
        ],
        "tables": [
            {
                "title": "Decision rules (one per leaf)",
                "headers": ["RULE", "SAMPLES", "CONFIDENCE"],
                "stretch_first": True,
                "rows": [
                    [
                        f"IF {' AND '.join(r.conditions).replace('<=', '≤') or 'always'}  THEN  {r.prediction}",
                        str(r.n_samples),
                        f"{r.confidence:.0%}",
                    ]
                    for r in sorted(result.rules, key=lambda r: -r.n_samples)
                ],
            },
            {
                "title": "Confusion matrix (test set)",
                "headers": ["ACTUAL / PREDICTED", str(negative).upper(), str(positive).upper()],
                "rows": [[str(negative), *map(str, matrix[0])], [str(positive), *map(str, matrix[1])]],
            },
        ],
        "warnings": result.warnings,
    }
    return report


def _random_forest_report(dataset, target, features=None, positive_class=None, dataset_name="Dataset") -> dict:
    from mlcookbook.classification import random_forest

    result = random_forest(dataset, target=target, features=features, positive_class=positive_class)
    negative, positive = result.classes

    def metrics(title, m):
        return {
            "title": f"{title} - {m.n_obs} rows",
            "items": [("ACCURACY", _num(m.accuracy)), ("PRECISION", _num(m.precision)),
                      ("RECALL", _num(m.recall)), ("F1", _num(m.f1)), ("ROC AUC", _num(m.roc_auc))],
        }

    matrix = result.test_metrics.confusion_matrix
    return {
        "dataset": f"{dataset_name} - {len(dataset.df)} rows",
        "target": f"{result.target}  (positive = {positive})",
        "features": ", ".join(result.features),
        "settings": f"{result.n_estimators} trees, bootstrap samples, max_features={result.max_features}, "
                    f"max_depth={result.max_depth}, min_samples_leaf={result.min_samples_leaf}",
        "plots": [{
            "title": "ROC curve - held-out rows",
            "x_label": "false positive rate",
            "y_label": "true positive rate (recall)",
            "series": [
                {"kind": "line", "color": "muted", "label": "chance", "points": [(0, 0), (1, 1)]},
                {"kind": "line", "color": "accent", "label": f"forest (AUC {_num(result.test_metrics.roc_auc)})",
                 "points": list(zip(result.roc_fpr, result.roc_tpr))},
            ],
        }] if result.roc_fpr else [],
        "metric_groups": [
            metrics("Test set", result.test_metrics), metrics("Train set", result.train_metrics),
            {"title": "Forest structure", "items": [
                ("TREES", str(result.n_estimators)),
                ("DEPTH RANGE", f"{min(result.tree_depths)} - {max(result.tree_depths)}"),
                ("LEAF COUNT RANGE", f"{min(result.tree_leaf_counts)} - {max(result.tree_leaf_counts)}"),
            ]},
        ],
        "bars": [{
            "title": "Feature importance (impurity reduction; encoded columns)",
            "items": [(name, value, f"{value:.1%}") for name, value in
                      sorted(result.feature_importances.items(), key=lambda item: -item[1]) if value > 0],
        }],
        "tables": [{
            "title": "Confusion matrix (test set)",
            "headers": ["ACTUAL / PREDICTED", str(negative).upper(), str(positive).upper()],
            "rows": [[str(negative), *map(str, matrix[0])], [str(positive), *map(str, matrix[1])]],
        }],
        "warnings": result.warnings,
    }


def _default_screen(key, demo=True):
    from mlcookbook import Dataset
    from mlcookbook.regression import linear_regression
    from mlcookbook.classification import logistic_regression, decision_tree, random_forest

    linear = key == "linear-regression"
    dataset = Dataset(_diabetes_frame() if linear else _breast_cancer_frame())
    target = "progression" if linear else "diagnosis"
    name = "Diabetes" if linear else "Breast cancer (Wisconsin)"
    func, screen_class = {
        "linear-regression": (linear_regression, LinearRegressionScreen),
        "logistic-regression": (logistic_regression, LogisticRegressionScreen),
        "decision-tree": (decision_tree, DecisionTreeScreen),
        "random-forest": (random_forest, RandomForestScreen),
    }[key]
    report = REPORT_BUILDERS[key](dataset, target, positive_class=None if linear else "malignant", dataset_name=name) if demo else {}
    screen = screen_class(_engine_info(func), report)
    screen.set_dataset(_dataset_view(dataset, name), classification=not linear, target=target)
    if not linear:
        screen.positive.setCurrentText("malignant")
    screen.set_status("Built-in example. Open your own file or adjust the selections and run.")
    screen.dataset = dataset
    screen.dataset_name = name
    return screen


REPORT_BUILDERS = {
    "linear-regression": _linear_regression_report,
    "logistic-regression": _logistic_regression_report,
    "decision-tree": _decision_tree_report,
    "random-forest": _random_forest_report,
}
SCREEN_FACTORIES = {
    "linear-regression": lambda demo=True: _default_screen("linear-regression", demo),
    "logistic-regression": lambda demo=True: _default_screen("logistic-regression", demo),
    "decision-tree": lambda demo=True: _default_screen("decision-tree", demo),
    "random-forest": lambda demo=True: _default_screen("random-forest", demo),
}


def _dataset_view(dataset, name):
    import pandas as pd

    frame = dataset.df
    return {
        "name": name,
        "rows": len(frame),
        "columns": [
            {"name": col.name, "type": col.var_type.value,
             "missing": int(frame[col.name].isna().sum()),
             "classes": frame[col.name].dropna().unique().tolist() if frame[col.name].nunique() == 2 else []}
            for col in dataset.schema().columns
        ],
        "preview": [["" if pd.isna(value) else str(value) for value in row]
                    for row in frame.head(20).itertuples(index=False, name=None)],
    }


def _run_report(key, dataset, name, request):
    import pandas as pd
    from mlcookbook import Dataset

    # Overrides describe roles; they never convert strings into numbers silently.
    for column, role in request["overrides"].items():
        values = dataset.df[column]
        if role == "numeric" and not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"{column!r} contains non-numeric data. Clean the source column before choosing numeric.")
        if role == "boolean" and not pd.api.types.is_bool_dtype(values):
            raise ValueError(f"{column!r} is not stored as boolean. Use categorical for coded labels.")
    configured = Dataset(dataset.df, overrides=request["overrides"])
    return REPORT_BUILDERS[key](configured, request["target"], request["features"],
                               request["positive_class"], name)


class _Job(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

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
        QApplication.instance().aboutToQuit.connect(self.wait_for_job)
        self.show_front_page()

    def wait_for_job(self):
        if self.job is not None:
            self.job.wait()

    def show_front_page(self) -> None:
        page = FrontPage()
        page.recipe_selected.connect(self.open_recipe)
        self.window.show_screen(page)

    def open_recipe(self, recipe: Recipe) -> None:
        factory = SCREEN_FACTORIES.get(recipe.key)
        if factory is None:
            return
        screen = factory(demo=self.imported is None)
        self.screen = screen
        self.key = recipe.key
        if self.imported is not None:
            self._use_import(self.imported)
        screen.back_requested.connect(self.show_front_page)
        screen.load_requested.connect(self.load_file)
        screen.run_requested.connect(self.run_analysis)
        self.window.show_screen(screen)

    def _start_job(self, operation, accept, message):
        if self.job is not None:
            return
        screen = self.screen
        screen.set_busy(True)
        screen.set_status(message)
        self.job = _Job(operation)
        self.job.succeeded.connect(accept)
        self.job.failed.connect(lambda error: screen.set_status("Unable to complete: " + error))
        self.job.finished.connect(self._job_finished)
        self.job.start()

    def _job_finished(self):
        self.screen.set_busy(False)
        self.job.deleteLater()
        self.job = None

    def load_file(self, path, delimiter, encoding):
        from mlcookbook.io import load_dataset

        def read():
            dataset = load_dataset(path, delimiter=delimiter, encoding=encoding)
            return dataset, Path(path).name, _dataset_view(dataset, Path(path).name)
        self._start_job(read, self._use_import, "Loading file...")

    def _use_import(self, loaded):
        self.imported = loaded
        dataset, name, view = loaded
        self.screen.dataset = dataset
        self.screen.dataset_name = name
        self.screen.set_dataset(view, classification=self.key != "linear-regression")
        self.screen.show_report({})
        self.screen.set_status("File loaded. Review column types, choose a target and features, then run.")

    def run_analysis(self, request):
        screen = self.screen
        key = self.key
        self._start_job(lambda: _run_report(key, screen.dataset, screen.dataset_name, request),
                        self._show_result, "Running analysis...")

    def _show_result(self, report):
        self.screen.show_report(report)
        self.screen.set_status("Analysis complete. Results reflect the settings used for this run.")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MLcookbook")
    app.setFont(QFont("Segoe UI", theme.BODY_POINT_SIZE))
    app.setStyleSheet(theme.stylesheet())
    icon_path = _assets_dir() / "mlcookbook.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    controller = MLCookbookApp()
    controller.window.resize(1280, 860)
    controller.window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
