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

from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication, QWidget

from gui import theme
from gui.front_page import FrontPage
from gui.main_window import MainWindow
from gui.recipes import Recipe
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

def _linear_regression_screen() -> QWidget:
    from mlcookbook import Dataset
    from mlcookbook.regression import linear_regression

    frame = _diabetes_frame()
    result = linear_regression(Dataset(frame), target="progression")

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
        "dataset": f"Diabetes · {len(frame)} rows",
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
    return LinearRegressionScreen(_engine_info(linear_regression), report)


def _logistic_regression_screen() -> QWidget:
    from mlcookbook import Dataset
    from mlcookbook.classification import logistic_regression

    frame = _breast_cancer_frame()
    result = logistic_regression(Dataset(frame), target="diagnosis", positive_class="malignant")

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
        "dataset": f"Breast cancer (Wisconsin) · {len(frame)} rows",
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
    return LogisticRegressionScreen(_engine_info(logistic_regression), report)


SCREEN_FACTORIES: dict[str, callable] = {
    "linear-regression": _linear_regression_screen,
    "logistic-regression": _logistic_regression_screen,
}


# --- Application --------------------------------------------------------------

class MLCookbookApp:
    """Owns the window and swaps screens; holds no analytical state."""

    def __init__(self) -> None:
        self.window = MainWindow()
        self.show_front_page()

    def show_front_page(self) -> None:
        page = FrontPage()
        page.recipe_selected.connect(self.open_recipe)
        self.window.show_screen(page)

    def open_recipe(self, recipe: Recipe) -> None:
        factory = SCREEN_FACTORIES.get(recipe.key)
        if factory is None:
            return  # no screen registered yet — card stays inert
        screen = factory()
        screen.back_requested.connect(self.show_front_page)
        self.window.show_screen(screen)


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
