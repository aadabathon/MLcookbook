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


def _assets_dir() -> Path:
    # Frozen builds unpack data files under sys._MEIPASS.
    return Path(getattr(sys, "_MEIPASS", _ROOT)) / "gui" / "assets"


# --- Screen factories: where a recipe meets its core module -------------------

def _linear_regression_screen() -> QWidget:
    """Bind the LR screen to the core's entry function (placeholder wiring).

    For now the screen only *displays* the core callable it will eventually
    invoke — enough to prove card click → app.py → core module → screen.
    """
    from mlcookbook.regression import linear_regression

    doc = (inspect.getdoc(linear_regression) or "").splitlines()
    return LinearRegressionScreen(
        {
            "callable": f"{linear_regression.__module__}.{linear_regression.__qualname__}",
            "signature": str(inspect.signature(linear_regression)),
            "summary": (doc[0] if doc else "").replace("*", ""),
        }
    )


SCREEN_FACTORIES: dict[str, callable] = {
    "linear-regression": _linear_regression_screen,
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
