"""The front page: section header + scrollable 3-column grid of recipe cards."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.recipe_card import RecipeCard
from gui.recipes import RECIPES, Recipe


class FrontPage(QWidget):
    """Renders a collection of recipes as a grid; add rows by adding data.

    Emits ``recipe_selected`` when a ready card is clicked; routing is app.py's
    concern, not this view's.
    """

    recipe_selected = Signal(object)  # emits a Recipe

    def __init__(self, recipes: list[Recipe] | None = None, parent=None) -> None:
        super().__init__(parent)
        self._count = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SPACE_XL, theme.SPACE_LG, theme.SPACE_XL, 0)
        outer.setSpacing(theme.SPACE_MD)

        outer.addWidget(self._build_header())

        self._grid = QGridLayout()
        self._grid.setSpacing(theme.SPACE_MD)
        self._grid.setContentsMargins(0, 0, theme.SPACE_SM, theme.SPACE_LG)
        for col in range(theme.GRID_COLUMNS):
            self._grid.setColumnStretch(col, 1)

        grid_host = QWidget()
        host_layout = QVBoxLayout(grid_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.addLayout(self._grid)
        host_layout.addStretch()  # keep a short list of cards pinned to the top

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(grid_host)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll, stretch=1)

        for recipe in recipes if recipes is not None else RECIPES:
            self.add_recipe(recipe)

    def add_recipe(self, recipe: Recipe) -> None:
        """Append a card to the grid (fills left-to-right, top-to-bottom)."""
        row, col = divmod(self._count, theme.GRID_COLUMNS)
        card = RecipeCard(recipe)
        card.clicked.connect(self.recipe_selected)
        self._grid.addWidget(card, row, col)
        self._count += 1

    def _build_header(self) -> QWidget:
        header = QWidget()
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_MD)

        tick = QFrame()
        tick.setObjectName("PageTitleTick")
        tick.setFixedSize(3, 46)
        row.addWidget(tick)

        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel("Recipes")
        title.setObjectName("PageTitle")
        subtitle = QLabel("A growing collection of machine learning recipes.")
        subtitle.setObjectName("PageSubtitle")
        text.addWidget(title)
        text.addWidget(subtitle)
        row.addLayout(text)
        row.addStretch()
        return header
