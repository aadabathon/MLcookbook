"""Reusable recipe card widget.

Purely presentational: renders one ``Recipe`` and emits ``clicked`` when a
*ready* card is clicked. Cards whose variant is not "ready" are inert. What a
click leads to is decided by app.py, never here.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout

from gui import theme
from gui.recipes import Recipe


class RecipeCard(QFrame):
    clicked = Signal(object)  # emits the card's Recipe

    def __init__(self, recipe: Recipe, parent=None) -> None:
        super().__init__(parent)
        self.recipe = recipe
        self.setObjectName("RecipeCard")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMinimumHeight(theme.CARD_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        if recipe.variant == "ready":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setAccessibleName(recipe.title)
            self.setAccessibleDescription(recipe.description)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*(theme.CARD_PADDING,) * 4)
        layout.setSpacing(theme.SPACE_SM)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        icon = QLabel(recipe.icon)
        icon.setObjectName("CardIcon")
        icon.setFixedSize(theme.ICON_SIZE, theme.ICON_SIZE)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon, alignment=Qt.AlignmentFlag.AlignHCenter)

        title = QLabel(recipe.title)
        title.setObjectName("CardTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        layout.addWidget(title)

        description = QLabel(recipe.description)
        description.setObjectName("CardDescription")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)

        layout.addStretch()

        pill = QLabel(recipe.status)
        pill.setObjectName("StatusPill")
        pill.setProperty("variant", recipe.variant)
        pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(pill, alignment=Qt.AlignmentFlag.AlignHCenter)

    def mouseReleaseEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.rect().contains(event.position().toPoint())
            and self.recipe.variant == "ready"
        ):
            self.clicked.emit(self.recipe)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if self.recipe.variant == "ready" and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            if not event.isAutoRepeat():
                self.clicked.emit(self.recipe)
            event.accept()
            return
        super().keyPressEvent(event)
