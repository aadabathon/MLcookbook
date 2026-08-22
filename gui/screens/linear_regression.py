"""Placeholder Linear Regression screen.

Layout only: a couple of inert inputs and a blank graph area. The ``engine``
dict (callable path, signature, summary) is supplied by app.py from the real
core function — this screen just displays it, proving the seam works without
this module ever importing mlcookbook.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui import theme


class LinearRegressionScreen(QWidget):
    back_requested = Signal()

    def __init__(self, engine: dict[str, str], parent=None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SPACE_XL, theme.SPACE_LG, theme.SPACE_XL, theme.SPACE_LG)
        outer.setSpacing(theme.SPACE_MD)

        back = QPushButton("←  Recipes")
        back.setObjectName("BackButton")
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.clicked.connect(self.back_requested)
        outer.addWidget(back, alignment=Qt.AlignmentFlag.AlignLeft)

        title = QLabel("Linear Regression")
        title.setObjectName("ScreenTitle")
        outer.addWidget(title)

        subtitle = QLabel(engine.get("summary", ""))
        subtitle.setObjectName("ScreenSubtitle")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        engine_line = QLabel(f"engine  {engine.get('callable', '?')}{engine.get('signature', '')}")
        engine_line.setObjectName("EngineInfo")
        engine_line.setWordWrap(True)
        engine_line.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(engine_line)

        body = QHBoxLayout()
        body.setSpacing(theme.SPACE_LG)
        body.addWidget(self._build_form(), stretch=0)
        body.addWidget(self._build_graph_placeholder(), stretch=1)
        outer.addLayout(body, stretch=1)

    def _build_form(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(320)
        form = QVBoxLayout(panel)
        form.setContentsMargins(0, theme.SPACE_SM, 0, 0)
        form.setSpacing(theme.SPACE_XS)

        for label_text, placeholder in [
            ("TARGET COLUMN", "e.g. price"),
            ("FEATURE COLUMNS", "comma-separated · blank = all usable"),
        ]:
            label = QLabel(label_text)
            label.setObjectName("FieldLabel")
            field = QLineEdit()
            field.setPlaceholderText(placeholder)
            form.addWidget(label)
            form.addWidget(field)
            form.addSpacing(theme.SPACE_SM)

        run = QPushButton("Run Regression")
        run.setObjectName("PrimaryButton")
        run.setEnabled(False)
        run.setToolTip("Not wired up yet — the recipe is still simmering.")
        form.addSpacing(theme.SPACE_XS)
        form.addWidget(run)
        form.addStretch()
        return panel

    def _build_graph_placeholder(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("GraphPlaceholder")
        layout = QVBoxLayout(frame)
        hint = QLabel("≈\n\nvisualization will render here")
        hint.setObjectName("GraphHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)
        return frame
