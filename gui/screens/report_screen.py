"""Shared layout for recipe screens that show one analysis run.

Dumb like every screen: it renders a plain ``report`` dict handed in by app.py
and never imports mlcookbook. The report shape (all values already strings):

    {
        "dataset":  "Diabetes · 442 rows",
        "target":   "progression",
        "features": "age, sex, bmi, ...",
        "plots":    [{"title": "...", "x_label": "...", "y_label": "...",
                      "series": [{"kind": "points"|"line", "color": "accent"|"alt"|"muted",
                                  "label": "...", "points": [(x, y), ...]}]}],
        "metric_groups": [{"title": "Test set", "items": [("R²", "0.452"), ...]}],
        "tables":   [{"title": "Coefficients", "headers": [...], "rows": [[...], ...]}],
        "warnings": ["..."],
    }
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.plot import PlotWidget


class ReportScreen(QWidget):
    back_requested = Signal()

    def __init__(self, title: str, engine: dict[str, str], report: dict, parent=None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SPACE_XL, theme.SPACE_LG, theme.SPACE_XL, theme.SPACE_LG)
        outer.setSpacing(theme.SPACE_MD)

        back = QPushButton("←  Recipes")
        back.setObjectName("BackButton")
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.clicked.connect(self.back_requested)
        outer.addWidget(back, alignment=Qt.AlignmentFlag.AlignLeft)

        heading = QLabel(title)
        heading.setObjectName("ScreenTitle")
        outer.addWidget(heading)

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
        body.addWidget(self._build_form(report), stretch=0)
        body.addWidget(self._build_report(report), stretch=1)
        outer.addLayout(body, stretch=1)

    # --- Left: what was run --------------------------------------------------

    def _build_form(self, report: dict) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(320)
        form = QVBoxLayout(panel)
        form.setContentsMargins(0, theme.SPACE_SM, 0, 0)
        form.setSpacing(theme.SPACE_XS)

        badge = QLabel(f"DEFAULT DATASET  ·  {report.get('dataset', '')}")
        badge.setObjectName("DatasetBadge")
        badge.setWordWrap(True)
        form.addWidget(badge)
        form.addSpacing(theme.SPACE_SM)

        for label_text, value in [
            ("TARGET COLUMN", report.get("target", "")),
            ("FEATURE COLUMNS", report.get("features", "")),
        ]:
            label = QLabel(label_text)
            label.setObjectName("FieldLabel")
            field = QLineEdit(value)
            field.setReadOnly(True)
            field.setCursorPosition(0)
            field.setToolTip(value)
            form.addWidget(label)
            form.addWidget(field)
            form.addSpacing(theme.SPACE_SM)

        run = QPushButton("Run Analysis")
        run.setObjectName("PrimaryButton")
        run.setEnabled(False)
        run.setToolTip("Custom data isn't wired up yet — this is the recipe run on its default dataset.")
        form.addSpacing(theme.SPACE_XS)
        form.addWidget(run)
        form.addStretch()
        return panel

    # --- Right: results ------------------------------------------------------

    def _build_report(self, report: dict) -> QWidget:
        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(0, theme.SPACE_SM, theme.SPACE_SM, 0)
        column.setSpacing(theme.SPACE_MD)

        for spec in report.get("plots", []):  # the graph leads the report
            column.addWidget(PlotWidget(spec))
        for group in report.get("metric_groups", []):
            column.addWidget(self._metric_group(group))
        for table in report.get("tables", []):
            column.addWidget(self._table(table))
        for warning in report.get("warnings", []):
            note = QLabel(f"⚠  {warning}")
            note.setObjectName("WarningNote")
            note.setWordWrap(True)
            column.addWidget(note)
        column.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _metric_group(self, group: dict) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        layout.addWidget(self._section_label(group["title"]))

        row = QHBoxLayout()
        row.setSpacing(theme.SPACE_SM)
        for label, value in group["items"]:
            tile = QFrame()
            tile.setObjectName("MetricTile")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setSpacing(2)
            number = QLabel(value)
            number.setObjectName("MetricValue")
            caption = QLabel(label)
            caption.setObjectName("MetricLabel")
            tile_layout.addWidget(number)
            tile_layout.addWidget(caption)
            row.addWidget(tile)
        layout.addLayout(row)
        return box

    def _table(self, spec: dict) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        layout.addWidget(self._section_label(spec["title"]))

        headers, rows = spec["headers"], spec["rows"]
        table = QTableWidget(len(rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setShowGrid(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for r, row_values in enumerate(rows):
            for c, text in enumerate(row_values):
                item = QTableWidgetItem(text)
                if c > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(r, c, item)
        row_height = table.verticalHeader().defaultSectionSize()
        table.setFixedHeight(table.horizontalHeader().height() + row_height * len(rows) + 4)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(table)
        return box

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("FieldLabel")
        return label
