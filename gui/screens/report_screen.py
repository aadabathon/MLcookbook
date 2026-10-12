"""Shared layout for recipe screens that show one analysis run.

Dumb like every screen: it renders a plain ``report`` dict handed in by app.py
and never imports mlcookbook. The report shape (all values already strings):

    {
        "dataset":  "Diabetes · 442 rows",
        "target":   "progression",
        "features": "age, sex, bmi, ...",
        "settings": "max_depth=3 · min_samples_leaf=5",        # optional
        "tree":     {...},   # optional node-link diagram, see gui/tree_plot.py; leads the report
        "bars":     [{"title": "...", "items": [(label, 0..1, "text"), ...]}],  # optional
        "plots":    [{"title": "...", "x_label": "...", "y_label": "...",
                      "series": [{"kind": "points"|"line", "color": "accent"|"alt"|"muted",
                                  "label": "...", "points": [(x, y), ...]}]}],
        "metric_groups": [{"title": "Test set", "items": [("R²", "0.452"), ...]}],
        "tables":   [{"title": "Coefficients", "headers": [...], "rows": [[...], ...],
                      "stretch_first": False}],  # True: first column takes the slack
        "warnings": ["..."],
    }
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QComboBox,
    QFileDialog,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.plot import BarList, PlotWidget
from gui.tree_plot import TreeWidget


class ReportScreen(QWidget):
    back_requested = Signal()
    load_requested = Signal(str, str, str)
    run_requested = Signal(dict)

    def __init__(self, title: str, engine: dict[str, str], report: dict, parent=None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SPACE_XL, theme.SPACE_LG, theme.SPACE_XL, theme.SPACE_LG)
        outer.setSpacing(theme.SPACE_MD)

        self._busy = False
        self._columns = []
        back = QPushButton("←  Recipes")
        self.back_button = back
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
        self.body = body
        self.form_panel = self._build_form(report)
        body.addWidget(self.form_panel, stretch=0)
        self.report_widget = self._build_report(report)
        body.addWidget(self.report_widget, stretch=1)
        outer.addLayout(body, stretch=1)

    # --- Left: what was run --------------------------------------------------

    def _build_form(self, report: dict) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(360)
        form = QVBoxLayout(panel)
        form.setContentsMargins(0, theme.SPACE_SM, 0, 0)
        form.setSpacing(theme.SPACE_XS)
        self.badge = QLabel(report.get("dataset", ""))
        self.badge.setObjectName("DatasetBadge")
        self.badge.setWordWrap(True)
        self.badge.setTextFormat(Qt.TextFormat.PlainText)
        form.addWidget(self.badge)
        self.load_button = QPushButton("Open CSV / Parquet...")
        self.load_button.clicked.connect(self._choose_file)
        form.addWidget(self.load_button)
        options = QHBoxLayout()
        self.delimiter = QComboBox()
        for label, value in [("Comma", ","), ("Semicolon", ";"), ("Tab", "\t"), ("Pipe", "|")]:
            self.delimiter.addItem(label, value)
        self.delimiter.setToolTip("CSV delimiter (Parquet ignores this)")
        self.encoding = QComboBox()
        self.encoding.addItem("UTF-8", "utf-8-sig")
        self.encoding.addItem("Windows-1252", "cp1252")
        self.encoding.addItem("UTF-16", "utf-16")
        self.encoding.setToolTip("CSV text encoding")
        options.addWidget(self.delimiter)
        options.addWidget(self.encoding)
        form.addLayout(options)
        self.preview_button = QPushButton("Preview first 20 rows...")
        self.preview_button.clicked.connect(self._preview)
        form.addWidget(self.preview_button)
        form.addWidget(self._section_label("Target column"))
        self.target = QComboBox()
        self.target.currentIndexChanged.connect(self._target_changed)
        form.addWidget(self.target)
        self.positive_label = self._section_label("Positive class")
        form.addWidget(self.positive_label)
        self.positive = QComboBox()
        self.positive.currentIndexChanged.connect(self._configuration_changed)
        form.addWidget(self.positive)
        form.addWidget(self._section_label("Features and column types"))
        help_label = QLabel("Check input features. Uncheck IDs or columns that reveal the answer. Types describe roles; values are not converted.")
        help_label.setWordWrap(True)
        form.addWidget(help_label)
        self.columns = QTableWidget(0, 3)
        self.columns.setHorizontalHeaderLabels(["USE / COLUMN", "TYPE", "MISSING"])
        self.columns.verticalHeader().setVisible(False)
        self.columns.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.columns.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.columns.setMinimumHeight(130)
        self.columns.verticalHeader().setDefaultSectionSize(34)
        self.columns.itemChanged.connect(self._configuration_changed)
        form.addWidget(self.columns, stretch=1)
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self._request_run)
        form.addWidget(self.run_button)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setObjectName("WarningNote")
        form.addWidget(self.status)
        return panel

    def set_dataset(self, view: dict, *, classification: bool, target: str | None = None):
        self.view = view
        self._classification = classification
        self._columns = view["columns"]
        self.badge.setText(f"{view['name']} · {view['rows']} rows · {len(self._columns)} columns")
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItem("Choose target...", None)
        for col in self._columns:
            self.target.addItem(col["name"], col["name"])
        if target is not None:
            self.target.setCurrentIndex(self.target.findData(target))
        self.target.blockSignals(False)
        self.columns.blockSignals(True)
        self.columns.setRowCount(len(self._columns))
        for row, col in enumerate(self._columns):
            item = QTableWidgetItem(col["name"])
            item.setToolTip(col["name"])
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if col["type"] in ("numeric", "categorical", "boolean") else Qt.CheckState.Unchecked)
            self.columns.setItem(row, 0, item)
            role = QComboBox()
            role.addItems(["numeric", "categorical", "boolean", "datetime", "text", "unsupported"])
            role.setCurrentText(col["type"])
            role.currentTextChanged.connect(self._roles_changed)
            self.columns.setCellWidget(row, 1, role)
            self.columns.setItem(row, 2, QTableWidgetItem(str(col["missing"])))
        self.columns.blockSignals(False)
        self.positive_label.setVisible(classification)
        self.positive.setVisible(classification)
        self._target_changed()

    def _target_changed(self, *_):
        target = self.target.currentData()
        self.positive.clear()
        for col in self._columns:
            if col["name"] == target:
                for value in col["classes"]:
                    self.positive.addItem(str(value), value)
        self._roles_changed()

    def _roles_changed(self, *_):
        self.columns.blockSignals(True)
        for row, col in enumerate(self._columns):
            item = self.columns.item(row, 0)
            role = self.columns.cellWidget(row, 1).currentText()
            usable = col["name"] != self.target.currentData() and role in ("numeric", "categorical", "boolean")
            item.setFlags((Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable) if usable else Qt.ItemFlag.NoItemFlags)
            if not usable:
                item.setCheckState(Qt.CheckState.Unchecked)
        self.columns.blockSignals(False)
        self._configuration_changed()

    def request(self) -> dict:
        return {
            "target": self.target.currentData(),
            "features": [col["name"] for row, col in enumerate(self._columns)
                         if self.columns.item(row, 0).checkState() == Qt.CheckState.Checked],
            "overrides": {col["name"]: self.columns.cellWidget(row, 1).currentText()
                          for row, col in enumerate(self._columns)},
            "positive_class": self.positive.currentData() if self._classification else None,
        }

    def _configuration_changed(self, *_):
        self._update_run()
        if hasattr(self, "status") and self._columns:
            request = self.request()
            if request["target"] is None:
                message = "Choose the column you want to predict."
            elif self._classification and self.positive.count() != 2:
                message = "Classification needs a target with exactly two non-missing classes."
            elif not self._classification and request["overrides"][request["target"]] != "numeric":
                message = "Linear regression needs a numeric target."
            elif not request["features"]:
                message = "Select at least one input feature."
            else:
                message = "Settings changed. Run Analysis to update the results."
            self.set_status(message)

    def _update_run(self):
        request = self.request() if self._columns else None
        valid = bool(request and request["target"] and request["features"])
        if valid:
            role = request["overrides"][request["target"]]
            valid = (role in ("numeric", "categorical", "boolean") and self.positive.count() == 2) if self._classification else role == "numeric"
        self.run_button.setEnabled(valid and not self._busy)
        self.run_button.setToolTip("Choose a numeric target for regression, or a two-class target for classification, and at least one feature.")

    def _request_run(self):
        self.run_requested.emit(self.request())

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset", "", "Datasets (*.csv *.parquet)")
        if path:
            self.load_requested.emit(path, self.delimiter.currentData(), self.encoding.currentData())

    def _preview(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Data preview - first 20 rows")
        dialog.resize(850, 500)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(len(self.view["preview"]), len(self._columns))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setHorizontalHeaderLabels([col["name"] for col in self._columns])
        for row, values in enumerate(self.view["preview"]):
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))
        layout.addWidget(table)
        dialog.exec()

    def set_busy(self, busy: bool):
        self._busy = busy
        self.form_panel.setEnabled(not busy)
        self.back_button.setEnabled(not busy)
        self._update_run()

    def set_status(self, message: str):
        self.status.setText(message)

    def show_report(self, report: dict):
        old = self.report_widget
        self.report_widget = self._build_report(report)
        self.body.replaceWidget(old, self.report_widget)
        old.hide()
        old.deleteLater()

    # --- Right: results ------------------------------------------------------

    def _build_report(self, report: dict) -> QWidget:
        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(0, theme.SPACE_SM, theme.SPACE_SM, 0)
        column.setSpacing(theme.SPACE_MD)

        if not report:
            prompt = QLabel("Choose your target and features, then run an analysis to see results here.")
            prompt.setWordWrap(True)
            column.addWidget(prompt)
        else:
            context = QLabel("Dataset: " + report.get("dataset", "") + "\nTarget: " + report.get("target", "")
                             + "\nFeatures: " + report.get("features", "")
                             + ("\nSettings: " + report["settings"] if report.get("settings") else ""))
            context.setTextFormat(Qt.TextFormat.PlainText)
            context.setWordWrap(True)
            column.addWidget(context)
        if "tree" in report:  # the diagram leads the report, in its own horizontal scroller
            column.addWidget(self._tree_view(report["tree"]))
        for spec in report.get("plots", []):  # graphs next
            column.addWidget(PlotWidget(spec))
        for group in report.get("metric_groups", []):
            column.addWidget(self._metric_group(group))
        for bars in report.get("bars", []):
            column.addWidget(self._section_label(bars["title"]))
            column.addWidget(BarList(bars))
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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _tree_view(spec: dict) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        legend = QLabel(
            f"<span style='color:{theme.PLOT_ALT}'>●</span> {spec.get('negative_label', '')} &nbsp;&nbsp; "
            f"<span style='color:{theme.ACCENT}'>●</span> {spec.get('positive_label', '')} &nbsp;&nbsp;·&nbsp;&nbsp; "
            "bar = class mix in the node &nbsp;·&nbsp; left branch = condition true (≤) &nbsp;·&nbsp; "
            "scroll sideways for the full tree"
        )
        legend.setObjectName("FieldLabel")
        legend.setWordWrap(True)
        layout.addWidget(legend)
        tree = TreeWidget(spec)
        scroll = QScrollArea()
        scroll.setWidget(tree)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(tree.height() + 18)
        # Open centered on the root once the scroller has its real width.
        QTimer.singleShot(0, scroll, lambda: scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().maximum() // 2))
        layout.addWidget(scroll)
        return box

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
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        if spec.get("stretch_first"):
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for r, row_values in enumerate(rows):
            for c, text in enumerate(row_values):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
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
