"""Shared layout for recipe screens that show one analysis run.

Dumb like every screen: it renders a plain ``report`` dict handed in by app.py
and never imports mlcookbook. The report shape (all values already strings):

    {
        "dataset":  "Diabetes · 442 rows",
        "target":   "progression",                              # optional (unsupervised)
        "features": "age, sex, bmi, ...",
        "settings": "max_depth=3 · min_samples_leaf=5",        # optional
        "tree":     {...},   # optional node-link diagram, see gui/tree_plot.py; leads the report
        "bars":     [{"title": "...", "items": [(label, 0..1, "text"), ...]}],  # optional
        "plots":    [{"title": "...", "x_label": "...", "y_label": "...",
                      "series": [{"kind": "points"|"line", "color": "accent"|"alt"|"muted"|"c0".."c9",
                                  "label": "...", "points": [(x, y), ...]}]}],
        "metric_groups": [{"title": "Test set", "items": [("R²", "0.452"), ...]}],
        "tables":   [{"title": "Coefficients", "headers": [...], "rows": [[...], ...],
                      "stretch_column": 0}],  # optional: this column takes the slack
        "warnings": ["..."],
    }

The form on the left describes the run: the task (``"regression"``,
``"classification"``, or ``"unsupervised"`` for recipes without a target),
target/positive class, feature checkboxes and roles, model settings declared by
the screen subclass as ``params`` specs, and preprocessing choices.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.plot import BarList, PlotWidget
from gui.preprocessing_dialog import DEFAULT_CV_FOLDS, PreprocessingDialog, default_preprocessing, summarize
from gui.tree_plot import TreeWidget

ROLES = ["numeric", "categorical", "boolean", "datetime", "text", "unsupported"]
FEATURE_ROLES = ("numeric", "categorical", "boolean", "datetime")
TARGET_ROLES = {"classification": ("numeric", "categorical", "boolean"), "regression": ("numeric",)}
MAX_CLASSES = 20  # app.py lists a column's classes when it has 2..MAX_CLASSES distinct values
UNCHECKED_ISSUES = {"likely_id", "constant", "all_missing"}  # start these columns unchecked
TASK_LABELS = {"classification": "Classification (predict a class)", "regression": "Regression (predict a number)"}
MULTICLASS = "All classes (multiclass)"


class ReportScreen(QWidget):
    back_requested = Signal()
    load_requested = Signal(str, str, str, str)  # path, delimiter, encoding, extra missing markers
    run_requested = Signal(dict)

    def __init__(
        self,
        title: str,
        engine: dict[str, str],
        report: dict,
        parent=None,
        *,
        task: str = "classification",
        tasks: tuple[str, ...] = (),
        params: list[dict] | None = None,
        feature_roles: tuple[str, ...] = FEATURE_ROLES,
        min_features: int = 1,
        cross_validation: bool = True,
        preprocessing: bool = True,
    ) -> None:
        super().__init__(parent)
        self._fixed_task = task
        self._tasks = tasks
        self._params = params or []
        self._feature_roles = feature_roles
        self._min_features = min_features
        self.preprocessing = default_preprocessing() if preprocessing else None
        self.cv_folds = DEFAULT_CV_FOLDS if cross_validation else None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SPACE_XL, theme.SPACE_LG, theme.SPACE_XL, theme.SPACE_LG)
        outer.setSpacing(theme.SPACE_MD)

        self._busy = False
        self._columns = []
        self.view = {"name": "", "rows": 0, "columns": [], "preview": [], "notes": [], "issues": []}
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
        form = QVBoxLayout(panel)
        form.setContentsMargins(0, theme.SPACE_SM, theme.SPACE_SM, 0)
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
        self.markers = QLineEdit()
        self.markers.setPlaceholderText("Extra missing markers, e.g.  ?, -999, n/a")
        self.markers.setToolTip("Values to read as missing, in addition to empty cells and NA/NaN. Applied when a file is opened.")
        form.addWidget(self.markers)
        self.preview_button = QPushButton("Preview first 20 rows...")
        self.preview_button.clicked.connect(self._preview)
        form.addWidget(self.preview_button)

        self.task_label = self._section_label("Task")
        self.task = QComboBox()
        for task in self._tasks:
            self.task.addItem(TASK_LABELS[task], task)
        self.task.currentIndexChanged.connect(self._task_changed)
        form.addWidget(self.task_label)
        form.addWidget(self.task)
        self.task_label.setVisible(len(self._tasks) > 1)
        self.task.setVisible(len(self._tasks) > 1)

        supervised = self._fixed_task != "unsupervised"
        self.target_label = self._section_label("Target column")
        form.addWidget(self.target_label)
        self.target = QComboBox()
        self.target.currentIndexChanged.connect(self._target_changed)
        form.addWidget(self.target)
        self.target_label.setVisible(supervised)
        self.target.setVisible(supervised)
        self.positive_label = self._section_label("Positive class")
        form.addWidget(self.positive_label)
        self.positive = QComboBox()
        self.positive.currentIndexChanged.connect(self._configuration_changed)
        form.addWidget(self.positive)

        form.addWidget(self._section_label("Features and column types" if supervised else "Columns and types"))
        help_label = QLabel(
            "Check input features. Uncheck IDs or columns that reveal the answer. Types describe roles; values are not converted."
            if supervised else "Check the columns to analyze. Types describe roles; values are not converted."
        )
        help_label.setWordWrap(True)
        form.addWidget(help_label)
        self.columns = QTableWidget(0, 3)
        self.columns.setHorizontalHeaderLabels(["USE / COLUMN", "TYPE", "MISSING"])
        self.columns.verticalHeader().setVisible(False)
        self.columns.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.columns.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.columns.setMinimumHeight(170)
        self.columns.verticalHeader().setDefaultSectionSize(34)
        self.columns.itemChanged.connect(self._configuration_changed)
        form.addWidget(self.columns, stretch=1)

        self.param_widgets: dict[str, QWidget] = {}
        self.param_rows: dict[str, tuple[QLabel, QWidget]] = {}
        if self._params:
            form.addWidget(self._section_label("Model settings"))
            grid = QFormLayout()
            grid.setSpacing(theme.SPACE_XS)
            for spec in self._params:
                widget = self._param_widget(spec)
                label = QLabel(spec["label"])
                label.setToolTip(spec.get("help", ""))
                widget.setToolTip(spec.get("help", ""))
                grid.addRow(label, widget)
                self.param_widgets[spec["key"]] = widget
                self.param_rows[spec["key"]] = (label, widget)
            form.addLayout(grid)

        self.preprocessing_button = QPushButton("Preprocessing & validation...")
        self.preprocessing_button.clicked.connect(self._edit_preprocessing)
        self.preprocessing_summary = QLabel("")
        self.preprocessing_summary.setObjectName("FieldLabel")
        self.preprocessing_summary.setWordWrap(True)
        if self.preprocessing is not None:
            form.addWidget(self._section_label("Preprocessing"))
            form.addWidget(self.preprocessing_button)
            form.addWidget(self.preprocessing_summary)
            self._update_preprocessing_summary()

        self.run_button = QPushButton("Run Analysis")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self._request_run)
        form.addWidget(self.run_button)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setObjectName("WarningNote")
        form.addWidget(self.status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setFixedWidth(384)
        self._task_changed()
        return scroll

    def _param_widget(self, spec: dict) -> QWidget:
        kind = spec["kind"]
        if kind == "int":
            widget = QSpinBox()
            widget.setRange(spec["min"], spec["max"])
            widget.setValue(spec["value"])
            if spec.get("none_at_min"):
                widget.setSpecialValueText(spec["none_at_min"])
            widget.valueChanged.connect(self._configuration_changed)
        elif kind == "float":
            widget = QDoubleSpinBox()
            widget.setDecimals(spec.get("decimals", 3))
            widget.setRange(spec["min"], spec["max"])
            widget.setSingleStep(spec.get("step", 0.1))
            widget.setValue(spec["value"])
            widget.valueChanged.connect(self._configuration_changed)
        else:
            widget = QComboBox()
            for label, value in spec["options"]:
                widget.addItem(label, value)
            widget.currentIndexChanged.connect(self._configuration_changed)
        return widget

    def _param_value(self, spec: dict):
        widget = self.param_widgets[spec["key"]]
        if spec["kind"] == "choice":
            return widget.currentData()
        if spec.get("none_at_min") and widget.value() == widget.minimum():
            return None
        return widget.value()

    def current_task(self) -> str:
        if self._tasks:
            return self.task.currentData()
        return self._fixed_task

    def set_dataset(self, view: dict, *, target: str | None = None):
        self.view = view
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
            issues = col.get("issues", [])
            item.setToolTip("\n".join([col["name"]] + [message for _, message in issues]))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            wanted = col["type"] in self._feature_roles and not UNCHECKED_ISSUES & {kind for kind, _ in issues}
            item.setCheckState(Qt.CheckState.Checked if wanted else Qt.CheckState.Unchecked)
            self.columns.setItem(row, 0, item)
            role = QComboBox()
            role.addItems(ROLES)
            role.setCurrentText(col["type"])
            role.currentTextChanged.connect(self._roles_changed)
            self.columns.setCellWidget(row, 1, role)
            missing = QTableWidgetItem(str(col["missing"]) + ("  ⚠" if issues else ""))
            missing.setToolTip("\n".join(message for _, message in issues))
            self.columns.setItem(row, 2, missing)
        self.columns.blockSignals(False)
        self._target_changed()

    def _column(self, name):
        return next((col for col in self._columns if col["name"] == name), None)

    def _role(self, name) -> str | None:
        for row, col in enumerate(self._columns):
            if col["name"] == name:
                return self.columns.cellWidget(row, 1).currentText()
        return None

    def _target_changed(self, *_):
        target = self.target.currentData()
        col = self._column(target)
        if self._tasks and col is not None:  # pick the natural task; the user can still switch
            continuous = self._role(target) == "numeric" and not col["classes"]
            wanted = "regression" if continuous and "regression" in self._tasks else "classification"
            index = self.task.findData(wanted)
            if index >= 0 and index != self.task.currentIndex():
                self.task.blockSignals(True)
                self.task.setCurrentIndex(index)
                self.task.blockSignals(False)
                self._task_visibility()
        self._fill_positive()
        self._roles_changed()

    def _task_changed(self, *_):
        self._task_visibility()
        self._fill_positive()
        self._configuration_changed()

    def _task_visibility(self):
        task = self.current_task()
        classification = task == "classification"
        self.positive_label.setVisible(classification)
        self.positive.setVisible(classification)
        for spec in self._params:
            visible = task in spec.get("tasks", (task,))
            for widget in self.param_rows[spec["key"]]:
                widget.setVisible(visible)

    def _fill_positive(self):
        self.positive.blockSignals(True)
        self.positive.clear()
        col = self._column(self.target.currentData())
        if col is not None and self.current_task() == "classification":
            classes = col["classes"]
            if len(classes) == 2:
                for value in classes:
                    self.positive.addItem(str(value), value)
                self.positive.setCurrentIndex(1)  # the core's default: the larger class in sorted order
            elif len(classes) > 2:
                self.positive.addItem(MULTICLASS, None)
        self.positive.blockSignals(False)

    def _roles_changed(self, *_):
        self.columns.blockSignals(True)
        target = self.target.currentData() if self._fixed_task != "unsupervised" else None
        for row, col in enumerate(self._columns):
            item = self.columns.item(row, 0)
            role = self.columns.cellWidget(row, 1).currentText()
            usable = col["name"] != target and role in self._feature_roles
            item.setFlags((Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable) if usable else Qt.ItemFlag.NoItemFlags)
            if not usable:
                item.setCheckState(Qt.CheckState.Unchecked)
        self.columns.blockSignals(False)
        self._configuration_changed()

    def request(self) -> dict:
        task = self.current_task()
        return {
            "task": task,
            "target": self.target.currentData() if task != "unsupervised" else None,
            "features": [col["name"] for row, col in enumerate(self._columns)
                         if self.columns.item(row, 0).checkState() == Qt.CheckState.Checked],
            "overrides": {col["name"]: self.columns.cellWidget(row, 1).currentText()
                          for row, col in enumerate(self._columns)},
            "positive_class": self.positive.currentData() if task == "classification" else None,
            "params": {spec["key"]: self._param_value(spec) for spec in self._params
                       if task in spec.get("tasks", (task,))},
            "preprocessing": dict(self.preprocessing) if self.preprocessing is not None else None,
            "cv_folds": self.cv_folds,
        }

    def _problem(self) -> str | None:
        """Why the current selections cannot run, or None if they can."""
        if not self._columns:
            return "Open a file to begin."
        request = self.request()
        task = request["task"]
        if task == "unsupervised":
            if len(request["features"]) < self._min_features:
                return f"Select at least {self._min_features} column(s)."
            return None
        if request["target"] is None:
            return "Choose the column you want to predict."
        role = request["overrides"][request["target"]]
        if role not in TARGET_ROLES[task]:
            return "Regression needs a numeric target." if task == "regression" else \
                "Classification needs a categorical, boolean, or numeric-code target."
        if task == "classification" and self.positive.count() == 0:
            return f"Classification needs a target with 2 to {MAX_CLASSES} classes."
        if not request["features"]:
            return "Select at least one input feature."
        return None

    def _configuration_changed(self, *_):
        self._update_run()
        if hasattr(self, "status") and self._columns:
            self.set_status(self._problem() or "Settings changed. Run Analysis to update the results.")

    def _update_run(self):
        self.run_button.setEnabled(bool(self._columns) and self._problem() is None and not self._busy)
        self.run_button.setToolTip("Choose a target that fits the task (or columns, for unsupervised recipes) "
                                   "and at least one feature.")

    def _request_run(self):
        self.run_requested.emit(self.request())

    def _edit_preprocessing(self):
        dialog = PreprocessingDialog(self.preprocessing, self.cv_folds, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_preprocessing(*dialog.values())

    def set_preprocessing(self, preprocessing: dict, cv_folds: int | None):
        self.preprocessing = dict(preprocessing)
        if self.cv_folds is not None:
            self.cv_folds = cv_folds
        self._update_preprocessing_summary()
        self._configuration_changed()

    def _update_preprocessing_summary(self):
        self.preprocessing_summary.setText(summarize(self.preprocessing, self.cv_folds))

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset", "", "Datasets (*.csv *.parquet)")
        if path:
            self.load_requested.emit(path, self.delimiter.currentData(), self.encoding.currentData(),
                                     self.markers.text())

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
            prompt = QLabel("Choose your target and features, then run an analysis to see results here."
                            if self._fixed_task != "unsupervised" else
                            "Choose the columns to analyze, then run to see results here.")
            prompt.setWordWrap(True)
            column.addWidget(prompt)
            checks = list(self.view.get("notes", [])) + list(self.view.get("issues", []))
            if checks:
                column.addWidget(self._section_label("Data checks"))
                for message in checks:
                    note = QLabel(f"•  {message}")
                    note.setWordWrap(True)
                    column.addWidget(note)
        else:
            lines = ["Dataset: " + report.get("dataset", "")]
            if report.get("target"):
                lines.append("Target: " + report["target"])
            lines.append(("Features: " if report.get("target") else "Columns: ") + report.get("features", ""))
            if report.get("settings"):
                lines.append("Settings: " + report["settings"])
            context = QLabel("\n".join(lines))
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
        if "legend" in spec:
            text = spec["legend"] + " &nbsp;·&nbsp; scroll sideways for the full tree"
        else:
            text = (
                f"<span style='color:{theme.PLOT_ALT}'>●</span> {spec.get('negative_label', '')} &nbsp;&nbsp; "
                f"<span style='color:{theme.ACCENT}'>●</span> {spec.get('positive_label', '')} &nbsp;&nbsp;·&nbsp;&nbsp; "
                "bar = class mix in the node &nbsp;·&nbsp; left branch = condition true (≤) &nbsp;·&nbsp; "
                "scroll sideways for the full tree"
            )
        legend = QLabel(text)
        legend.setTextFormat(Qt.TextFormat.RichText)
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
        stretch = 0 if spec.get("stretch_first") else spec.get("stretch_column")
        if stretch is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
        for r, row_values in enumerate(rows):
            for c, text in enumerate(row_values):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                if c > 0 and c != stretch:
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
