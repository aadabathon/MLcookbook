"""Shared layout for recipe screens that show one analysis run.

Dumb like every screen: it renders a plain ``report`` dict handed in by app.py
and never imports mlcookbook. The report shape (all values already strings):

    {
        "dataset":  "Diabetes · 442 rows",
        "target":   "progression",                              # optional (unsupervised)
        "features": "age, sex, bmi, ...",
        "settings": "max_depth=3 · min_samples_leaf=5",        # optional
        "run_details": ["Created (UTC): ...", "Runtime: ..."],  # optional, formatted provenance
        "summary":  "Mean absolute error on 111 held-out rows: 42.1 ...",  # optional, formatted by app.py
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

from copy import deepcopy
from math import isfinite

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.preprocessing_dialog import (
    CV_OPTIONS,
    DEFAULT_CV_FOLDS,
    PREPROCESSING_OPTIONS,
    PreprocessingDialog,
    default_preprocessing,
    summarize,
)
from gui.recipe_reference import RecipeReferenceDialog
from gui.report_view import METRIC_CAPTIONS as METRIC_CAPTIONS
from gui.report_view import ReportView

ROLES = ["numeric", "categorical", "boolean", "datetime", "text", "unsupported"]
FEATURE_ROLES = ("numeric", "categorical", "boolean", "datetime")
TARGET_ROLES = {"classification": ("numeric", "categorical", "boolean"), "regression": ("numeric",)}
MAX_CLASSES = 20  # app.py lists a column's classes when it has 2..MAX_CLASSES distinct values
UNCHECKED_ISSUES = {"likely_id", "constant", "all_missing"}  # start these columns unchecked
TASK_LABELS = {"classification": "Classification (predict a class)", "regression": "Regression (predict a number)"}
MULTICLASS = "All classes (multiclass)"
# Presentation aliases for fixed metric captions, including saved older reports.
# Unknown labels and user-supplied class/column names retain their exact case.


class ReportScreen(QWidget):
    back_requested = Signal()
    load_requested = Signal(str, str, str, str)  # path, delimiter, encoding, extra missing markers
    run_requested = Signal(dict)
    cancel_requested = Signal()
    configuration_changed = Signal()
    example_requested = Signal()
    workspace_requested = Signal()

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
        settings_title: str = "Model settings",
    ) -> None:
        super().__init__(parent)
        self._fixed_task = task
        self._tasks = tasks
        self._params = params or []
        self._feature_roles = feature_roles
        self._min_features = min_features
        self._settings_title = settings_title
        self.recipe_title = title
        self.preprocessing = default_preprocessing() if preprocessing else None
        self.cv_folds = DEFAULT_CV_FOLDS if cross_validation else None
        self._report = {}
        self._report_request = None
        self._restoring = False
        self._status_message = ""
        self.sections: dict[str, QPushButton] = {}
        self.engine = engine
        self.is_example = False
        self._examples_available = False
        self._using_example = False
        self._has_workspace = False
        self._has_example = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SPACE_LG, theme.SPACE_MD, theme.SPACE_LG, theme.SPACE_MD)
        outer.setSpacing(theme.SPACE_SM)

        self._busy = False
        self._columns = []
        self.view = {"name": "", "rows": 0, "columns": [], "preview": [], "notes": [], "issues": []}
        back = QPushButton("←  Recipes")
        self.back_button = back
        back.setObjectName("BackButton")
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.clicked.connect(self.back_requested)
        navigation = QHBoxLayout()
        navigation.addWidget(back)
        navigation.addStretch()
        about = QPushButton("Recipe reference")
        about.clicked.connect(self._about)
        navigation.addWidget(about)
        outer.addLayout(navigation)

        heading = QLabel(title)
        heading.setObjectName("ScreenTitle")
        outer.addWidget(heading)

        self.input_context = QLabel()
        self.input_context.setObjectName("InputContext")
        self.input_context.setTextFormat(Qt.TextFormat.PlainText)
        self.input_context.setWordWrap(True)
        self.input_context.setMinimumWidth(0)
        self.input_context.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.input_context.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_context, stretch=1)
        self.load_button = QPushButton("Open dataset…")
        self.load_button.setToolTip("Open a CSV or Parquet file. CSV options are in the form below.")
        self.load_button.clicked.connect(self._choose_file)
        input_row.addWidget(self.load_button)
        self.example_button = QPushButton("Run example")
        self.example_button.setToolTip("Run this recipe with its bundled dataset. Your data and choices are kept.")
        self.example_button.clicked.connect(self.example_requested)
        self.example_button.hide()
        self.example_button.setEnabled(False)
        input_row.addWidget(self.example_button)
        self.workspace_button = QPushButton("Use my data")
        self.workspace_button.setToolTip("Return to this recipe with your active dataset and previous choices.")
        self.workspace_button.clicked.connect(self.workspace_requested)
        self.workspace_button.hide()
        self.workspace_button.setEnabled(False)
        input_row.addWidget(self.workspace_button)
        outer.addLayout(input_row)

        actions = self._build_actions()
        body = QHBoxLayout()
        body.setSpacing(theme.SPACE_MD)
        self.body = body
        self.form_panel = self._build_form(report)
        body.addWidget(self.form_panel, stretch=4)
        results = QWidget()
        self.result_layout = QVBoxLayout(results)
        self.result_layout.setContentsMargins(0, 0, 0, 0)
        self.result_state = QLabel("No completed run")
        self.result_state.setObjectName("ResultState")
        self.result_state.setWordWrap(True)
        self.result_state.setTextFormat(Qt.TextFormat.PlainText)
        self.result_layout.addWidget(self.result_state)
        self.warning_button = QPushButton()
        self.warning_button.setObjectName("WarningSummary")
        self.warning_button.clicked.connect(self._show_warnings)
        self.warning_button.hide()
        self.result_layout.addWidget(self.warning_button)
        self.report_widget = self._build_report(report)
        self.result_layout.addWidget(self.report_widget, stretch=1)
        body.addWidget(results, stretch=6)
        outer.addLayout(body, stretch=1)
        outer.addWidget(actions)

    def _build_actions(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("ActionBar")
        row = QHBoxLayout(footer)
        row.setContentsMargins(0, theme.SPACE_SM, 0, 0)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setMinimumWidth(0)
        self.status.setMaximumHeight(60)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.status, stretch=1)
        self.status_details = QPushButton("Details")
        self.status_details.clicked.connect(lambda: self._show_notes("Analysis status", [self._status_message]))
        self.status_details.hide()
        row.addWidget(self.status_details)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(90, 6)
        self.progress.hide()
        row.addWidget(self.progress)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.cancel_button.hide()
        row.addWidget(self.cancel_button)
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.setMinimumWidth(180)
        self.run_button.clicked.connect(self._request_run)
        row.addWidget(self.run_button)
        return footer

    def _expandable(self, key: str, title: str, content: QWidget) -> QWidget:
        title = title.replace("&", "&&")  # literal ampersands, not keyboard mnemonics
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        button = QPushButton("▸  " + title)
        button.setObjectName("SectionToggle")
        button.setCheckable(True)
        button.setProperty("disclosure_title", title)
        button.toggled.connect(content.setVisible)
        button.toggled.connect(self._toggle_disclosure)
        self.sections[key] = button
        layout.addWidget(button)
        layout.addWidget(content)
        content.hide()
        return box

    @Slot(bool)
    def _toggle_disclosure(self, opened: bool):
        button = self.sender()
        button.setText(("▾  " if opened else "▸  ") + button.property("disclosure_title"))

    def _show_notes(self, title: str, messages: list[str]):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(620, 400)
        layout = QVBoxLayout(dialog)
        content = QWidget()
        notes = QVBoxLayout(content)
        for message in messages:
            label = QLabel(message)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            notes.addWidget(label)
        notes.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _about(self):
        dialog = RecipeReferenceDialog(self.recipe_title, self.engine, self)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def _show_warnings(self):
        self._show_notes("Warnings for the displayed result", self._report.get("warnings", []))

    def _data_checks(self):
        self._show_notes("Data checks", self.view.get("notes", []) + self.view.get("issues", [])
                         or ["No import or data-quality issues were flagged."])

    # --- Left: what was run --------------------------------------------------

    def _build_form(self, report: dict) -> QWidget:
        panel = QWidget()
        form = QVBoxLayout(panel)
        form.setContentsMargins(0, theme.SPACE_SM, theme.SPACE_SM, 0)
        form.setSpacing(theme.SPACE_XS)
        import_options = QWidget()
        import_layout = QVBoxLayout(import_options)
        import_layout.setContentsMargins(0, 0, 0, 0)
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
        import_layout.addLayout(options)
        self.markers = QLineEdit()
        self.markers.setPlaceholderText("Extra missing markers, e.g.  ?, -999, n/a")
        self.markers.setToolTip("Values to read as missing, in addition to empty cells and NA/NaN. Applied when a file is opened.")
        import_layout.addWidget(self.markers)
        form.addWidget(self._expandable("import", "CSV import options", import_options))
        self.preview_button = QPushButton("Preview first 20 rows...")
        self.preview_button.clicked.connect(self._preview)
        inspection = QHBoxLayout()
        inspection.addWidget(self.preview_button)
        self.checks_button = QPushButton("Data checks")
        self.checks_button.clicked.connect(self._data_checks)
        inspection.addWidget(self.checks_button)
        form.addLayout(inspection)

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

        self.columns_label = self._section_label("Features and column types" if supervised else "Columns and types")
        form.addWidget(self.columns_label)
        self.columns = QTableWidget(0, 3)
        self._set_columns_help(
            "Check input features. Uncheck IDs or columns that reveal the answer. Types describe roles; values are not converted."
            if supervised else "Check the columns to analyze. Types describe roles; values are not converted."
        )
        self.columns.setHorizontalHeaderLabels(["Use / column", "Type", "Missing"])
        self.columns.verticalHeader().setVisible(False)
        self.columns.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.columns.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.columns.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.columns.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.columns.setColumnWidth(1, 112)
        self.columns.setColumnWidth(2, 68)
        self.columns.setMinimumHeight(170)
        self.columns.verticalHeader().setDefaultSectionSize(34)
        self.columns.itemChanged.connect(self._configuration_changed)
        form.addWidget(self.columns, stretch=1)

        self.param_widgets: dict[str, QWidget] = {}
        self.param_rows: dict[str, tuple[QLabel, QWidget]] = {}
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        self.evaluation_options = QWidget()
        evaluation_layout = QVBoxLayout(self.evaluation_options)
        evaluation_layout.setContentsMargins(0, 0, 0, 0)
        evaluation_layout.setSpacing(theme.SPACE_XS)
        self.evaluation_form = QFormLayout()
        self.evaluation_form.setSpacing(theme.SPACE_XS)
        self.evaluation_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        if self._params:
            grid = QFormLayout()
            self.param_form = grid
            grid.setSpacing(theme.SPACE_XS)
            grid.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            for spec in self._params:
                widget = self._param_widget(spec)
                label = QLabel(spec["label"])
                label.setWordWrap(True)
                label.setToolTip(spec.get("help", ""))
                widget.setToolTip(spec.get("help", ""))
                target_form = self.evaluation_form if spec.get("section") == "evaluation" else grid
                if spec.get("group"):
                    target_form.addRow(self._section_label(spec["group"]))
                target_form.addRow(label, widget)
                self.param_widgets[spec["key"]] = widget
                self.param_rows[spec["key"]] = (label, widget)
            settings_layout.addLayout(grid)

        # Keep the historical section keys for saved experiments. Preprocessing
        # and evaluation stay accessible even when model controls are collapsed.
        if self._params or self.preprocessing is not None:
            settings_section = self._expandable("settings", self._settings_title, settings)
            form.addWidget(settings_section)
            settings_section.setVisible(any(spec.get("section") != "evaluation" for spec in self._params))
        else:
            settings.deleteLater()

        self.preprocessing_button = QPushButton("Preprocessing && validation...", self.evaluation_options)
        self.preprocessing_button.clicked.connect(self._edit_preprocessing)
        self.preprocessing_summary = QLabel("", self.evaluation_options)
        self.preprocessing_summary.setObjectName("HelpText")
        self.preprocessing_summary.setWordWrap(True)
        if self.preprocessing is not None:
            evaluation_layout.addWidget(self._section_label("Preprocessing and evaluation" if self.cv_folds is not None
                                                             else "Preprocessing"))
            evaluation_layout.addWidget(self.preprocessing_button)
            evaluation_layout.addWidget(self.preprocessing_summary)
        else:
            self.preprocessing_button.hide()
            self.preprocessing_summary.hide()
        evaluation_layout.addLayout(self.evaluation_form)
        self.evaluation_summary = QLabel()
        self.evaluation_summary.setObjectName("HelpText")
        self.evaluation_summary.setTextFormat(Qt.TextFormat.PlainText)
        self.evaluation_summary.setWordWrap(True)
        self.evaluation_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        evaluation_layout.addWidget(self.evaluation_summary)
        form.addWidget(self.evaluation_options)
        if self.preprocessing is not None:
            self._update_preprocessing_summary()
        else:
            self._update_evaluation_summary()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(412)
        self._task_changed()
        return scroll

    def _param_widget(self, spec: dict) -> QWidget:
        kind = spec["kind"]
        if kind == "model":
            widget = QComboBox()
            widget.setEditable(True)
            widget.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            widget.lineEdit().setMaxLength(spec.get("max_length", 512))
            widget.setEditText(spec.get("value", ""))
            widget.currentTextChanged.connect(self._configuration_changed)
        elif kind in ("text", "path", "file"):
            widget = QLineEdit(spec.get("value", ""))
            widget.setMaxLength(spec.get("max_length", 4096))
            widget.textChanged.connect(self._configuration_changed)
            if kind in ("path", "file"):
                action = widget.addAction(widget.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
                                          QLineEdit.ActionPosition.TrailingPosition)
                action.setToolTip("Choose local model folder" if kind == "path" else "Choose local GGUF file")

                def choose_directory():
                    path = (QFileDialog.getExistingDirectory(self, "Choose local Hugging Face model folder", widget.text())
                            if kind == "path" else QFileDialog.getOpenFileName(self, "Choose local GGUF file", widget.text(),
                                                                            "GGUF files (*.gguf);;All files (*)")[0])
                    if path:
                        widget.setText(path)

                action.triggered.connect(choose_directory)
        elif kind == "multiline":
            widget = QPlainTextEdit(spec.get("value", ""))
            widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            widget.setMinimumHeight(105)
            widget.setMaximumHeight(150)
            widget.textChanged.connect(self._configuration_changed)
        elif kind == "int":
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
        if spec["kind"] == "model":
            return widget.currentText()
        if spec["kind"] in ("text", "path", "file"):
            return widget.text()
        if spec["kind"] == "multiline":
            return widget.toPlainText()
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
        prefix = "Input: built-in example · " if self.is_example else "Input: "
        self.set_input_context(prefix + view["name"])
        self.preview_button.setEnabled(bool(view["rows"]))
        self.checks_button.setEnabled(bool(self._columns or view.get("notes") or view.get("issues")))
        check_count = len(view.get("notes", [])) + len(view.get("issues", []))
        self.checks_button.setText(f"Data checks ({check_count})")
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
        binary = self.current_task() == "classification" and self.positive.count() == 2
        self.positive_label.setVisible(binary)
        self.positive.setVisible(binary)

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
        if self._restoring:
            return
        self._update_run()
        self._update_result_state()
        self._update_evaluation_summary()
        if hasattr(self, "status") and self._columns:
            self.set_status(self._problem() or "")
        self.configuration_changed.emit()

    def _set_columns_help(self, text: str):
        self.columns.setToolTip(text)
        self.columns.setAccessibleDescription(text)
        self.columns_label.setToolTip(text)

    def set_input_context(self, text: str, *, include_shape: bool = True):
        if include_shape and self._columns:
            text += f" · {self.view['rows']} rows · {len(self._columns)} columns"
        self.input_context.setText(text)

    def set_example_actions(self, *, available: bool, using_example: bool, has_workspace: bool, has_example: bool = False):
        """Render the coordinator's source choice without changing recipe state."""
        self._examples_available = available
        self._using_example = using_example
        self._has_workspace = has_workspace
        self._has_example = has_example
        self._update_example_actions()
        self._update_result_state()

    def _update_example_actions(self):
        example = self._examples_available and not self._using_example
        workspace = self._examples_available and self._using_example and self._has_workspace
        self.example_button.setVisible(example)
        self.example_button.setText("View example" if self._has_example else "Run example")
        self.example_button.setToolTip("Restore the example's settings and last result without rerunning."
                                       if self._has_example else "Run this recipe with its bundled dataset. Your data and choices are kept.")
        self.workspace_button.setVisible(workspace)
        self.example_button.setEnabled(example and not self._busy)
        self.workspace_button.setEnabled(workspace and not self._busy)

    def _update_result_state(self):
        if not hasattr(self, "result_state"):
            return
        if not self._report:
            text = "No completed run"
        elif self.request() != self._report_request:
            text = "Settings changed · showing the previous result"
        else:
            text = "Results match current settings"
        if self.is_example:
            text = "Built-in example · " + text
        self.result_state.setText(text)
        self.result_state.setVisible(not self._report or self.request() != self._report_request)

    def _update_run(self):
        self.run_button.setEnabled(bool(self._columns) and self._problem() is None and not self._busy)
        self.run_button.setToolTip(self._problem() or "Run with the current data and settings.")

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
        self.preprocessing_summary.setText(summarize(self.preprocessing, None))
        self._update_evaluation_summary()

    def _update_evaluation_summary(self):
        if not hasattr(self, "evaluation_summary"):
            return
        parts = []
        if "test_size" in self.param_widgets and "random_state" in self.param_widgets:
            parts.append(f"Test holdout: {self.param_widgets['test_size'].value():.0%} · "
                         f"split/model seed {self.param_widgets['random_state'].value()}.")
        if self.engine.get("evaluation"):
            parts.append(self.engine["evaluation"])
        if self.cv_folds is not None:
            parts.append(f"Training cross-validation: {self.cv_folds} folds." if self.cv_folds else
                         "Training cross-validation is off.")
        self.evaluation_summary.setText("\n".join(parts))
        self.evaluation_summary.setVisible(bool(parts))

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
        self.load_button.setEnabled(not busy)
        self.back_button.setEnabled(not busy)
        self._update_example_actions()
        self.progress.setVisible(busy)
        if not busy:
            self.cancel_button.hide()
        self._update_run()

    def set_cancellable(self, cancellable: bool):
        self.cancel_button.setEnabled(cancellable)
        self.cancel_button.setVisible(cancellable)

    def set_status(self, message: str):
        self._status_message = message
        detailed = len(message) > 180 or "\n" in message
        self.status.setText(" ".join(message.split())[:177].rstrip() + "…" if detailed else message)
        self.status.setToolTip(message)
        self.status_details.setVisible(detailed)

    def show_report(self, report: dict, *, request: dict | None = None):
        self._report = deepcopy(report)
        self._report_request = deepcopy(self.request() if request is None else request) if report else None
        old = self.report_widget
        self.report_widget = self._build_report(report)
        self.result_layout.replaceWidget(old, self.report_widget)
        old.hide()
        old.deleteLater()
        warnings = report.get("warnings", [])
        self.warning_button.setText(f"{len(warnings)} warning{'s' if len(warnings) != 1 else ''} · Review before interpreting")
        self.warning_button.setVisible(bool(warnings))
        self._update_result_state()

    def session_state(self) -> dict:
        """Plain UI state; the coordinator owns its lifetime and dataset association."""
        return deepcopy({
            "request": self.request(),
            "params": {spec["key"]: self._param_value(spec) for spec in self._params},
            "report": self._report, "report_request": self._report_request,
            "delimiter": self.delimiter.currentData(), "encoding": self.encoding.currentData(),
            "markers": self.markers.text(), "status": self._status_message,
            "sections": {key: button.isChecked() for key, button in self.sections.items()},
            "scroll": (self.form_panel.verticalScrollBar().value(), self.report_widget.verticalScrollBar().value()),
            "is_example": self.is_example,
        })

    def prepare_session(self, state: dict) -> dict:
        """Recipe-specific migrations run before saved controls are validated."""
        return state

    def restore_session(self, state: dict):
        self._restoring = True
        request = state["request"]
        self.is_example = state["is_example"]
        for row, col in enumerate(self._columns):
            self.columns.cellWidget(row, 1).setCurrentText(request["overrides"][col["name"]])
        self.target.setCurrentIndex(self.target.findData(request["target"]))
        if self._tasks:
            self.task.setCurrentIndex(self.task.findData(request["task"]))
        self._fill_positive()
        self.positive.setCurrentIndex(self.positive.findData(request["positive_class"]))
        for row, col in enumerate(self._columns):
            self.columns.item(row, 0).setCheckState(Qt.CheckState.Checked if col["name"] in request["features"]
                                                  else Qt.CheckState.Unchecked)
        for spec in self._params:
            value = state["params"][spec["key"]]
            widget = self.param_widgets[spec["key"]]
            if spec["kind"] == "choice":
                widget.setCurrentIndex(widget.findData(value))
            elif spec["kind"] == "model":
                widget.setEditText(value)
            elif spec["kind"] in ("text", "path", "file"):
                widget.setText(value)
            elif spec["kind"] == "multiline":
                widget.setPlainText(value)
            else:
                widget.setValue(widget.minimum() if value is None else value)
        if self.preprocessing is not None:
            self.set_preprocessing(request["preprocessing"], request["cv_folds"])
        self.delimiter.setCurrentIndex(self.delimiter.findData(state["delimiter"]))
        self.encoding.setCurrentIndex(self.encoding.findData(state["encoding"]))
        self.markers.setText(state["markers"])
        for key, opened in state["sections"].items():
            self.sections[key].setChecked(opened)
        self._restoring = False
        self.show_report(state["report"], request=state["report_request"])
        self._update_run()
        self.set_status(state["status"])
        # Wait for restored content to be laid out; canceled when this screen is deleted.
        QTimer.singleShot(0, self, lambda: self._restore_scroll(state["scroll"]))

    def validate_session(self, state: dict):
        """Reject incompatible saved controls before changing the current experiment."""
        def require(condition):
            if not condition:
                raise ValueError("Experiment settings are invalid or incompatible with this recipe.")

        def parameters(values, specs):
            require(isinstance(values, dict) and set(values) == {spec["key"] for spec in specs})
            for spec in specs:
                value = values[spec["key"]]
                if spec["kind"] == "choice":
                    require(value in [item[1] for item in spec["options"]])
                elif spec["kind"] in ("text", "path", "file", "multiline", "model"):
                    require(isinstance(value, str) and len(value) <= spec.get("max_length", 4096))
                elif value is None:
                    require(bool(spec.get("none_at_min")))
                else:
                    require(type(value) in (int, float) and isfinite(value) and spec["min"] <= value <= spec["max"])
                    if spec["kind"] == "int":
                        require(type(value) is int)

        def request_valid(request):
            require(isinstance(request, dict))
            require(set(request) == set(self.request()))
            require(request["task"] in (self._tasks or (self._fixed_task,)))
            names = [column["name"] for column in self._columns]
            require(request["target"] is None or request["target"] in names)
            require(isinstance(request["features"], list) and all(isinstance(name, str) for name in request["features"]))
            require(len(request["features"]) == len(set(request["features"])) and set(request["features"]) <= set(names))
            require(isinstance(request["overrides"], dict) and set(request["overrides"]) == set(names))
            require(all(role in ROLES for role in request["overrides"].values()))
            parameters(request["params"], [spec for spec in self._params
                                          if request["task"] in spec.get("tasks", (request["task"],))])
            if self.preprocessing is not None:
                config = request["preprocessing"]
                require(isinstance(config, dict) and set(config) == set(self.preprocessing))
                for key, _, options in PREPROCESSING_OPTIONS:
                    require(config[key] in [value for _, value in options])
            else:
                require(request["preprocessing"] is None)
            require(request["cv_folds"] in ([value for _, value in CV_OPTIONS] if self.cv_folds is not None else [None]))
            if request["positive_class"] is not None:
                column = self._column(request["target"])
                require(column is not None and request["positive_class"] in column["classes"])

        require(isinstance(state, dict) and set(state) == set(self.session_state()))
        request_valid(state["request"])
        parameters(state["params"], self._params)
        require(isinstance(state["report"], dict))
        if state["report"]:
            request_valid(state["report_request"])
        else:
            require(state["report_request"] is None)
        require(isinstance(state["sections"], dict) and set(state["sections"]) == set(self.sections))
        require(all(type(value) is bool for value in state["sections"].values()))
        require(type(state["is_example"]) is bool)
        require(isinstance(state["status"], str) and isinstance(state["markers"], str))
        require(self.delimiter.findData(state["delimiter"]) >= 0 and self.encoding.findData(state["encoding"]) >= 0)
        require(isinstance(state["scroll"], (list, tuple)) and len(state["scroll"]) == 2
                and all(type(value) is int and value >= 0 for value in state["scroll"]))

    def _restore_scroll(self, positions):
        self.form_panel.verticalScrollBar().setValue(positions[0])
        self.report_widget.verticalScrollBar().setValue(positions[1])

    # --- Right: results ------------------------------------------------------

    def _build_report(self, report: dict) -> QWidget:
        return ReportView(report, checks=self.view.get("notes", []) + self.view.get("issues", []))




    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("FieldLabel")
        label.setWordWrap(True)
        return label
