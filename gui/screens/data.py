"""The ordinary dataset workspace: plain data in, import requests out."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import theme


def _label(text: str, name: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName(name)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _table() -> QTableWidget:
    table = QTableWidget()
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(30)
    table.horizontalHeader().setDefaultSectionSize(160)
    table.setMinimumWidth(0)
    table.setMinimumHeight(200)
    table.setMaximumHeight(330)
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return table


class DataScreen(QWidget):
    """Inspect one active ordinary dataset, without editing its statistical roles."""

    load_requested = Signal(str, str, str, str)
    sqlite_requested = Signal()
    workflow_context_changed = Signal(dict)
    workflow_requested = Signal(str, dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.view: dict | None = None
        self._discovery: dict = {}
        self._workflow_initialized = False
        self._updating_workflows = False
        self._workflow_pending = False
        self._workflow_context_checks: dict[str, bool] = {}
        self._busy = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_MD)
        outer.setSpacing(theme.SPACE_SM)

        heading = QHBoxLayout()
        heading.addWidget(_label("Data", "PageTitle"), stretch=1)
        self.load_button = QPushButton("Open CSV / Parquet…")
        self.load_button.setObjectName("PrimaryButton")
        self.load_button.clicked.connect(self._choose_file)
        heading.addWidget(self.load_button)
        self.sqlite_button = QPushButton("Open SQLite…")
        self.sqlite_button.clicked.connect(self.sqlite_requested.emit)
        heading.addWidget(self.sqlite_button)
        outer.addLayout(heading)

        self.import_panel = QWidget()
        import_layout = QVBoxLayout(self.import_panel)
        import_layout.setContentsMargins(0, 0, 0, 0)
        import_layout.setSpacing(theme.SPACE_XS)
        self.import_toggle = QPushButton("▸  CSV import options")
        self.import_toggle.setObjectName("SectionToggle")
        self.import_toggle.setCheckable(True)
        import_layout.addWidget(self.import_toggle)
        self.import_controls = QWidget()
        options = QVBoxLayout(self.import_controls)
        options.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.delimiter = QComboBox()
        for label, value in [("Comma", ","), ("Semicolon", ";"), ("Tab", "\t"), ("Pipe", "|")]:
            self.delimiter.addItem(label, value)
        self.delimiter.setToolTip("CSV delimiter (Parquet ignores this)")
        self.encoding = QComboBox()
        for label, value in [("UTF-8", "utf-8-sig"), ("Windows-1252", "cp1252"), ("UTF-16", "utf-16")]:
            self.encoding.addItem(label, value)
        self.encoding.setToolTip("CSV text encoding")
        delimiter_label = _label("Delimiter", "FieldLabel")
        delimiter_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        delimiter_label.setFixedWidth(64)
        encoding_label = _label("Encoding", "FieldLabel")
        encoding_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        encoding_label.setFixedWidth(64)
        row.addWidget(delimiter_label)
        row.addWidget(self.delimiter, stretch=1)
        row.addWidget(encoding_label)
        row.addWidget(self.encoding, stretch=1)
        options.addLayout(row)
        self.markers = QLineEdit()
        self.markers.setPlaceholderText("Extra missing markers, e.g. ?, -999, n/a")
        self.markers.setToolTip("Applied when opening a file, in addition to the default missing markers.")
        options.addWidget(self.markers)
        import_layout.addWidget(self.import_controls)
        self.import_controls.hide()
        self.import_toggle.toggled.connect(self._toggle_import)
        outer.addWidget(self.import_panel)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(0, theme.SPACE_SM, theme.SPACE_SM, theme.SPACE_LG)
        body.setSpacing(theme.SPACE_MD)
        self.empty = _label("No dataset open", "DatasetBadge")
        body.addWidget(self.empty)
        self.dataset_panel = QWidget()
        dataset = QVBoxLayout(self.dataset_panel)
        dataset.setContentsMargins(0, 0, 0, 0)
        dataset.setSpacing(theme.SPACE_SM)
        self.identity = _label("", "DatasetBadge")
        dataset.addWidget(self.identity)
        self.source = _label("", "HelpText")
        dataset.addWidget(self.source)

        self.workflow_panel = self._build_workflows()
        dataset.addWidget(self.workflow_panel)

        self.checks_toggle = QPushButton("Data checks")
        self.checks_toggle.setObjectName("SectionToggle")
        self.checks_toggle.setCheckable(True)
        self.checks_toggle.toggled.connect(self._toggle_checks)
        dataset.addWidget(self.checks_toggle)
        self.checks = _label("", "HelpText")
        self.checks.hide()
        dataset.addWidget(self.checks)
        dataset.addWidget(_label("Preview · first 20 rows", "FieldLabel"))
        self.preview_table = _table()
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        dataset.addWidget(self.preview_table)
        roles_heading = _label("Columns and source roles", "FieldLabel")
        roles_help = ("These roles describe the active dataset. Adjust roles for an analysis in that recipe; "
                      "those choices do not change this source snapshot.")
        roles_heading.setToolTip(roles_help)
        dataset.addWidget(roles_heading)
        self.columns_table = _table()
        self.columns_table.setToolTip(roles_help)
        self.columns_table.setAccessibleDescription(roles_help)
        dataset.addWidget(self.columns_table)
        body.addWidget(self.dataset_panel)
        body.addStretch()
        self.scroll.setWidget(content)
        outer.addWidget(self.scroll, stretch=1)

        footer = QFrame()
        footer.setObjectName("ActionBar")
        footer_row = QHBoxLayout(footer)
        footer_row.setContentsMargins(0, theme.SPACE_SM, 0, 0)
        self.status = _label("", "HelpText")
        footer_row.addWidget(self.status, stretch=1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(90, 6)
        self.progress.hide()
        footer_row.addWidget(self.progress)
        outer.addWidget(footer)
        self.set_dataset(None)

    def _toggle_import(self, opened: bool) -> None:
        self.import_controls.setVisible(opened)
        self.import_toggle.setText(("▾  " if opened else "▸  ") + "CSV import options")

    def _toggle_checks(self, opened: bool) -> None:
        self.checks.setVisible(opened)

    def _build_workflows(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, theme.SPACE_SM, 0, theme.SPACE_SM)
        layout.setSpacing(theme.SPACE_SM)
        layout.addWidget(_label("Analyze this dataset", "SectionTitle"))
        fields = QGridLayout()
        fields.setColumnStretch(1, 1)
        self.workflow_intent = QComboBox()
        for text, value in (("Explore relationships", "relationships"), ("Predict a number", "regression"),
                            ("Predict a category", "classification"), ("Examine structure", "structure")):
            self.workflow_intent.addItem(text, value)
        self.workflow_intent.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.workflow_intent.setMinimumContentsLength(12)
        task_label = _label("Task", "FieldLabel")
        task_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        fields.addWidget(task_label, 0, 0)
        fields.addWidget(self.workflow_intent, 0, 1)
        self.workflow_target_label = _label("Target", "FieldLabel")
        self.workflow_target_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.workflow_target = QComboBox()
        self.workflow_target.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.workflow_target.setMinimumContentsLength(12)
        fields.addWidget(self.workflow_target_label, 1, 0)
        fields.addWidget(self.workflow_target, 1, 1)
        self.workflow_method = QComboBox()
        self.workflow_method.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.workflow_method.setMinimumContentsLength(12)
        method_label = _label("Method", "FieldLabel")
        method_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        fields.addWidget(method_label, 2, 0)
        fields.addWidget(self.workflow_method, 2, 1)
        layout.addLayout(fields)
        self.workflow_columns_toggle = QPushButton("▸  Input columns")
        self.workflow_columns_toggle.setObjectName("SectionToggle")
        self.workflow_columns_toggle.setCheckable(True)
        layout.addWidget(self.workflow_columns_toggle)
        self.workflow_columns = _table()
        self.workflow_columns.setMinimumHeight(130)
        self.workflow_columns.setMaximumHeight(230)
        self.workflow_columns.setColumnCount(3)
        self.workflow_columns.setHorizontalHeaderLabels(["Use", "Column", "Source role"])
        self.workflow_columns.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.workflow_columns.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.workflow_columns.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.workflow_columns.hide()
        self.workflow_columns_toggle.toggled.connect(self.workflow_columns.setVisible)
        layout.addWidget(self.workflow_columns)
        self.workflow_context_toggle = QPushButton("▸  Evaluation context")
        self.workflow_context_toggle.setObjectName("SectionToggle")
        self.workflow_context_toggle.setCheckable(True)
        layout.addWidget(self.workflow_context_toggle)
        self.workflow_evaluation = QWidget()
        evaluation = QGridLayout(self.workflow_evaluation)
        evaluation.setContentsMargins(0, 0, 0, 0)
        evaluation.setColumnStretch(1, 1)
        self.workflow_group = QComboBox()
        self.workflow_time = QComboBox()
        for row, (label, combo) in enumerate((("Repeated group", self.workflow_group), ("Time order", self.workflow_time))):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.setToolTip("Declare known dependence. Unsupported row-wise workflows will be unavailable.")
            field_label = _label(label, "FieldLabel")
            field_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            evaluation.addWidget(field_label, row, 0)
            evaluation.addWidget(combo, row, 1)
        self.workflow_evaluation.hide()
        self.workflow_context_toggle.toggled.connect(self.workflow_evaluation.setVisible)
        layout.addWidget(self.workflow_evaluation)
        self.workflow_reason = _label("", "HelpText")
        self.workflow_reason.setAccessibleName("Workflow requirements")
        layout.addWidget(self.workflow_reason)
        self.workflow_open = QPushButton("Open workflow")
        self.workflow_open.setObjectName("PrimaryButton")
        self.workflow_open.clicked.connect(self._open_workflow)
        layout.addWidget(self.workflow_open, alignment=Qt.AlignmentFlag.AlignLeft)
        self.workflow_intent.currentIndexChanged.connect(self._intent_changed)
        self.workflow_target.currentIndexChanged.connect(self._workflow_context_edited)
        self.workflow_group.currentIndexChanged.connect(self._workflow_context_edited)
        self.workflow_time.currentIndexChanged.connect(self._workflow_context_edited)
        self.workflow_columns.itemChanged.connect(self._workflow_context_edited)
        self.workflow_method.currentIndexChanged.connect(self._workflow_method_changed)
        return panel

    def workflow_context(self) -> dict:
        """User choices only; applicability is supplied by the coordinator."""
        target = self.workflow_target.currentData() if self.workflow_intent.currentData() in ("regression", "classification") else None
        excluded = {target, self.workflow_group.currentData(), self.workflow_time.currentData()}
        features = [self.workflow_columns.item(row, 1).text() for row in range(self.workflow_columns.rowCount())
                    if self.workflow_columns.item(row, 0).checkState() == Qt.CheckState.Checked
                    and self.workflow_columns.item(row, 1).text() not in excluded]
        return {"target": target, "features": features,
                "group_column": self.workflow_group.currentData(), "time_column": self.workflow_time.currentData()}

    def set_workflows(self, discovery: dict) -> None:
        """Install plain core applicability without importing an analytical module."""
        self._updating_workflows = True
        try:
            self._discovery = deepcopy(discovery)
            self._workflow_pending = False
            columns = discovery.get("columns", [])
            context = discovery.get("context", {})
            if not self._workflow_initialized:
                for combo, empty in ((self.workflow_target, "Select target…"), (self.workflow_group, "Independent rows / not declared"),
                                     (self.workflow_time, "Not declared")):
                    combo.clear()
                    combo.addItem(empty, None)
                    for column in columns:
                        combo.addItem(column["name"], column["name"])
                for combo, key in ((self.workflow_target, "target"), (self.workflow_group, "group_column"),
                                   (self.workflow_time, "time_column")):
                    combo.setCurrentIndex(max(0, combo.findData(context.get(key))))
                self.workflow_columns.setRowCount(len(columns))
                selected = context.get("features")
                for row, column in enumerate(columns):
                    use = QTableWidgetItem()
                    use.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                    use.setCheckState(Qt.CheckState.Checked if (column["name"] in selected if selected is not None
                                                               else column.get("suggested", False)) else Qt.CheckState.Unchecked)
                    use.setToolTip(column.get("reason", ""))
                    self.workflow_columns.setItem(row, 0, use)
                    for index, key in ((1, "name"), (2, "role")):
                        item = QTableWidgetItem(str(column.get(key, "")))
                        item.setToolTip(column.get("reason", ""))
                        self.workflow_columns.setItem(row, index, item)
                self.workflow_columns.setFixedHeight(min(230, max(130, 34 + len(columns) * 30)))
                self._workflow_initialized = True
            reasons = {column["name"]: column.get("reason", "") for column in columns}
            for row in range(self.columns_table.rowCount()):
                name = self.columns_table.item(row, 0).text()
                self.columns_table.item(row, 1).setToolTip(reasons.get(name, ""))
            self.workflow_panel.setVisible(bool(columns) and self.view is not None)
            self._intent_changed()
        finally:
            self._updating_workflows = False

    def workflow_state(self) -> dict:
        """In-memory Data-page choices, separate from saved recipe settings."""
        option = self.workflow_method.currentData() or {}
        return {"intent": self.workflow_intent.currentData(), "target": self.workflow_target.currentData(),
                "features": [self.workflow_columns.item(row, 1).text() for row in range(self.workflow_columns.rowCount())
                             if self.workflow_columns.item(row, 0).checkState() == Qt.CheckState.Checked
                             or self._workflow_context_checks.get(self.workflow_columns.item(row, 1).text(), False)],
                "group_column": self.workflow_group.currentData(), "time_column": self.workflow_time.currentData(),
                "method": option.get("key"), "inputs_expanded": self.workflow_columns_toggle.isChecked(),
                "context_expanded": self.workflow_context_toggle.isChecked()}

    def restore_workflow_state(self, state: dict) -> None:
        if not self._workflow_initialized:
            return
        self._updating_workflows = True
        try:
            self._workflow_context_checks.clear()
            for row in range(self.workflow_columns.rowCount()):
                self.workflow_columns.item(row, 0).setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            for combo, key in ((self.workflow_intent, "intent"), (self.workflow_target, "target"),
                               (self.workflow_group, "group_column"), (self.workflow_time, "time_column")):
                index = combo.findData(state.get(key))
                if index >= 0:
                    combo.setCurrentIndex(index)
            if "features" in state:
                for row in range(self.workflow_columns.rowCount()):
                    name = self.workflow_columns.item(row, 1).text()
                    self.workflow_columns.item(row, 0).setCheckState(
                        Qt.CheckState.Checked if name in state["features"] else Qt.CheckState.Unchecked)
            self.workflow_columns_toggle.setChecked(bool(state.get("inputs_expanded")))
            self.workflow_context_toggle.setChecked(bool(state.get("context_expanded")))
            self._intent_changed()
            for index in range(self.workflow_method.count()):
                if self.workflow_method.itemData(index).get("key") == state.get("method"):
                    self.workflow_method.setCurrentIndex(index)
                    break
        finally:
            self._updating_workflows = False
        self._workflow_context_edited()

    def _intent_changed(self, *_):
        intent = self.workflow_intent.currentData()
        supervised = intent in ("regression", "classification")
        self.workflow_target_label.setVisible(supervised)
        self.workflow_target.setVisible(supervised)
        old = self.workflow_method.currentData()
        self.workflow_method.blockSignals(True)
        self.workflow_method.clear()
        for option in self._discovery.get("workflows", []):
            if option.get("intent") == intent:
                self.workflow_method.addItem(option["label"], option)
        if old:
            for index in range(self.workflow_method.count()):
                candidate = self.workflow_method.itemData(index)
                if candidate.get("key") == old.get("key"):
                    self.workflow_method.setCurrentIndex(index)
                    break
        self.workflow_method.blockSignals(False)
        self._workflow_method_changed()
        if not self._updating_workflows and self._workflow_initialized:
            self._workflow_context_edited()

    def _workflow_context_edited(self, *_):
        if self._updating_workflows or not self._workflow_initialized:
            return
        self._sync_context_columns()
        self.workflow_open.setEnabled(False)
        self._workflow_pending = True
        self.workflow_reason.setText("Checking workflow requirements…")
        self.workflow_context_changed.emit(self.workflow_context())

    def _sync_context_columns(self):
        context = self.workflow_context()
        excluded = {context["target"], context["group_column"], context["time_column"]} - {None}
        self._updating_workflows = True
        try:
            for row in range(self.workflow_columns.rowCount()):
                name = self.workflow_columns.item(row, 1).text()
                use = self.workflow_columns.item(row, 0)
                if name in excluded:
                    if name not in self._workflow_context_checks:
                        self._workflow_context_checks[name] = use.checkState() == Qt.CheckState.Checked
                    use.setCheckState(Qt.CheckState.Unchecked)
                    use.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    use.setToolTip("Selected as target or evaluation context; excluded from input features.")
                elif name in self._workflow_context_checks:
                    use.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                    use.setCheckState(Qt.CheckState.Checked if self._workflow_context_checks.pop(name) else Qt.CheckState.Unchecked)
                    use.setToolTip(self.workflow_columns.item(row, 1).toolTip())
        finally:
            self._updating_workflows = False

    def _workflow_method_changed(self, *_):
        option = self.workflow_method.currentData() or {}
        notes = option.get("warnings", [])
        message = option.get("reason", "Select inputs to see available workflows.")
        self.workflow_reason.setText(message)
        self.workflow_reason.setToolTip("\n\n".join([message, *notes]))
        self.workflow_open.setEnabled(bool(option.get("available")) and not self._busy and not self._workflow_pending)
        self.workflow_columns_toggle.setText(("▾  " if self.workflow_columns_toggle.isChecked() else "▸  ") +
                                             f"Input columns ({len(self.workflow_context()['features'])} selected)")

    def _open_workflow(self):
        option = self.workflow_method.currentData() or {}
        if not self._busy and not self._workflow_pending and option.get("available"):
            context = self.workflow_context()
            context["task"] = option["task"]
            self.workflow_requested.emit(option["key"], context)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset", "", "Datasets (*.csv *.parquet)")
        if path:
            self.load_requested.emit(path, self.delimiter.currentData(), self.encoding.currentData(), self.markers.text())

    def import_options(self) -> dict:
        return {"delimiter": self.delimiter.currentData(), "encoding": self.encoding.currentData(),
                "markers": self.markers.text(), "expanded": self.import_toggle.isChecked()}

    def restore_import_options(self, options: dict) -> None:
        for key in ("delimiter", "encoding"):
            combo = getattr(self, key)
            index = combo.findData(options.get(key))
            if index >= 0:
                combo.setCurrentIndex(index)
        self.markers.setText(options.get("markers", ""))
        self.import_toggle.setChecked(options.get("expanded", False))

    def set_dataset(self, view: dict | None) -> None:
        self.view = deepcopy(view)
        self._workflow_initialized = False
        self._workflow_pending = False
        self._workflow_context_checks.clear()
        self._discovery = {}
        self.workflow_panel.hide()
        self.empty.setVisible(view is None)
        self.dataset_panel.setVisible(view is not None)
        self.checks_toggle.setChecked(False)
        if view is None:
            self.identity.clear()
            self.source.clear()
            self.checks.clear()
            for table in (self.preview_table, self.columns_table):
                table.clear()
                table.setRowCount(0)
                table.setColumnCount(0)
            return
        columns = view.get("columns", [])
        self.identity.setText(f"{view['name']} · {view['rows']} rows · {len(columns)} columns")
        self.source.setText(view.get("source_summary", ""))
        self.source.setVisible(bool(self.source.text()))
        self.source.setToolTip(self.source.text())
        notes = view.get("notes", []) + view.get("issues", [])
        self.checks_toggle.setText(f"Data checks ({len(notes)})")
        self.checks.setText("\n\n".join(notes) if notes else "No import or data-quality issues were flagged.")
        self._fill(self.preview_table, [column["name"] for column in columns], view.get("preview", [])[:20])
        self._fill(self.columns_table, ["Column", "Source role", "Missing"],
                   [[column["name"], column["type"], str(column["missing"])] for column in columns])
        self.columns_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for index in (1, 2):
            self.columns_table.horizontalHeader().setSectionResizeMode(index, QHeaderView.ResizeMode.ResizeToContents)
        if view.get("workflows"):
            self.set_workflows(view["workflows"])

    @staticmethod
    def _fill(table: QTableWidget, headers: list[str], rows: list[list[str]]) -> None:
        table.clear()
        table.setColumnCount(len(headers))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(headers)
        for column, title in enumerate(headers):
            table.horizontalHeaderItem(column).setToolTip(title)
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                table.setItem(row, column, item)
        table.setFixedHeight(min(330, max(200, table.horizontalHeader().height() + len(rows) * 30 + 4)))

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.load_button.setEnabled(not busy)
        self.sqlite_button.setEnabled(not busy)
        self.import_panel.setEnabled(not busy)
        self.workflow_panel.setEnabled(not busy)
        self._workflow_method_changed()
        self.progress.setVisible(busy)

    def set_status(self, message: str) -> None:
        compact = " ".join(message.split())
        self.status.setText(compact if len(compact) <= 180 else compact[:177].rstrip() + "…")
        self.status.setToolTip(message)
        self.status.setAccessibleDescription(message)
