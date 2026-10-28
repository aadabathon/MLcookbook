"""Data playground presentation: source settings and previews, without network I/O."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import QDate, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import theme


def _label(text: str = "", name: str = "") -> QLabel:
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
    table.horizontalHeader().setDefaultSectionSize(150)
    table.setMinimumWidth(0)
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return table


class DataPlaygroundScreen(QWidget):
    """Render plain source specs and a separately supplied completed preview."""

    fetch_requested = Signal(str, dict)
    use_requested = Signal()
    cancel_requested = Signal()
    configuration_changed = Signal()

    def __init__(self, specs: list[dict], parent=None) -> None:
        super().__init__(parent)
        if not specs:
            raise ValueError("The data playground needs at least one source.")
        self.specs = deepcopy(specs)
        self._specs = {spec["key"]: spec for spec in self.specs}
        if len(self._specs) != len(self.specs):
            raise ValueError("Data playground source keys must be unique.")
        self._restoring = True
        self._busy = False
        self._cancellable = True
        self._status_message = ""
        self.view: dict | None = None
        self._preview_request: tuple[str, dict] | None = None
        self._forced_stale = False
        self._fields: dict[str, dict[str, QWidget]] = {}
        self._limits: dict[str, QSpinBox] = {}
        self._columns: dict[str, QTableWidget] = {}
        self._column_toggles: dict[str, QPushButton] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_MD)
        outer.setSpacing(theme.SPACE_SM)
        outer.addWidget(_label("Data playground", "PageTitle"))

        source_row = QHBoxLayout()
        source_label = QLabel("Source")
        source_label.setObjectName("FieldLabel")
        source_row.addWidget(source_label)
        self.source_combo = QComboBox()
        self.source_combo.setAccessibleName("Data source")
        self.source_combo.setMinimumWidth(0)
        self.source_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        for spec in self.specs:
            self.source_combo.addItem(spec["title"], spec["key"])
        source_label.setBuddy(self.source_combo)
        source_row.addWidget(self.source_combo, stretch=1)
        self.docs_button = QPushButton("Source documentation")
        self.docs_button.clicked.connect(self._open_documentation)
        source_row.addWidget(self.docs_button)
        outer.addLayout(source_row)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.content.setMinimumWidth(0)
        body = QVBoxLayout(self.content)
        body.setContentsMargins(0, theme.SPACE_SM, theme.SPACE_SM, theme.SPACE_MD)
        body.setSpacing(theme.SPACE_SM)
        self.description = _label("", "HelpText")
        body.addWidget(self.description)

        self.query_panel = QStackedWidget()
        self.query_panel.setMinimumWidth(0)
        self.query_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        for spec in self.specs:
            self.query_panel.addWidget(self._make_source_page(spec))
        body.addWidget(self.query_panel)

        self.result_state = _label("Fetch a preview to inspect data from this source.", "ResultState")
        body.addWidget(self.result_state)
        self.preview_panel = QWidget()
        preview = QVBoxLayout(self.preview_panel)
        preview.setContentsMargins(0, theme.SPACE_SM, 0, 0)
        preview.setSpacing(theme.SPACE_SM)
        self.identity = _label("", "DatasetBadge")
        preview.addWidget(self.identity)
        self.provenance_toggle = QPushButton("▸  Source details")
        self.provenance_toggle.setObjectName("SectionToggle")
        self.provenance_toggle.setCheckable(True)
        self.provenance_toggle.toggled.connect(self._toggle_provenance)
        preview.addWidget(self.provenance_toggle)
        self.provenance = _label("", "HelpText")
        self.provenance.hide()
        preview.addWidget(self.provenance)
        self.warning_button = QPushButton()
        self.warning_button.setObjectName("WarningSummary")
        self.warning_button.setCheckable(True)
        self.warning_button.toggled.connect(self._toggle_warnings)
        preview.addWidget(self.warning_button)
        self.warnings = _label("", "WarningNote")
        self.warnings.hide()
        preview.addWidget(self.warnings)
        preview.addWidget(_label("Preview · first 20 rows", "FieldLabel"))
        self.preview_table = _table()
        self.preview_table.setMinimumHeight(180)
        self.preview_table.setMaximumHeight(330)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        preview.addWidget(self.preview_table)
        body.addWidget(self.preview_panel)
        body.addStretch()
        self.scroll.setWidget(self.content)
        outer.addWidget(self.scroll, stretch=1)

        footer = QFrame()
        footer.setObjectName("ActionBar")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(0, theme.SPACE_SM, 0, 0)
        footer_layout.setSpacing(theme.SPACE_SM)
        status_row = QHBoxLayout()
        self.status = _label("", "HelpText")
        status_row.addWidget(self.status, stretch=1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(90, 6)
        self.progress.hide()
        status_row.addWidget(self.progress)
        footer_layout.addLayout(status_row)
        actions = QHBoxLayout()
        self.fetch_button = QPushButton("Fetch preview")
        self.fetch_button.setObjectName("PrimaryButton")
        self.fetch_button.clicked.connect(self._fetch)
        actions.addWidget(self.fetch_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.hide()
        actions.addWidget(self.cancel_button)
        actions.addStretch()
        self.use_button = QPushButton("Use dataset")
        self.use_button.setToolTip("Make the displayed snapshot the active dataset for recipes.")
        self.use_button.clicked.connect(self._use)
        actions.addWidget(self.use_button)
        footer_layout.addLayout(actions)
        outer.addWidget(footer)

        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self._restoring = False
        self._source_changed(0)
        self.set_preview(None)

    @property
    def source_key(self) -> str:
        return self.source_combo.currentData()

    @property
    def field_widgets(self) -> dict[str, QWidget]:
        return self._fields[self.source_key]

    @property
    def limit(self) -> QSpinBox:
        return self._limits[self.source_key]

    @property
    def columns(self) -> QTableWidget:
        return self._columns[self.source_key]

    @property
    def preview_stale(self) -> bool:
        if self.view is None:
            return False
        return self._forced_stale or self._safe_request() != self._preview_request

    def _make_source_page(self, spec: dict) -> QWidget:
        page = QWidget()
        page.setMinimumWidth(0)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(theme.SPACE_MD)
        form.setVerticalSpacing(theme.SPACE_XS)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        widgets = {}
        for field in spec.get("fields", []):
            widget = self._make_field(field)
            widgets[field["key"]] = widget
            label = QLabel(field["label"])
            label.setObjectName("FieldLabel")
            label.setWordWrap(True)
            label.setMinimumWidth(150)
            label.setMaximumWidth(185)
            label.setBuddy(widget)
            form.addRow(label, widget)
        limit = QSpinBox()
        limit.setRange(1, spec.get("max_limit", 5000))
        limit.setValue(spec.get("default_limit", 500))
        limit.setAccessibleName("Maximum rows")
        limit.valueChanged.connect(self._query_changed)
        limit_label = QLabel("Maximum rows")
        limit_label.setObjectName("FieldLabel")
        limit_label.setBuddy(limit)
        form.addRow(limit_label, limit)
        layout.addLayout(form)
        self._fields[spec["key"]] = widgets
        self._limits[spec["key"]] = limit

        toggle = QPushButton()
        toggle.setObjectName("SectionToggle")
        toggle.setCheckable(True)
        layout.addWidget(toggle)
        table = _table()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Column", "Unit", "Role"])
        table.setRowCount(len(spec.get("columns", [])))
        table.setFixedHeight(min(270, max(140, 38 + table.rowCount() * 30)))
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for index in (1, 2):
            table.horizontalHeader().setSectionResizeMode(index, QHeaderView.ResizeMode.ResizeToContents)
        selected = set(spec.get("default_columns", []))
        for row, column in enumerate(spec.get("columns", [])):
            name = QTableWidgetItem(column.get("label", column["key"]))
            name.setFlags(name.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            name.setCheckState(Qt.CheckState.Checked if column["key"] in selected else Qt.CheckState.Unchecked)
            name.setData(Qt.ItemDataRole.UserRole, column["key"])
            name.setToolTip(column["key"])
            table.setItem(row, 0, name)
            for index, value in enumerate((column.get("unit", ""), column.get("role", "")), start=1):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                table.setItem(row, index, item)
        table.itemChanged.connect(self._query_changed)
        table.hide()
        layout.addWidget(table)
        self._columns[spec["key"]] = table
        self._column_toggles[spec["key"]] = toggle
        toggle.toggled.connect(self._toggle_columns)
        self._update_column_caption(spec["key"])
        return page

    def _make_field(self, field: dict) -> QWidget:
        kind = field.get("kind", "text")
        default = field.get("default")
        if kind == "choice":
            widget = QComboBox()
            widget.setMinimumWidth(0)
            widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            for label, value in field.get("options", []):
                widget.addItem(label, value)
            index = widget.findData(default)
            widget.setCurrentIndex(max(0, index))
            widget.currentIndexChanged.connect(self._query_changed)
        elif kind == "int":
            widget = QSpinBox()
            widget.setRange(field.get("min", -1000000), field.get("max", 1000000))
            widget.setValue(default if default is not None else 0)
            widget.valueChanged.connect(self._query_changed)
        elif kind == "float":
            widget = QDoubleSpinBox()
            widget.setDecimals(field.get("decimals", 4))
            widget.setRange(field.get("min", -1000000), field.get("max", 1000000))
            widget.setValue(default if default is not None else 0)
            widget.valueChanged.connect(self._query_changed)
        elif kind == "date":
            widget = QDateEdit()
            widget.setCalendarPopup(True)
            widget.setDisplayFormat("yyyy-MM-dd")
            widget.setDate(QDate.fromString(str(default), Qt.DateFormat.ISODate))
            widget.dateChanged.connect(self._query_changed)
        elif kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(default))
            widget.toggled.connect(self._query_changed)
        elif kind == "text":
            widget = QLineEdit(str(default or ""))
            if field.get("secret"):
                widget.setEchoMode(QLineEdit.EchoMode.Password)
                widget.setClearButtonEnabled(True)
            widget.setPlaceholderText(field.get("placeholder", ""))
            widget.textChanged.connect(self._query_changed)
        else:
            raise ValueError(f"Unsupported playground field kind: {kind}")
        widget.setMinimumWidth(0)
        widget.setAccessibleName(field["label"])
        widget.setToolTip(field.get("help", ""))
        return widget

    @staticmethod
    def _field_value(widget: QWidget):
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        if isinstance(widget, QDateEdit):
            return widget.date().toString(Qt.DateFormat.ISODate)
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        return widget.text()

    def _source_query(self, key: str, *, secrets: bool) -> dict:
        spec = self._specs[key]
        query = {field["key"]: self._field_value(self._fields[key][field["key"]])
                 for field in spec.get("fields", []) if secrets or not field.get("secret")}
        query["limit"] = self._limits[key].value()
        table = self._columns[key]
        query["columns"] = [table.item(row, 0).data(Qt.ItemDataRole.UserRole)
                            for row in range(table.rowCount())
                            if table.item(row, 0).checkState() == Qt.CheckState.Checked]
        return query

    def request(self, *, require_credentials: bool = False) -> tuple[str, dict]:
        """Return the explicit query; credentials remain transient in this call only."""
        key = self.source_key
        query = self._source_query(key, secrets=True)
        if not query["columns"]:
            raise ValueError("Select at least one column to fetch.")
        for field in self._specs[key].get("fields", []):
            if (field.get("required") and (require_credentials or not field.get("secret"))
                    and not str(query[field["key"]]).strip()):
                raise ValueError(f"Enter {field['label'].lower()} before fetching.")
        if "start_date" in query and "end_date" in query and query["start_date"] > query["end_date"]:
            raise ValueError("The start date must be on or before the end date.")
        return key, query

    def _safe_request(self) -> tuple[str, dict]:
        return self.source_key, self._source_query(self.source_key, secrets=False)

    @Slot(int)
    def _source_changed(self, index: int) -> None:
        self.query_panel.setCurrentIndex(index)
        spec = self._specs[self.source_key]
        self.description.setText(spec.get("description", ""))
        self.docs_button.setEnabled(bool(spec.get("url")))
        self.source_combo.setToolTip(spec["title"])
        self._query_changed()

    @Slot()
    def _query_changed(self, *args) -> None:
        if self._restoring:
            return
        self._forced_stale = False
        self._update_column_caption(self.source_key)
        self._update_preview_state()
        if not self._busy:
            self.set_status("")
        self.configuration_changed.emit()

    def _update_column_caption(self, key: str) -> None:
        table = self._columns[key]
        count = sum(table.item(row, 0).checkState() == Qt.CheckState.Checked for row in range(table.rowCount()))
        toggle = self._column_toggles[key]
        prefix = "▾" if toggle.isChecked() else "▸"
        toggle.setText(f"{prefix}  Columns ({count} selected)")

    @Slot(bool)
    def _toggle_columns(self, opened: bool) -> None:
        key = next((key for key, toggle in self._column_toggles.items() if toggle is self.sender()), self.source_key)
        self._columns[key].setVisible(opened)
        self._update_column_caption(key)
        self.query_panel.updateGeometry()

    @Slot(bool)
    def _toggle_provenance(self, opened: bool) -> None:
        self.provenance.setVisible(opened)
        self.provenance_toggle.setText(("▾" if opened else "▸") + "  Source details")

    @Slot(bool)
    def _toggle_warnings(self, opened: bool) -> None:
        self.warnings.setVisible(opened and bool(self.warnings.text()))

    @Slot()
    def _open_documentation(self) -> None:
        from PySide6.QtCore import QUrl

        url = self._specs[self.source_key].get("url", "")
        if url:
            QDesktopServices.openUrl(QUrl(url))

    @Slot()
    def _fetch(self) -> None:
        if self._busy:
            return
        try:
            source, query = self.request(require_credentials=True)
        except ValueError as exc:
            self.set_status(str(exc))
            return
        self.fetch_requested.emit(source, query)

    @Slot()
    def _use(self) -> None:
        if not self._busy and self.view is not None and not self.preview_stale and self.view.get("rows", 0) > 0:
            self.use_requested.emit()

    @Slot()
    def _cancel(self) -> None:
        if self._busy and self._cancellable:
            self.cancel_requested.emit()
            self.cancel_button.setEnabled(False)
            self.set_status("Cancelling fetch…")

    def _update_preview_state(self) -> None:
        if self.view is None:
            self.result_state.setText("Fetch a preview to inspect data from this source.")
            self.result_state.show()
        elif self.preview_stale:
            self.result_state.setText("Settings changed. Fetch a new preview before using this dataset.")
            self.result_state.show()
        else:
            self.result_state.hide()
        self.fetch_button.setEnabled(not self._busy)
        self.use_button.setEnabled(not self._busy and self.view is not None and not self.preview_stale
                                   and self.view.get("rows", 0) > 0)

    def set_preview(self, view: dict | None, *, stale: bool = False) -> None:
        """Install a completed formatted preview, leaving query controls untouched."""
        self.view = deepcopy(view)
        self._forced_stale = stale
        self.preview_panel.setVisible(view is not None)
        self.provenance_toggle.setChecked(False)
        self.warning_button.setChecked(False)
        if view is None:
            self._preview_request = None
            self.identity.clear()
            self.provenance.clear()
            self.warnings.clear()
            self.preview_table.clear()
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            self._update_preview_state()
            return
        completed = view.get("playground_request")
        if completed is not None:
            source = completed["source"]
            query = deepcopy(completed["query"])
            if source in self._specs:
                for field in self._specs[source].get("fields", []):
                    if field.get("secret"):
                        query.pop(field["key"], None)
            self._preview_request = source, query
        elif not stale or self._preview_request is None:
            self._preview_request = self._safe_request()
        columns = view.get("columns", [])
        self.identity.setText(f"{view['name']} · {view['rows']} rows · {len(columns)} columns")
        details = view.get("provenance", [])
        if isinstance(details, str):
            details = [details]
        summary = view.get("source_summary", "")
        pieces = [summary, *details] if summary and summary != "\n".join(details) else details
        self.provenance.setText("\n\n".join(pieces))
        self.provenance_toggle.setVisible(bool(self.provenance.text()))
        warnings = list(dict.fromkeys(view.get("warnings", []) + view.get("notes", []) + view.get("issues", [])))
        self.warning_button.setText(f"Warnings ({len(warnings)})")
        self.warning_button.setVisible(bool(warnings))
        self.warnings.setText("\n\n".join(warnings))
        table = self.preview_table
        table.clear()
        table.setColumnCount(len(columns))
        rows = view.get("preview", [])[:20]
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels([column["name"] for column in columns])
        for index, column in enumerate(columns):
            details = [column["name"], column.get("unit", ""), column.get("type", column.get("role", ""))]
            table.horizontalHeaderItem(index).setToolTip(" · ".join(value for value in details if value))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                table.setItem(row_index, column_index, item)
        table.setFixedHeight(min(330, max(180, table.horizontalHeader().height() + len(rows) * 30 + 4)))
        self._update_preview_state()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.source_combo.setEnabled(not busy)
        self.query_panel.setEnabled(not busy)
        self.cancel_button.setVisible(busy and self._cancellable)
        self.cancel_button.setEnabled(busy and self._cancellable)
        self.progress.setVisible(busy)
        self._update_preview_state()

    def set_cancellable(self, cancellable: bool) -> None:
        self._cancellable = cancellable
        self.cancel_button.setVisible(self._busy and cancellable)
        self.cancel_button.setEnabled(self._busy and cancellable)

    def set_status(self, message: str) -> None:
        self._status_message = message
        compact = " ".join(message.split())
        self.status.setText(compact if len(compact) <= 180 else compact[:177].rstrip() + "…")
        self.status.setToolTip(message)
        self.status.setAccessibleDescription(message)

    def session_state(self) -> dict:
        """Keep unrun edits in memory; API credentials are deliberately excluded."""
        return {"source": self.source_key,
                "queries": {key: self._source_query(key, secrets=False) for key in self._specs},
                "columns_expanded": {key: toggle.isChecked() for key, toggle in self._column_toggles.items()},
                "scroll": self.scroll.verticalScrollBar().value(),
                "preview_request": deepcopy(self._preview_request), "preview_stale": self.preview_stale}

    def restore_session(self, state: dict) -> None:
        self._restoring = True
        try:
            for key, query in state.get("queries", {}).items():
                if key not in self._specs or not isinstance(query, dict):
                    continue
                for field in self._specs[key].get("fields", []):
                    if field.get("secret") or field["key"] not in query:
                        continue
                    widget = self._fields[key][field["key"]]
                    value = query[field["key"]]
                    if isinstance(widget, QComboBox):
                        index = widget.findData(value)
                        if index >= 0:
                            widget.setCurrentIndex(index)
                    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                        widget.setValue(value)
                    elif isinstance(widget, QDateEdit):
                        date = QDate.fromString(value, Qt.DateFormat.ISODate)
                        if date.isValid():
                            widget.setDate(date)
                    elif isinstance(widget, QCheckBox):
                        widget.setChecked(value)
                    else:
                        widget.setText(value)
                if "limit" in query:
                    self._limits[key].setValue(query["limit"])
                if "columns" in query:
                    selected = set(query["columns"])
                    table = self._columns[key]
                    for row in range(table.rowCount()):
                        item = table.item(row, 0)
                        item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in selected
                                           else Qt.CheckState.Unchecked)
                opened = state.get("columns_expanded", {}).get(key, False)
                self._column_toggles[key].setChecked(opened)
                self._columns[key].setVisible(opened)
                self._update_column_caption(key)
            index = self.source_combo.findData(state.get("source"))
            if index >= 0:
                self.source_combo.setCurrentIndex(index)
            self.query_panel.setCurrentIndex(self.source_combo.currentIndex())
            self.description.setText(self._specs[self.source_key].get("description", ""))
            completed = state.get("preview_request")
            self._preview_request = (completed[0], deepcopy(completed[1])) if completed else None
            self._forced_stale = state.get("preview_stale", False)
            position = state.get("scroll", 0)
            self.scroll.verticalScrollBar().setValue(position)
            QTimer.singleShot(0, self, lambda: self.scroll.verticalScrollBar().setValue(position))
        finally:
            self._restoring = False
        self._update_preview_state()
