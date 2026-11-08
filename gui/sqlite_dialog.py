"""Plain-data SQLite query dialog; app.py owns all database work."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from gui import theme


class SQLiteDialog(QDialog):
    catalog_requested = Signal(str)
    preview_requested = Signal(str, str, int)
    import_requested = Signal(str, str, int)
    cancel_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open SQLite dataset")
        self.setMinimumSize(620, 570)
        self.resize(800, 700)
        self._busy = False
        self._close_pending = False
        self._preview_identity = None
        self._preview_nonempty = False
        layout = QVBoxLayout(self)
        layout.setSpacing(theme.SPACE_SM)
        path_row = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setPlaceholderText("SQLite database file")
        self.path.setAccessibleName("SQLite database path")
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self._choose_path)
        self.catalog_button = QPushButton("Inspect database")
        self.catalog_button.clicked.connect(self._inspect)
        path_row.addWidget(self.path, 1)
        path_row.addWidget(self.browse_button)
        path_row.addWidget(self.catalog_button)
        layout.addLayout(path_row)

        table_row = QHBoxLayout()
        table_row.addWidget(QLabel("Table or view"))
        self.objects = QComboBox()
        self.objects.setMinimumWidth(0)
        self.objects.currentIndexChanged.connect(self._select_object)
        table_row.addWidget(self.objects, 1)
        layout.addLayout(table_row)
        layout.addWidget(QLabel("Read-only query"))
        self.query = QPlainTextEdit()
        self.query.setPlaceholderText('SELECT * FROM "table" ORDER BY "id"')
        self.query.setAccessibleName("SQLite SELECT query")
        self.query.setMinimumHeight(95)
        self.query.setMaximumHeight(170)
        layout.addWidget(self.query)

        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("Import row cap"))
        self.max_rows = QSpinBox()
        self.max_rows.setRange(1, 1_000_000)
        self.max_rows.setValue(100_000)
        self.max_rows.setGroupSeparatorShown(True)
        self.max_rows.setToolTip("Import fails if the query returns more rows. Use WHERE or an explicit LIMIT to select a smaller dataset.")
        action_row.addWidget(self.max_rows)
        action_row.addStretch()
        self.preview_button = QPushButton("Preview query")
        self.preview_button.clicked.connect(self._preview)
        action_row.addWidget(self.preview_button)
        layout.addLayout(action_row)

        self.preview_label = QLabel("Preview the query before importing its result.")
        self.preview_label.setObjectName("HelpText")
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.preview_label)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(145)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(150)
        layout.addWidget(self.table, 1)
        self.status = QLabel()
        self.status.setObjectName("HelpText")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        footer = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(90, 6)
        self.progress.hide()
        footer.addWidget(self.progress)
        footer.addStretch()
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        self.import_button = QPushButton("Import dataset")
        self.import_button.setObjectName("PrimaryButton")
        self.import_button.clicked.connect(self._import)
        footer.addWidget(self.close_button)
        footer.addWidget(self.import_button)
        layout.addLayout(footer)

        self.path.textChanged.connect(self._changed)
        self.query.textChanged.connect(self._changed)
        self.max_rows.valueChanged.connect(self._changed)
        self._changed()

    def identity(self) -> tuple[str, str, int]:
        return self.path.text().strip(), self.query.toPlainText(), self.max_rows.value()

    def _changed(self) -> None:
        path, query, _ = self.identity()
        self.catalog_button.setEnabled(bool(path) and not self._busy)
        self.preview_button.setEnabled(bool(path and query.strip()) and not self._busy)
        self.import_button.setEnabled(not self._busy and self._preview_nonempty and self._preview_identity == self.identity())

    def _choose_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open SQLite database", "", "SQLite databases (*.sqlite *.sqlite3 *.db);;All files (*)")
        if path:
            self.path.setText(path)
            self._inspect()

    def _inspect(self) -> None:
        if self.path.text().strip() and not self._busy:
            self.catalog_requested.emit(self.path.text().strip())

    def _select_object(self, index: int) -> None:
        name = self.objects.itemData(index)
        if name is not None:
            self.query.setPlainText('SELECT * FROM "' + name.replace('"', '""') + '"')

    def _preview(self) -> None:
        if self.preview_button.isEnabled():
            self.preview_requested.emit(*self.identity())

    def _import(self) -> None:
        if self.import_button.isEnabled():
            self.import_requested.emit(*self.identity())

    def set_catalog(self, catalog: dict) -> None:
        self.objects.blockSignals(True)
        self.objects.clear()
        for item in catalog.get("objects", []):
            self.objects.addItem(f"{item['name']} ({item['kind']})", item["name"])
        self.objects.blockSignals(False)
        if self.objects.count():
            self._select_object(0)
        else:
            self.set_status("This database has no user tables or views. You can still enter a SELECT query.")

    def set_preview(self, view: dict, *, path: str | None = None, query: str | None = None, max_rows: int | None = None) -> None:
        current = self.identity()
        self._preview_identity = (path if path is not None else current[0], query if query is not None else current[1], max_rows if max_rows is not None else current[2])
        self._preview_nonempty = view.get("rows", 0) > 0
        columns = view.get("columns", [])
        rows = view.get("preview", [])
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels([column["name"] for column in columns])
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.table.setItem(row, column, item)
        more = view.get("source", {}).get("has_more", False)
        self.preview_label.setText(f"Preview: {view.get('rows', 0):,} rows · {len(columns)} columns" + (" · more rows exist" if more else "") + ". Import executes the query again and accepts all returned rows up to the cap.")
        self._changed()

    def set_status(self, message: str) -> None:
        self.status.setText(message)

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        for widget in (self.path, self.query, self.objects, self.browse_button, self.max_rows):
            widget.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.close_button.setText("Cancel" if busy else "Close")
        self._changed()
        if not busy and self._close_pending:
            self._close_pending = False
            super().reject()

    def reject(self) -> None:
        if self._busy:
            self._close_pending = True
            self.cancel_requested.emit()
            self.set_status("Cancelling SQLite operation…")
        else:
            super().reject()

    def closeEvent(self, event) -> None:
        if self._busy:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
