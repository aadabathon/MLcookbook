"""The ordinary dataset workspace: plain data in, import requests out."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.view: dict | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_MD)
        outer.setSpacing(theme.SPACE_SM)

        heading = QHBoxLayout()
        heading.addWidget(_label("Data", "PageTitle"), stretch=1)
        self.load_button = QPushButton("Open CSV / Parquet…")
        self.load_button.setObjectName("PrimaryButton")
        self.load_button.clicked.connect(self._choose_file)
        heading.addWidget(self.load_button)
        outer.addLayout(heading)
        outer.addWidget(_label("Import a CSV or Parquet file, inspect its columns, then choose an analysis.", "PageSubtitle"))

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
        self.empty = _label("No dataset imported. Open a CSV or Parquet file to inspect it, then choose a recipe.",
                            "ResultSummary")
        body.addWidget(self.empty)
        self.dataset_panel = QWidget()
        dataset = QVBoxLayout(self.dataset_panel)
        dataset.setContentsMargins(0, 0, 0, 0)
        dataset.setSpacing(theme.SPACE_SM)
        self.identity = _label("", "DatasetBadge")
        dataset.addWidget(self.identity)
        self.source = _label("", "HelpText")
        dataset.addWidget(self.source)

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
        dataset.addWidget(_label("Columns and source roles", "FieldLabel"))
        dataset.addWidget(_label("These roles describe the active dataset. Adjust roles for an analysis in that recipe; "
                                 "those choices do not change this source snapshot.", "HelpText"))
        self.columns_table = _table()
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
        self.source.setText(view.get("source_summary") or f"Active dataset: {view['name']}")
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
        self.load_button.setEnabled(not busy)
        self.import_panel.setEnabled(not busy)
        self.progress.setVisible(busy)

    def set_status(self, message: str) -> None:
        compact = " ".join(message.split())
        self.status.setText(compact if len(compact) <= 180 else compact[:177].rstrip() + "…")
        self.status.setToolTip(message)
        self.status.setAccessibleDescription(message)
