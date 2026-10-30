"""Native, plain-data presentation for model inspection and performance research.

The coordinator owns files, inference and capture processes. This module only
edits requests, renders supplied artifacts and emits explicit user actions.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy

from PySide6.QtCore import QAbstractTableModel, QPointF, QRectF, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui import theme


def _text(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else "Unavailable"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _json(value):
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _label(text="", *, name="HelpText"):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setObjectName(name)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


class _RowsModel(QAbstractTableModel):
    """Keep full plain rows while Qt asks only for the visible cell values."""

    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.rows = []

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.rows)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return None
        value = _text(self.rows[index.row()].get(self.columns[index.column()][0]))
        return value[:4000] if role == Qt.ItemDataRole.ToolTipRole else value[:400]

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        return self.columns[section][1] if orientation == Qt.Orientation.Horizontal else str(section + 1)

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()


def _table(columns, parent):
    table = QTableView(parent)
    table.setMinimumWidth(0)
    table.setAlternatingRowColors(True)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    table.horizontalHeader().setStretchLastSection(True)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    model = _RowsModel(columns, table)
    proxy = QSortFilterProxyModel(table)
    proxy.setSourceModel(model)
    proxy.setFilterKeyColumn(-1)
    proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    table.setModel(proxy)
    table.setSortingEnabled(False)
    for col in range(len(columns)):
        table.setColumnWidth(col, 150 if col == 0 else 115)
    return table, model, proxy


def _result_scroll(page, layout):
    """Let results scroll while each tab's execution controls remain visible."""
    content = QWidget()
    content.setMinimumWidth(0)
    body = QVBoxLayout(content)
    body.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea(page)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setMinimumSize(0, 60)
    scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
    scroll.setWidget(content)
    layout.addWidget(scroll, 1)
    return body


class _GraphItem(QGraphicsRectItem):
    def __init__(self, key, title, subtitle, *, group=False):
        super().__init__(0, 0, 240, 64)
        self.key = key
        self.title = title
        self.subtitle = subtitle
        self.group = group
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        self.setToolTip(f"{title}\n{subtitle}")

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(theme.UI_SELECTION if self.isSelected() else theme.CARD))
        painter.setPen(QPen(QColor(theme.UI_ACCENT if self.isSelected() else theme.BORDER), 2 if self.isSelected() else 1))
        painter.drawRoundedRect(self.rect(), 5, 5)
        painter.setFont(QFont(theme.FONT_FAMILY, 10))
        painter.setPen(QColor(theme.TEXT))
        title = painter.fontMetrics().elidedText(self.title, Qt.TextElideMode.ElideMiddle, 218)
        painter.drawText(QRectF(11, 7, 218, 24), Qt.AlignmentFlag.AlignVCenter, title)
        painter.setFont(QFont(theme.FONT_FAMILY, 9))
        painter.setPen(QColor(theme.UI_ACCENT if self.group else theme.TEXT_MUTED))
        detail = painter.fontMetrics().elidedText(self.subtitle, Qt.TextElideMode.ElideRight, 218)
        painter.drawText(QRectF(11, 33, 218, 23), Qt.AlignmentFlag.AlignVCenter, detail)


class _GraphView(QGraphicsView):
    selected = Signal(str)
    group_opened = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(theme.SURFACE))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(0, 170)
        self.scene().selectionChanged.connect(self._selected)
        self.items_by_key = {}

    def _selected(self):
        selected = self.scene().selectedItems()
        if selected and isinstance(selected[0], _GraphItem):
            self.selected.emit(selected[0].key)

    def mouseDoubleClickEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, _GraphItem) and item.group:
            self.group_opened.emit(item.key)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            selected = self.scene().selectedItems()
            if selected and isinstance(selected[0], _GraphItem) and selected[0].group:
                self.group_opened.emit(selected[0].key)
                event.accept()
                return
        if event.key() == Qt.Key.Key_F:
            self.fit()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
            event.accept()
        else:
            super().wheelEvent(event)

    def _zoom(self, factor):
        if 0.08 <= self.transform().m11() * factor <= 4:
            self.scale(factor, factor)

    def zoom_in(self):
        self._zoom(1.2)

    def zoom_out(self):
        self._zoom(1 / 1.2)

    def focus_key(self, key):
        item = self.items_by_key.get(key)
        if item is None:
            return
        if self.transform().m11() < .65:
            self.resetTransform()
        self.centerOn(item.sceneBoundingRect().center())
        scene = self.scene()
        previous = scene.blockSignals(True)
        try:
            scene.clearSelection()
            item.setSelected(True)
        finally:
            scene.blockSignals(previous)

    def fit(self):
        if self.items_by_key:
            self.fitInView(self.scene().itemsBoundingRect().adjusted(-18, -18, 18, 18), Qt.AspectRatioMode.KeepAspectRatio)

    def show_items(self, rows, edges, *, groups=False):
        self.scene().clear()
        self.items_by_key = {}
        for index, row in enumerate(rows):
            item = _GraphItem(str(row["id"]), row["name"], row["detail"], group=groups)
            item.setPos((index % 3) * 295, (index // 3) * 118)
            self.scene().addItem(item)
            self.items_by_key[str(row["id"])] = item
        for source, target in edges:
            a, b = self.items_by_key.get(str(source)), self.items_by_key.get(str(target))
            if a is None or b is None or a is b:
                continue
            start, end = a.sceneBoundingRect().center(), b.sceneBoundingRect().center()
            start += QPointF(0, 32)
            end -= QPointF(0, 32)
            path = QPainterPath(start)
            path.cubicTo(start + QPointF(0, 45), end - QPointF(0, 45), end)
            edge = self.scene().addPath(path, QPen(QColor(theme.TEXT_FAINT), 1))
            edge.setZValue(-1)
            arrow = QPainterPath(end)
            arrow.lineTo(end + QPointF(-4, -7))
            arrow.moveTo(end)
            arrow.lineTo(end + QPointF(4, -7))
            self.scene().addPath(arrow, QPen(QColor(theme.TEXT_FAINT), 1))
        self.scene().setSceneRect(self.scene().itemsBoundingRect().adjusted(-30, -30, 30, 30))
        self.fit()


class _TrialPlot(QWidget):
    """Request wall and backend decode durations, without mixing denominators."""

    def __init__(self, parent=None, *, throughput=False):
        super().__init__(parent)
        self.rows = []
        self.throughput = throughput
        self.setMinimumHeight(130)
        self.setMaximumHeight(180)

    def set_rows(self, rows):
        self.rows = rows
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.SURFACE))
        painter.setFont(QFont(theme.FONT_FAMILY, theme.PLOT_POINT_SIZE))
        painter.setPen(QColor(theme.TEXT_MUTED))
        title = ("Throughput (tok/s) · backend decode / end-to-end" if self.throughput else
                 "Durations (s) · client wall / backend decode")
        painter.drawText(12, 21, title)
        keys = (("decode_tokens_per_second", "end_to_end_tokens_per_second") if self.throughput else
                ("seconds", "decode_seconds"))
        values = [(row.get(keys[0]), row.get(keys[1])) for row in self.rows]
        finite = [v for pair in values for v in pair if type(v) in (float, int) and math.isfinite(v) and v >= 0]
        if not finite:
            painter.drawText(12, 55, "No completed measured trials")
            return
        maximum = max(finite) or 1
        area = QRectF(48, 35, max(1, self.width() - 72), max(1, self.height() - 65))
        painter.drawText(5, 43, f"{maximum:.3g}")
        painter.drawText(20, int(area.bottom()), "0")
        for series, color in ((0, theme.ACCENT), (1, theme.PLOT_ALT)):
            previous = None
            for index, pair in enumerate(values):
                value = pair[series]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    previous = None
                    continue
                point = QPointF(area.left() + (index + 0.5) * area.width() / len(values),
                                area.bottom() - value / maximum * area.height())
                painter.setPen(QPen(QColor(color), 2))
                if previous is not None:
                    painter.drawLine(previous, point)
                painter.setBrush(QColor(color))
                painter.drawEllipse(point, 3, 3)
                previous = point
        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.drawText(QRectF(area.left(), area.bottom() + 5, area.width(), 20),
                         Qt.AlignmentFlag.AlignCenter, "Measured trial order")


class _GraphComparisonDialog(QDialog):
    """Render coordinator-supplied structural differences without recomputation."""

    def __init__(self, result, parent=None, *, model=False):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("GGUF model comparison" if model else "Captured graph comparison")
        self.resize(900, 560)
        layout = QVBoxLayout(self)
        layout.addWidget(_label(_text(result.get("summary", "Recorded structural differences"))))
        columns = [("name", "Tensor" if model else "Operation / tensor"), ("change", "Difference")]
        if not model:
            columns += [("left_op", "First operation"), ("right_op", "Second operation")]
        columns += [("left_shape", "First shape"), ("right_shape", "Second shape"),
                    ("left_dtype", "First type"), ("right_dtype", "Second type")]
        if model:
            columns += [("left_bytes", "First bytes"), ("right_bytes", "Second bytes")]
        self.table, self.model, self.proxy = _table(columns, self)
        self.model.set_rows(deepcopy(result.get("rows", [])))
        layout.addWidget(self.table, 1)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMaximumHeight(150)
        self.detail.setPlainText("\n".join(result.get("warnings", [])))
        self.table.selectionModel().currentRowChanged.connect(self._selected)
        if model:
            self.metadata_table, self.metadata_model, self.metadata_proxy = _table(
                [("key", "Metadata key"), ("left", "First value"), ("right", "Second value")], self)
            self.metadata_model.set_rows(deepcopy(result.get("metadata_changes", [])))
            self.metadata_table.setMaximumHeight(160)
            self.metadata_table.selectionModel().currentRowChanged.connect(self._metadata_selected)
            layout.addWidget(self.metadata_table)
        layout.addWidget(self.detail)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        layout.addWidget(close)

    def _selected(self, index, previous):
        if index.isValid():
            row = self.proxy.mapToSource(index).row()
            self.detail.setPlainText(_json(self.model.rows[row]))

    def _metadata_selected(self, index, previous):
        if index.isValid():
            row = self.metadata_proxy.mapToSource(index).row()
            self.detail.setPlainText(_json(self.metadata_model.rows[row]))


class LLMLabPanel(QWidget):
    """Independent research session, embedded by the existing Local LLMs screen."""

    action_requested = Signal(str, object)
    configuration_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._restoring = False
        self._busy = False
        self._inspection = self._capture = self._benchmark = self._capabilities = None
        self._model_comparison = self._graph_comparison = None
        self._graph_nodes = []
        self._filtered_nodes = []
        self._node_by_id = {}
        self._groups = {}
        self._selected_node = ""
        self._graph_page = 0
        self._connection = {"base_url": "http://127.0.0.1:8080", "model": ""}
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.open_button = QPushButton("Open lab…")
        self.save_button = QPushButton("Save lab…")
        self.export_button = QPushButton("Export JSON…")
        self.probe_button = QPushButton("Inspect server capabilities")
        for button in (self.open_button, self.save_button, self.export_button, self.probe_button):
            toolbar.addWidget(button)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.open_button.clicked.connect(self._open_lab)
        self.save_button.clicked.connect(self._save_lab)
        self.export_button.clicked.connect(self._export_lab)
        self.probe_button.clicked.connect(self._probe)
        self.server_controls = QWidget()
        connection_row = QHBoxLayout(self.server_controls)
        connection_row.setContentsMargins(0, 0, 0, 0)
        self.server_address = QLineEdit()
        self.server_address.setMaxLength(2048)
        self.server_address.setAccessibleName("Research llama.cpp server address")
        self.server_address.setToolTip("An already-running llama-server, for example http://127.0.0.1:8081. "
                                       "Start the server separately. Ollama normally uses port 11434 and the Ollama engine.")
        self.server_model = QLineEdit()
        self.server_model.setMaxLength(512)
        self.server_model.setPlaceholderText("Model already loaded by this server")
        self.server_model.setAccessibleName("Research llama.cpp loaded model")
        self.server_model.setToolTip("The server's model ID or alias, not a GGUF path. Inspect capabilities to discover the loaded model.")
        connection_row.addWidget(_label("Server URL"))
        connection_row.addWidget(self.server_address, 2)
        connection_row.addWidget(_label("Loaded model"))
        connection_row.addWidget(self.server_model, 1)
        layout.addWidget(self.server_controls)
        self.connection_label = _label()
        self.connection_label.setVisible(False)
        layout.addWidget(self.connection_label)
        self.server_address.textChanged.connect(self._connection_edited)
        self.server_model.textChanged.connect(self._connection_edited)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("KitchenInspector")
        self.tabs.setMinimumWidth(0)
        layout.addWidget(self.tabs, 1)
        self._model_page()
        self._graph_page_ui()
        self._performance_page()
        self.warning_button = QPushButton()
        self.warning_button.setObjectName("WarningSummary")
        self.warning_button.setVisible(False)
        self.warning_button.clicked.connect(self._show_warnings)
        layout.addWidget(self.warning_button)
        self.status = _label()
        self.status.setMaximumHeight(42)
        self.status.hide()
        layout.addWidget(self.status)
        self.tabs.currentChanged.connect(self._changed)
        self.set_connection(**self._connection)
        self._changed()

    def _model_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 10, 4, 4)
        row = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Explicitly selected local GGUF")
        self.model_path.setMaxLength(4096)
        self.model_path.setAccessibleName("GGUF model path")
        self.model_path.setToolTip("Open this file directly for tensor inspection or CPU graph capture. Neither action needs an HTTP server.")
        self.inspect_button = QPushButton("Open / inspect GGUF…")
        self.compare_model_button = QPushButton("Compare GGUF…")
        self.compare_model_button.setEnabled(False)
        self.view_model_comparison_button = QPushButton("View comparison")
        self.view_model_comparison_button.hide()
        row.addWidget(self.model_path, 1)
        row.addWidget(self.inspect_button)
        row.addWidget(self.compare_model_button)
        row.addWidget(self.view_model_comparison_button)
        layout.addLayout(row)
        self.inspection_summary = _label("No model inspection")
        layout.addWidget(self.inspection_summary)
        self.tensor_search = QLineEdit()
        self.tensor_search.setPlaceholderText("Filter tensor names, shapes, storage types or groups")
        self.tensor_search.setAccessibleName("Filter tensors")
        layout.addWidget(self.tensor_search)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.tensor_table, self.tensor_model, self.tensor_proxy = _table([
            ("name", "Tensor"), ("shape", "Shape"), ("dtype", "Storage type"),
            ("elements", "Logical elements"), ("bytes", "Stored bytes"), ("file_offset", "File offset"), ("group", "Group")], page)
        self.tensor_table.setColumnWidth(0, 260)
        splitter.addWidget(self.tensor_table)
        lower = QWidget()
        lower_layout = QHBoxLayout(lower)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        self.metadata_table, self.metadata_model, _ = _table([("key", "Metadata"), ("value", "Recorded value")], page)
        self.tensor_detail = QPlainTextEdit()
        self.tensor_detail.setReadOnly(True)
        self.tensor_detail.setPlaceholderText("Select a tensor or metadata entry to inspect its complete recorded fields.")
        lower_layout.addWidget(self.metadata_table, 1)
        lower_layout.addWidget(self.tensor_detail, 1)
        splitter.addWidget(lower)
        splitter.setSizes([310, 150])
        _result_scroll(page, layout).addWidget(splitter, 1)
        self.tabs.addTab(page, "Model inspection")
        self.inspect_button.clicked.connect(self._inspect)
        self.compare_model_button.clicked.connect(self._compare_model)
        self.view_model_comparison_button.clicked.connect(self._view_model_comparison)
        self.model_path.textChanged.connect(self._changed)
        self.tensor_search.textChanged.connect(self.tensor_proxy.setFilterFixedString)
        self.tensor_search.textChanged.connect(self._changed)
        self.tensor_table.selectionModel().currentRowChanged.connect(self._tensor_selected)
        self.metadata_table.selectionModel().currentRowChanged.connect(self._metadata_selected)

    def _graph_page_ui(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 10, 4, 4)
        self.capture_controls = QWidget()
        form = QFormLayout(self.capture_controls)
        form.setContentsMargins(0, 0, 0, 0)
        self.capture_options = QWidget()
        options = QFormLayout(self.capture_options)
        options.setContentsMargins(0, 0, 0, 0)
        helper_row = QHBoxLayout()
        self.helper_path = QLineEdit()
        self.helper_path.setMaxLength(4096)
        self.helper_path.setPlaceholderText("Bundled helper (or select a research helper executable)")
        self.helper_path.setAccessibleName("Graph capture helper executable")
        self.helper_button = QPushButton("Browse…")
        helper_row.addWidget(self.helper_path, 1)
        helper_row.addWidget(self.helper_button)
        options.addRow("Capture helper", helper_row)
        runtime_row = QHBoxLayout()
        self.runtime_dir = QLineEdit()
        self.runtime_dir.setMaxLength(4096)
        self.runtime_dir.setPlaceholderText("Optional · otherwise locate llama-server when Capture is clicked")
        self.runtime_dir.setAccessibleName("llama.cpp runtime directory")
        self.runtime_dir.setToolTip("Folder containing llama-server.exe and its DLLs, not your GGUF model folder. "
                                    "Capture loads the selected GGUF in a separate CPU helper; no running server is required.")
        self.runtime_button = QPushButton("Browse…")
        runtime_row.addWidget(self.runtime_dir, 1)
        runtime_row.addWidget(self.runtime_button)
        options.addRow("Runtime directory", runtime_row)
        form.addRow(self.capture_options)
        self.capture_options.hide()
        self.capture_prompt = QLineEdit("Explain matrix multiplication.")
        self.capture_prompt.setMaxLength(16000)
        form.addRow("Capture prompt", self.capture_prompt)
        limits = QHBoxLayout()
        self.capture_tokens = QSpinBox()
        self.capture_tokens.setRange(1, 16)
        self.capture_tokens.setValue(1)
        self.capture_seconds = QDoubleSpinBox()
        self.capture_seconds.setRange(1, 600)
        self.capture_seconds.setValue(90)
        self.capture_seconds.setSuffix(" s")
        limits.addWidget(_label("Decode steps"))
        limits.addWidget(self.capture_tokens)
        limits.addWidget(_label("Deadline"))
        limits.addWidget(self.capture_seconds)
        self.capture_button = QPushButton("Capture graph")
        self.capture_button.setObjectName("PrimaryButton")
        self.import_graph_button = QPushButton("Import capture…")
        self.capture_options_toggle = QPushButton("Runtime settings")
        self.capture_options_toggle.setCheckable(True)
        self.capture_options_toggle.toggled.connect(self.capture_options.setVisible)
        limits.addWidget(self.capture_options_toggle)
        limits.addWidget(self.capture_button)
        limits.addWidget(self.import_graph_button)
        limits.addStretch()
        form.addRow(limits)
        layout.addWidget(self.capture_controls)
        self.capture_summary = _label("No captured compute graph. Capture uses the explicitly selected model from Model inspection.")
        capture_info = QHBoxLayout()
        capture_info.addWidget(self.capture_summary, 1)
        self.compare_button = QPushButton("Compare phases…")
        self.compare_button.setEnabled(False)
        self.compare_button.clicked.connect(self._compare_graph_request)
        capture_info.addWidget(self.compare_button)
        self.view_graph_comparison_button = QPushButton("View comparison")
        self.view_graph_comparison_button.hide()
        self.view_graph_comparison_button.clicked.connect(self._view_graph_comparison)
        capture_info.addWidget(self.view_graph_comparison_button)
        layout.addLayout(capture_info)
        selectors = QHBoxLayout()
        self.graph_phase = QComboBox()
        self.graph_phase.setAccessibleName("Captured graph phase")
        self.graph_group = QComboBox()
        self.graph_group.setAccessibleName("Graph layer or group")
        self.graph_search = QLineEdit()
        self.graph_search.setPlaceholderText("Find operation or tensor")
        self.graph_search.setAccessibleName("Find graph operation")
        self.fit_button = QPushButton("Fit")
        self.focus_button = QPushButton("Focus selection")
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setAccessibleName("Zoom graph in")
        self.zoom_in_button.setMaximumWidth(35)
        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.setAccessibleName("Zoom graph out")
        self.zoom_out_button.setMaximumWidth(35)
        selectors.addWidget(self.graph_phase)
        selectors.addWidget(self.graph_group)
        selectors.addWidget(self.graph_search, 1)
        selectors.addWidget(self.fit_button)
        selectors.addWidget(self.focus_button)
        selectors.addWidget(self.zoom_out_button)
        selectors.addWidget(self.zoom_in_button)
        layout.addLayout(selectors)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_view = _GraphView()
        self.graph_view.setToolTip(
            "Layer groups follow recorded tensor/operation names; edges follow captured inputs and view aliases. "
            "Enter expands a selected group. Ctrl+wheel zooms; F fits the view.")
        left_layout.addWidget(self.graph_view, 1)
        paging = QHBoxLayout()
        self.previous_button = QPushButton("Previous nodes")
        self.next_button = QPushButton("Next nodes")
        self.graph_coverage = _label()
        paging.addWidget(self.previous_button)
        paging.addWidget(self.next_button)
        paging.addWidget(self.graph_coverage, 1)
        left_layout.addLayout(paging)
        self.node_table, self.node_model, self.node_proxy = _table([
            ("name", "Operation / tensor"), ("op", "Operation"), ("shape", "Shape"), ("dtype", "Type")], page)
        self.node_table.setMinimumHeight(110)
        self.node_table.setMaximumHeight(200)
        left_layout.addWidget(self.node_table)
        splitter.addWidget(left)
        right = QWidget()
        right.setMinimumWidth(0)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        trace_row = QHBoxLayout()
        self.trace_upstream = QPushButton("Inputs")
        self.trace_downstream = QPushButton("Consumers")
        self.trace_clear = QPushButton("Clear trace")
        for button in (self.trace_upstream, self.trace_downstream, self.trace_clear):
            trace_row.addWidget(button)
        right_layout.addLayout(trace_row)
        self.graph_detail = QPlainTextEdit()
        self.graph_detail.setReadOnly(True)
        self.graph_detail.setPlaceholderText("Select a captured operation to inspect shape, strides, bytes and exact dependencies.")
        right_layout.addWidget(self.graph_detail, 1)
        splitter.addWidget(right)
        splitter.setSizes([600, 280])
        _result_scroll(page, layout).addWidget(splitter, 1)
        self.tabs.addTab(page, "Compute graph")
        self._trace_ids = None
        self.helper_button.clicked.connect(self._choose_helper)
        self.runtime_button.clicked.connect(self._choose_runtime)
        self.capture_button.clicked.connect(self._capture_request)
        self.import_graph_button.clicked.connect(self._import_graph)
        self.graph_phase.currentIndexChanged.connect(self._phase_changed)
        self.graph_group.currentIndexChanged.connect(self._filter_graph)
        self.graph_search.textChanged.connect(self._filter_graph)
        self.fit_button.clicked.connect(self.graph_view.fit)
        self.focus_button.clicked.connect(self._focus_selection)
        self.zoom_in_button.clicked.connect(self.graph_view.zoom_in)
        self.zoom_out_button.clicked.connect(self.graph_view.zoom_out)
        self.graph_view.selected.connect(self._graph_selected)
        self.graph_view.group_opened.connect(self._open_group)
        self.node_table.selectionModel().currentRowChanged.connect(self._node_selected)
        self.previous_button.clicked.connect(self._previous_nodes)
        self.next_button.clicked.connect(self._next_nodes)
        self.trace_upstream.clicked.connect(self._trace_inputs)
        self.trace_downstream.clicked.connect(self._trace_consumers)
        self.trace_clear.clicked.connect(self._clear_trace)
        for field in (self.helper_path, self.runtime_dir, self.capture_prompt):
            field.textChanged.connect(self._changed)
        for field in (self.capture_tokens, self.capture_seconds):
            field.valueChanged.connect(self._changed)

    def _performance_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 10, 4, 4)
        self.benchmark_controls = QWidget()
        form = QFormLayout(self.benchmark_controls)
        form.setContentsMargins(0, 0, 0, 0)
        self.benchmark_prompt = QPlainTextEdit("Explain why matrix multiplication is important for language models.")
        self.benchmark_prompt.setMaximumHeight(90)
        form.addRow("Trial prompt", self.benchmark_prompt)
        controls = QHBoxLayout()
        self.repetitions = QSpinBox()
        self.repetitions.setRange(1, 100)
        self.repetitions.setValue(3)
        self.warmups = QSpinBox()
        self.warmups.setRange(0, 10)
        self.warmups.setValue(1)
        self.benchmark_tokens = QSpinBox()
        self.benchmark_tokens.setRange(1, 512)
        self.benchmark_tokens.setValue(64)
        self.benchmark_seconds = QDoubleSpinBox()
        self.benchmark_seconds.setRange(1, 120)
        self.benchmark_seconds.setValue(90)
        self.benchmark_seconds.setSuffix(" s")
        for label, control in (("Trials", self.repetitions), ("Warmups", self.warmups),
                               ("Output cap", self.benchmark_tokens), ("Deadline / trial", self.benchmark_seconds)):
            controls.addWidget(_label(label))
            controls.addWidget(control)
        controls.addStretch()
        form.addRow(controls)
        cache_row = QHBoxLayout()
        self.cache_policy = QComboBox()
        self.cache_policy.addItem("Server default / uncontrolled", "uncontrolled")
        self.cache_policy.addItem("Request no prompt cache", "disabled")
        self.cache_policy.addItem("Request prefix reuse", "reuse")
        self.cache_policy.setToolTip("Requested cache policy; observed cache state is recorded separately by the server when available.")
        self.benchmark_button = QPushButton("Run performance trials")
        self.benchmark_button.setObjectName("PrimaryButton")
        cache_row.addWidget(self.cache_policy, 1)
        cache_row.addWidget(self.benchmark_button)
        form.addRow("Prompt cache", cache_row)
        layout.addWidget(self.benchmark_controls)
        self.benchmark_plan = _label()
        layout.addWidget(self.benchmark_plan)
        self.benchmark_summary = _label("No completed performance trials")
        layout.addWidget(self.benchmark_summary)
        results = _result_scroll(page, layout)
        self.summary_table, self.summary_model, _ = _table([
            ("metric", "Measurement"), ("count", "Samples"), ("median", "Median"),
            ("mean", "Mean"), ("sd", "SD"), ("min", "Minimum"), ("max", "Maximum")], page)
        self.summary_table.setMaximumHeight(135)
        self.summary_table.setVisible(False)
        results.addWidget(self.summary_table)
        plots = QHBoxLayout()
        self.trial_plot = _TrialPlot()
        self.throughput_plot = _TrialPlot(throughput=True)
        plots.addWidget(self.trial_plot, 1)
        plots.addWidget(self.throughput_plot, 1)
        results.addLayout(plots)
        self.trial_table, self.trial_model, _ = _table([
            ("index", "Trial"), ("warmup", "Warmup"), ("status", "Outcome"), ("seconds", "Client wall (s)"),
            ("first_output_seconds", "First output (s)"), ("first_answer_seconds", "First answer (s)"),
            ("prompt_seconds", "Prompt (s)"),
            ("decode_seconds", "Decode (s)"), ("generated_tokens", "Output tokens"),
            ("decode_tokens_per_second", "Decode tok/s"),
            ("end_to_end_tokens_per_second", "End-to-end tok/s")], page)
        self.trial_table.setMinimumHeight(110)
        results.addWidget(self.trial_table, 1)
        self.trial_detail = QPlainTextEdit()
        self.trial_detail.setReadOnly(True)
        self.trial_detail.setMaximumHeight(115)
        self.trial_detail.setPlaceholderText("Select a trial for its output, raw measurements and provenance.")
        results.addWidget(self.trial_detail)
        self.tabs.addTab(page, "Performance")
        self.benchmark_button.clicked.connect(self._benchmark_request)
        self.benchmark_prompt.textChanged.connect(self._changed)
        self.cache_policy.currentIndexChanged.connect(self._changed)
        for field in (self.repetitions, self.warmups, self.benchmark_tokens, self.benchmark_seconds):
            field.valueChanged.connect(self._changed)
        self.trial_table.selectionModel().currentRowChanged.connect(self._trial_selected)
        self._update_plan()

    def _changed(self, *args):
        if hasattr(self, "tabs"):
            performance = self.tabs.currentIndex() == 2
            self.server_controls.setVisible(performance)
            self.probe_button.setVisible(performance)
        if hasattr(self, "benchmark_plan"):
            self._update_plan()
        self._update_recorded_state()
        if not self._restoring:
            self.configuration_changed.emit()

    def _update_recorded_state(self):
        """Display unrun edits without changing the completed artifact."""
        if self._inspection and hasattr(self, "_inspection_summary_base"):
            changed = self.model_path.text().strip() != self._inspection.get("path")
            prefix = "File selection changed; displaying the recorded inventory. " if changed else ""
            self.inspection_summary.setText(prefix + self._inspection_summary_base)
        if self._capture and hasattr(self, "_capture_summary_base"):
            provenance = self._capture.get("provenance", {})
            models = provenance.get("model_files", [])
            changed = (("prompt" in provenance and provenance["prompt"] != self.capture_prompt.text())
                       or ("requested_decode_steps" in provenance and provenance["requested_decode_steps"] != self.capture_tokens.value())
                       or (models and models[0].get("path") != self.model_path.text().strip()))
            prefix = "Controls changed; displaying the recorded capture. " if changed else ""
            self.capture_summary.setText(prefix + self._capture_summary_base)
        if self._benchmark and hasattr(self, "_benchmark_summary_base"):
            settings = self._benchmark.get("settings", {})
            pending = {"prompt": self.benchmark_prompt.toPlainText(), "repetitions": self.repetitions.value(),
                       "warmups": self.warmups.value(), "max_new_tokens": self.benchmark_tokens.value(),
                       "max_seconds": self.benchmark_seconds.value(), "cache_policy": self.cache_policy.currentData()}
            changed = (any(value != settings.get(key) for key, value in pending.items())
                       or any(self._connection[key] != self._benchmark.get(key) for key in ("base_url", "model")))
            prefix = "Controls changed; displayed trials retain recorded settings. " if changed else ""
            self.benchmark_summary.setText(prefix + self._benchmark_summary_base)

    def _update_plan(self):
        count = self.repetitions.value() + self.warmups.value()
        self.benchmark_plan.setText(
            f"{self.warmups.value()} warmup + {self.repetitions.value()} measured requests · "
            f"up to {count * self.benchmark_tokens.value():,} output tokens · "
            f"{self.benchmark_seconds.value():g} s deadline per request. Serial trials.")

    def set_connection(self, base_url, model):
        self._connection = {"base_url": str(base_url), "model": str(model)}
        self.server_address.blockSignals(True)
        self.server_model.blockSignals(True)
        self.server_address.setText(str(base_url))
        self.server_model.setText(str(model))
        self.server_address.blockSignals(False)
        self.server_model.blockSignals(False)
        self.connection_label.setText(f"llama.cpp · {base_url} · {model or 'No loaded model selected'}")
        self._changed()

    def _connection_edited(self, *args):
        self._connection = {"base_url": self.server_address.text().strip(), "model": self.server_model.text().strip()}
        self._changed()

    def set_busy(self, busy):
        self._busy = bool(busy)
        for widget in (self.open_button, self.save_button, self.export_button, self.probe_button,
                       self.inspect_button, self.model_path, self.capture_controls, self.benchmark_controls):
            widget.setEnabled(not self._busy)
        self.server_address.setEnabled(not self._busy)
        self.server_model.setEnabled(not self._busy)
        self.compare_button.setEnabled(not self._busy and bool(self._capture and len(self._capture.get("graphs", [])) >= 2))
        self.compare_model_button.setEnabled(not self._busy and self._inspection is not None)

    def show_status(self, message):
        text = str(message)
        self.status.setToolTip(text)
        self.status.setText(text if len(text) <= 220 else " ".join(text.split())[:217].rstrip() + "…")
        self.status.setVisible(bool(text))

    def _emit(self, action, payload):
        if not self._busy:
            self.action_requested.emit(action, payload)

    def _inspect(self):
        if self._busy:
            return
        path = self.model_path.text().strip()
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Inspect GGUF", "", "GGUF model (*.gguf);;All files (*)")
        if path:
            self.model_path.setText(path)
            self._emit("inspect", {"path": path})

    def _choose_helper(self):
        if self._busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select graph capture helper", "", "Executable (*.exe);;All files (*)")
        if path:
            self.helper_path.setText(path)

    def _compare_model(self):
        if self._busy or self._inspection is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Compare GGUF with inspected model", "", "GGUF model (*.gguf);;All files (*)")
        if path:
            self._emit("compare_model", {"path": path, "baseline": deepcopy(self._inspection)})

    def show_model_comparison(self, result):
        self._model_comparison = deepcopy(result)
        self.view_model_comparison_button.show()
        self._refresh_warnings()
        return self._view_model_comparison()

    def _view_model_comparison(self):
        if self._model_comparison is None:
            return None
        dialog = _GraphComparisonDialog(self._model_comparison, self, model=True)
        dialog.open()
        return dialog

    def _capture_request(self):
        self._emit("capture", {"model_path": self.model_path.text().strip(),
                               "helper_path": self.helper_path.text().strip(),
                               "runtime_dir": self.runtime_dir.text().strip(),
                               "prompt": self.capture_prompt.text(), "max_tokens": self.capture_tokens.value(),
                               "max_seconds": self.capture_seconds.value()})

    def _choose_runtime(self):
        if self._busy:
            return
        path = QFileDialog.getExistingDirectory(self, "Select llama.cpp runtime directory")
        if path:
            self.runtime_dir.setText(path)

    def _import_graph(self):
        if self._busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import captured graph", "", "Captured graph (*.json)")
        if path:
            self._emit("import_graph", {"path": path})

    def _compare_graph_request(self):
        graphs = self._capture.get("graphs", []) if self._capture else []
        if len(graphs) >= 2:
            current = self.graph_phase.currentData()
            self._emit("compare_graph", {"capture": deepcopy(self._capture), "left": 0,
                                         "right": current if current not in (None, 0) else 1})

    def show_graph_comparison(self, result):
        self._graph_comparison = deepcopy(result)
        self.view_graph_comparison_button.show()
        self._refresh_warnings()
        return self._view_graph_comparison()

    def _view_graph_comparison(self):
        if self._graph_comparison is None:
            return None
        dialog = _GraphComparisonDialog(self._graph_comparison, self)
        dialog.open()
        return dialog

    def _benchmark_request(self):
        self._emit("benchmark", {**self._connection, "prompt": self.benchmark_prompt.toPlainText(),
                                 "repetitions": self.repetitions.value(), "warmups": self.warmups.value(),
                                 "max_new_tokens": self.benchmark_tokens.value(),
                                 "max_seconds": self.benchmark_seconds.value(),
                                 "cache_policy": self.cache_policy.currentData()})

    def _probe(self):
        self._emit("probe", deepcopy(self._connection))

    def _open_lab(self):
        if self._busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open LLM research lab", "", "LLM research lab (*.mlcl)")
        if path:
            self._emit("open", {"path": path})

    def _save_lab(self):
        if self._busy:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save LLM research lab", "llm-research.mlcl", "LLM research lab (*.mlcl)")
        if path:
            self._emit("save", {"path": path, "document": self.lab_document()})

    def _export_lab(self):
        if self._busy:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export research JSON", "llm-research.json", "JSON (*.json)")
        if path:
            self._emit("export", {"path": path, "document": self.lab_document()})

    def show_inspection(self, result):
        self._inspection = deepcopy(result)
        self.model_path.setText(str(result.get("path", "")))
        summary = result.get("summary", {})
        self.inspection_summary.setText(" · ".join(f"{key.replace('_', ' ')}: {_text(summary[key])}"
                                                   for key in ("architecture", "tensor_count", "elements", "storage_bytes", "file_count")
                                                   if key in summary) or "Model descriptors recorded")
        self.inspection_summary.setToolTip(_json(summary))
        self.tensor_model.set_rows(self._inspection.get("tensors", []))
        metadata = [{"key": key, "value": value} for key, value in self._inspection.get("metadata", {}).items()]
        metadata.append({"key": "Inspection summary", "value": summary})
        self.metadata_model.set_rows(metadata)
        self.tensor_detail.clear()
        self.compare_model_button.setEnabled(not self._busy)
        self._refresh_warnings()
        self._inspection_summary_base = self.inspection_summary.text()
        self._update_recorded_state()

    def _tensor_selected(self, index, previous):
        if index.isValid():
            row = self.tensor_proxy.mapToSource(index).row()
            self.tensor_detail.setPlainText(_json(self.tensor_model.rows[row]))

    def _metadata_selected(self, index, previous):
        if index.isValid():
            self.tensor_detail.setPlainText(_json(self.metadata_model.rows[index.row()]))

    def show_capture(self, result):
        self._capture = deepcopy(result)
        graphs = result.get("graphs", [])
        old_phase = self.graph_phase.currentData()
        self.graph_phase.blockSignals(True)
        self.graph_phase.clear()
        for index, graph in enumerate(graphs):
            self.graph_phase.addItem(f"{graph.get('phase', 'graph')} · {graph.get('id', index)}", index)
        if old_phase is not None and old_phase < len(graphs):
            self.graph_phase.setCurrentIndex(old_phase)
        self.graph_phase.blockSignals(False)
        provenance = result.get("provenance", {})
        self.capture_summary.setText(f"{len(graphs)} captured graphs · "
                                     f"{provenance.get('attribution', provenance.get('scope', 'See capture provenance'))}")
        self.capture_summary.setToolTip(_json(provenance))
        self._capture_summary_base = self.capture_summary.text()
        self.compare_button.setEnabled(len(graphs) >= 2 and not self._busy)
        self._phase_changed()
        self._refresh_warnings()
        self._update_recorded_state()

    @staticmethod
    def _node_group(node):
        name = str(node.get("name", ""))
        match = re.search(r"(?:blk[.\-]|layers?[.\-])(\d+)", name)
        if not match:
            match = re.search(r"-(\d+)(?:[.\-]|$)", name)
        if match:
            return f"Layer {int(match.group(1))}"
        return "Inputs / weights" if node.get("op") in ("NONE", "none", "input") else "Other operations"

    def _phase_changed(self, *args):
        graphs = self._capture.get("graphs", []) if self._capture else []
        index = self.graph_phase.currentData()
        graph = graphs[index] if isinstance(index, int) and 0 <= index < len(graphs) else {}
        self._graph_nodes = graph.get("nodes", [])
        self._node_by_id = {str(node["id"]): node for node in self._graph_nodes}
        self._groups = {}
        for node in self._graph_nodes:
            self._groups.setdefault(self._node_group(node), []).append(node)
        self.graph_group.blockSignals(True)
        self.graph_group.clear()
        self.graph_group.addItem("Layer overview", "")
        for name, nodes in self._groups.items():
            self.graph_group.addItem(f"{name} ({len(nodes):,})", name)
        self.graph_group.blockSignals(False)
        self._trace_ids = None
        self._selected_node = ""
        self.graph_detail.setPlainText(_json({"graph": {key: value for key, value in graph.items() if key != "nodes"},
                                              "provenance": self._capture.get("provenance", {}) if self._capture else {}}))
        self._filter_graph()

    @staticmethod
    def _raw_inputs(node):
        return [str(value.get("id", value.get("node", ""))) if isinstance(value, dict) else str(value)
                for value in node.get("inputs", [])]

    @classmethod
    def _inputs(cls, node):
        inputs = cls._raw_inputs(node)
        source = node.get("view_source")
        if source is not None and str(source) not in inputs:
            inputs.append(str(source))
        return inputs

    def _filter_graph(self, *args):
        self._graph_page = 0
        group = self.graph_group.currentData()
        query = self.graph_search.text().casefold()
        self._filtered_nodes = [node for node in self._graph_nodes
                                if (not group or self._node_group(node) == group)
                                and (not query or query in _text(node).casefold())
                                and (self._trace_ids is None or str(node["id"]) in self._trace_ids)]
        self.node_model.set_rows(self._filtered_nodes)
        self._render_graph()
        self._changed()

    def _render_graph(self):
        nodes = self._filtered_nodes
        overview = not self.graph_group.currentData() and not self.graph_search.text() and self._trace_ids is None
        if overview:
            group_by_id = {str(node["id"]): self._node_group(node) for node in nodes}
            rows = [{"id": name, "name": name, "detail": f"{len(group):,} recorded nodes · double-click to expand"}
                    for name, group in self._groups.items()]
            edges = {(group_by_id[source], group_by_id[str(node["id"])]) for node in nodes
                     for source in self._inputs(node) if source in group_by_id}
            self.graph_view.show_items(rows, edges, groups=True)
            self.graph_coverage.setText(f"{len(nodes):,} nodes in {len(rows):,} name-based groups. "
                                        "Edges follow captured inputs and view aliases.")
        else:
            start = self._graph_page * 120
            visible = nodes[start:start + 120]
            rows = [{"id": str(node["id"]), "name": str(node.get("name", node["id"])),
                     "detail": f"{node.get('op', 'Unknown')} · {_text(node.get('shape'))} · {node.get('dtype', 'Unknown')}"}
                    for node in visible]
            ids = {str(node["id"]) for node in visible}
            edges = [(source, str(node["id"])) for node in visible for source in self._inputs(node) if source in ids]
            boundary = sum(source not in ids for node in visible for source in self._inputs(node))
            self.graph_view.show_items(rows, edges)
            self.graph_coverage.setText(f"{start + 1 if visible else 0}–{start + len(visible)} of {len(nodes):,} nodes · "
                                        f"{boundary} inputs outside this view. Full dependencies remain in details.")
        self.previous_button.setEnabled(not overview and self._graph_page > 0)
        self.next_button.setEnabled(not overview and (self._graph_page + 1) * 120 < len(nodes))

    def _previous_nodes(self):
        self._graph_page = max(0, self._graph_page - 1)
        self._render_graph()

    def _next_nodes(self):
        if (self._graph_page + 1) * 120 < len(self._filtered_nodes):
            self._graph_page += 1
            self._render_graph()

    def _open_group(self, name):
        index = self.graph_group.findData(name)
        if index >= 0:
            self.graph_group.setCurrentIndex(index)

    def _graph_selected(self, key):
        if key in self._node_by_id:
            self._select_node(key)

    def _node_selected(self, index, previous):
        if index.isValid():
            row = self.node_proxy.mapToSource(index).row()
            self._select_node(str(self.node_model.rows[row]["id"]), focus=True)

    def _focus_selection(self):
        if self._selected_node:
            self._select_node(self._selected_node, focus=True)
        else:
            selected = self.graph_view.scene().selectedItems()
            if selected and isinstance(selected[0], _GraphItem):
                self.graph_view.focus_key(selected[0].key)

    def _select_node(self, key, *, focus=False):
        self._selected_node = key
        node = self._node_by_id[key]
        details = deepcopy(node)
        details["recorded_inputs"] = [{"id": value, "name": self._node_by_id.get(value, {}).get("name", "Outside capture")}
                                      for value in self._raw_inputs(node)]
        alias = node.get("view_source")
        if alias is not None:
            details["recorded_alias_source"] = {"id": str(alias), "name": self._node_by_id.get(str(alias), {}).get("name", "Outside capture")}
        details["recorded_consumers"] = [{"id": str(other["id"]), "name": other.get("name", "")}
                                         for other in self._graph_nodes if key in self._inputs(other)]
        for field in ("strides", "bytes", "backend", "source_reference"):
            if field not in details:
                details[field] = "Unavailable in this capture"
        self.graph_detail.setPlainText(_json(details))
        if focus:
            group = self._node_group(node)
            if not self.graph_group.currentData() and not self.graph_search.text() and self._trace_ids is None:
                self.graph_group.setCurrentIndex(self.graph_group.findData(group))
            for index, candidate in enumerate(self._filtered_nodes):
                if str(candidate["id"]) == key:
                    page = index // 120
                    if page != self._graph_page:
                        self._graph_page = page
                        self._render_graph()
                    self.graph_view.focus_key(key)
                    break
        self._changed()

    def _trace(self, upstream):
        if not self._selected_node:
            return
        found, pending = {self._selected_node}, [self._selected_node]
        if upstream:
            adjacency = {str(node["id"]): self._inputs(node) for node in self._graph_nodes}
        else:
            adjacency = {str(node["id"]): [] for node in self._graph_nodes}
            for node in self._graph_nodes:
                for source in self._inputs(node):
                    if source in adjacency:
                        adjacency[source].append(str(node["id"]))
        while pending:
            for neighbor in adjacency.get(pending.pop(), []):
                if neighbor in self._node_by_id and neighbor not in found:
                    found.add(neighbor)
                    pending.append(neighbor)
        self._trace_ids = found
        self.graph_group.setCurrentIndex(0)
        self._filter_graph()

    def _trace_inputs(self):
        self._trace(True)

    def _trace_consumers(self):
        self._trace(False)

    def _clear_trace(self):
        self._trace_ids = None
        self._filter_graph()

    def show_benchmark(self, result):
        self._benchmark = deepcopy(result)
        rows = self._benchmark.get("trials", [])
        self.trial_model.set_rows(rows)
        measured = [row for row in rows if not row.get("warmup", False) and row.get("status") == "completed"]
        self.trial_plot.set_rows(measured)
        self.throughput_plot.set_rows(measured)
        summary = result.get("summary", {})
        self.benchmark_summary.setText(" · ".join(f"{key.replace('_', ' ')}: {_text(value)}"
                                                  for key, value in summary.items() if not isinstance(value, dict))
                                       or f"{len(rows)} trials recorded")
        self._benchmark_summary_base = self.benchmark_summary.text()
        summaries = [{"metric": key.replace("_", " "), **value}
                     for key, value in summary.get("aggregates", {}).items()]
        self.summary_model.set_rows(summaries)
        self.summary_table.setVisible(bool(summaries))
        self.trial_detail.setPlainText(_json({"settings": result.get("settings", {}),
                                             "provenance": result.get("provenance", {}),
                                             "measurement_definitions": result.get("measurement_definitions", {})}))
        self._refresh_warnings()
        self._update_recorded_state()

    def _trial_selected(self, index, previous):
        if index.isValid():
            self.trial_detail.setPlainText(_json(self.trial_model.rows[index.row()]))

    def show_capabilities(self, result):
        self._capabilities = deepcopy(result)
        self.status.setText("Server capabilities recorded. Open warnings/details to inspect the complete observation.")
        self._refresh_warnings()

    def _refresh_warnings(self):
        warnings = self._warnings()
        count = len(warnings)
        self.warning_button.setText(f"{count} research warning{'s' if count != 1 else ''} / provenance details…")
        self.warning_button.setVisible(bool(count or self._capabilities))

    def _warnings(self):
        return list(dict.fromkeys(str(value) for result in (self._inspection, self._capture, self._benchmark,
                                                            self._capabilities, self._model_comparison, self._graph_comparison)
                                  if result for value in result.get("warnings", [])))

    def _show_warnings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Research warnings and capabilities")
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.resize(700, 500)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText("\n\n".join(self._warnings()) +
                          ("\n\nServer capabilities\n" + _json(self._capabilities) if self._capabilities else ""))
        layout.addWidget(text)
        close = QPushButton("Close")
        close.clicked.connect(dialog.close)
        layout.addWidget(close)
        dialog.open()

    def session_state(self):
        center = self.graph_view.mapToScene(self.graph_view.viewport().rect().center())
        tensor_index = self.tensor_table.currentIndex()
        trial_index = self.trial_table.currentIndex()
        return {"tab": self.tabs.currentIndex(), "model_path": self.model_path.text(),
                "helper_path": self.helper_path.text(), "runtime_dir": self.runtime_dir.text(),
                "capture_prompt": self.capture_prompt.text(),
                "capture_tokens": self.capture_tokens.value(), "capture_seconds": self.capture_seconds.value(),
                "tensor_search": self.tensor_search.text(), "graph_phase": self.graph_phase.currentData(),
                "graph_group": self.graph_group.currentData(), "graph_search": self.graph_search.text(),
                "graph_selected": self._selected_node, "graph_page": self._graph_page,
                "graph_trace": sorted(self._trace_ids) if self._trace_ids is not None else None,
                "graph_view": {"scale": self.graph_view.transform().m11(), "center": [center.x(), center.y()]},
                "tensor_selected": self.tensor_proxy.mapToSource(tensor_index).row() if tensor_index.isValid() else None,
                "trial_selected": trial_index.row() if trial_index.isValid() else None,
                "benchmark_prompt": self.benchmark_prompt.toPlainText(), "repetitions": self.repetitions.value(),
                "warmups": self.warmups.value(), "benchmark_tokens": self.benchmark_tokens.value(),
                "benchmark_seconds": self.benchmark_seconds.value(), "cache_policy": self.cache_policy.currentData(),
                "connection": deepcopy(self._connection)}

    def restore_session(self, state):
        self.validate_session(state)
        restoring = self._restoring
        self._restoring = True
        try:
            for key, field in (("model_path", self.model_path), ("helper_path", self.helper_path),
                               ("runtime_dir", self.runtime_dir),
                               ("capture_prompt", self.capture_prompt), ("tensor_search", self.tensor_search),
                               ("graph_search", self.graph_search)):
                if key in state:
                    field.setText(str(state[key]))
            if "benchmark_prompt" in state:
                self.benchmark_prompt.setPlainText(str(state["benchmark_prompt"]))
            for key, field in (("capture_tokens", self.capture_tokens), ("capture_seconds", self.capture_seconds),
                               ("repetitions", self.repetitions), ("warmups", self.warmups),
                               ("benchmark_tokens", self.benchmark_tokens), ("benchmark_seconds", self.benchmark_seconds)):
                value = state.get(key)
                if type(value) in (int, float) and math.isfinite(value):
                    field.setValue(value if isinstance(field, QDoubleSpinBox) else int(value))
            for key, field in (("graph_phase", self.graph_phase), ("graph_group", self.graph_group),
                               ("cache_policy", self.cache_policy)):
                index = field.findData(state.get(key))
                if index >= 0:
                    field.setCurrentIndex(index)
            trace = state.get("graph_trace")
            self._trace_ids = set(trace) if trace is not None else None
            self._filter_graph()
            selected = state.get("graph_selected")
            if selected in self._node_by_id:
                self._select_node(selected)
            page = state.get("graph_page", 0)
            if type(page) is int and 0 <= page <= len(self._graph_nodes) // 120:
                self._graph_page = page
                self._render_graph()
            view = state.get("graph_view")
            if view is not None:
                self.graph_view.resetTransform()
                self.graph_view.scale(view["scale"], view["scale"])
                self.graph_view.centerOn(*view["center"])
            selected_tensor = state.get("tensor_selected")
            if selected_tensor is not None and selected_tensor < len(self.tensor_model.rows):
                source = self.tensor_model.index(selected_tensor, 0)
                self.tensor_table.setCurrentIndex(self.tensor_proxy.mapFromSource(source))
            selected_trial = state.get("trial_selected")
            if selected_trial is not None and selected_trial < len(self.trial_model.rows):
                self.trial_table.setCurrentIndex(self.trial_table.model().index(selected_trial, 0))
            connection = state.get("connection")
            if isinstance(connection, dict):
                self.set_connection(connection.get("base_url", self._connection["base_url"]),
                                    connection.get("model", self._connection["model"]))
            tab = state.get("tab", 0)
            if type(tab) is int and 0 <= tab < self.tabs.count():
                self.tabs.setCurrentIndex(tab)
        finally:
            self._restoring = restoring

    def lab_document(self):
        return {"format": "mlcookbook-llm-lab", "version": 1, "state": self.session_state(),
                "inspection": deepcopy(self._inspection), "capture": deepcopy(self._capture),
                "benchmark": deepcopy(self._benchmark), "capabilities": deepcopy(self._capabilities),
                "model_comparison": deepcopy(self._model_comparison), "graph_comparison": deepcopy(self._graph_comparison)}

    def restore_document(self, document):
        self.validate_document(document)
        self._restoring = True
        try:
            self._inspection = self._capture = self._benchmark = self._capabilities = None
            self.inspection_summary.setText("No model inspection")
            self.benchmark_summary.setText("No completed performance trials")
            self.tensor_detail.clear()
            self.trial_detail.clear()
            for name in ("_inspection_summary_base", "_capture_summary_base", "_benchmark_summary_base"):
                if hasattr(self, name):
                    delattr(self, name)
            self._model_comparison = self._graph_comparison = None
            self.view_model_comparison_button.hide()
            self.view_graph_comparison_button.hide()
            self.tensor_model.set_rows([])
            self.metadata_model.set_rows([])
            self.trial_model.set_rows([])
            self.trial_plot.set_rows([])
            self.throughput_plot.set_rows([])
            self.summary_model.set_rows([])
            self.summary_table.setVisible(False)
            self.inspection_summary.setText("No model inspection")
            self.compare_model_button.setEnabled(False)
            self.benchmark_summary.setText("No completed performance trials")
            self.tensor_detail.clear()
            self.trial_detail.clear()
            if document.get("capture") is not None:
                self.show_capture(document["capture"])
            else:
                self.graph_phase.clear()
                self.graph_group.clear()
                self._graph_nodes, self._filtered_nodes, self._node_by_id, self._groups = [], [], {}, {}
                self._trace_ids, self._selected_node, self._graph_page = None, "", 0
                self.node_model.set_rows([])
                self.graph_view.show_items([], [])
                self.graph_detail.clear()
                self.graph_coverage.clear()
                self.capture_summary.setText("No captured compute graph. Capture uses the explicitly selected model from Model inspection.")
                self.compare_button.setEnabled(False)
            if document.get("inspection") is not None:
                self.show_inspection(document["inspection"])
            if document.get("benchmark") is not None:
                self.show_benchmark(document["benchmark"])
            if document.get("capabilities") is not None:
                self.show_capabilities(document["capabilities"])
            self._model_comparison = deepcopy(document.get("model_comparison"))
            self._graph_comparison = deepcopy(document.get("graph_comparison"))
            self.view_model_comparison_button.setVisible(self._model_comparison is not None)
            self.view_graph_comparison_button.setVisible(self._graph_comparison is not None)
            self.restore_session(document.get("state", {}))
            self._refresh_warnings()
        finally:
            self._restoring = False

    @staticmethod
    def validate_session(state):
        """Reject unrepresentable settings before replacing the current view."""
        if not isinstance(state, dict):
            raise ValueError("Research settings must be an object.")
        limits = {"tab": (0, 2, True), "capture_tokens": (1, 16, True), "capture_seconds": (1, 600, False),
                  "repetitions": (1, 100, True), "warmups": (0, 10, True), "benchmark_tokens": (1, 512, True),
                  "benchmark_seconds": (1, 120, False), "graph_page": (0, 1_000_000, True)}
        for key, (minimum, maximum, integer) in limits.items():
            if key not in state:
                continue
            value = state[key]
            if (type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value)
                    or not minimum <= value <= maximum):
                raise ValueError(f"Research setting {key} is outside its supported range.")
        string_limits = {"model_path": 4096, "helper_path": 4096, "runtime_dir": 4096,
                         "capture_prompt": 16000, "benchmark_prompt": 16000,
                         "tensor_search": 16000, "graph_search": 16000, "graph_selected": 4096}
        for key, maximum in string_limits.items():
            if key in state and (not isinstance(state[key], str) or len(state[key]) > maximum):
                raise ValueError(f"Invalid research setting: {key}.")
        allowed = set(limits) | set(string_limits) | {
            "cache_policy", "graph_phase", "graph_group", "connection", "graph_trace", "graph_view",
            "tensor_selected", "trial_selected",
        }
        if set(state) - allowed:
            raise ValueError("Research settings contain unsupported fields.")
        if "cache_policy" in state and state["cache_policy"] not in ("uncontrolled", "disabled", "reuse"):
            raise ValueError("Unsupported prompt-cache policy.")
        if "graph_phase" in state and state["graph_phase"] is not None:
            if type(state["graph_phase"]) is not int or state["graph_phase"] < 0:
                raise ValueError("Invalid graph phase selection.")
        if "graph_group" in state and state["graph_group"] is not None and not isinstance(state["graph_group"], str):
            raise ValueError("Invalid graph group selection.")
        for key, maximum in (("tensor_selected", 1_000_000), ("trial_selected", 110)):
            value = state.get(key)
            if value is not None and (type(value) is not int or not 0 <= value <= maximum):
                raise ValueError(f"Invalid research row selection: {key}.")
        trace = state.get("graph_trace")
        if trace is not None and (not isinstance(trace, list) or len(trace) > 100_000
                                  or any(not isinstance(value, str) or len(value) > 4096 for value in trace)):
            raise ValueError("Invalid graph trace selection.")
        view = state.get("graph_view")
        if view is not None:
            if (not isinstance(view, dict) or set(view) != {"scale", "center"}
                    or type(view["scale"]) not in (int, float) or not math.isfinite(view["scale"])
                    or not 0.001 <= view["scale"] <= 10
                    or not isinstance(view["center"], list) or len(view["center"]) != 2
                    or any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e9
                           for value in view["center"])):
                raise ValueError("Invalid graph viewport.")
        if "connection" in state:
            value = state["connection"]
            if (not isinstance(value, dict) or set(value) != {"base_url", "model"}
                    or not isinstance(value["base_url"], str) or len(value["base_url"]) > 2048
                    or not isinstance(value["model"], str) or len(value["model"]) > 512):
                raise ValueError("Invalid research server connection.")

    @classmethod
    def validate_document(cls, document):
        """Check renderable shapes before touching the displayed research session.

        Core readers additionally validate artifact semantics. Presentation still
        rejects malformed plain data before clearing the previous completed view.
        """
        keys = {"format", "version", "state", "inspection", "capture", "benchmark", "capabilities"}
        optional = {"model_comparison", "graph_comparison"}
        if (not isinstance(document, dict) or not keys <= set(document) or set(document) - keys - optional
                or document.get("format") != "mlcookbook-llm-lab" or type(document.get("version")) is not int
                or document["version"] != 1):
            raise ValueError("Unsupported LLM research lab document.")
        cls.validate_session(document["state"])
        try:
            raw = json.dumps(document, allow_nan=False, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("Research data must contain finite JSON values.") from exc
        if len(raw.encode("utf-8")) > 64 * 1024 * 1024:
            raise ValueError("Research data exceeds the supported display budget.")

        def require(condition, name):
            if not condition:
                raise ValueError(f"Invalid research artifact: {name}.")

        for name in ("inspection", "capture", "benchmark", "capabilities", "model_comparison", "graph_comparison"):
            value = document.get(name)
            if value is None:
                continue
            require(isinstance(value, dict), name)
            warnings = value.get("warnings", [])
            require(isinstance(warnings, list) and all(isinstance(item, str) for item in warnings), name)
        for name in ("model_comparison", "graph_comparison"):
            comparison = document.get(name)
            if comparison is not None:
                require(isinstance(comparison.get("rows", []), list)
                        and all(isinstance(row, dict) for row in comparison.get("rows", [])), name)
                metadata = comparison.get("metadata_changes", [])
                require(isinstance(metadata, list) and all(isinstance(row, dict) for row in metadata), name)
        inspection = document["inspection"]
        if inspection is not None:
            require(isinstance(inspection.get("metadata", {}), dict)
                    and isinstance(inspection.get("summary", {}), dict), "inspection")
            tensors = inspection.get("tensors", [])
            require(isinstance(tensors, list) and all(isinstance(row, dict) for row in tensors), "inspection")
        benchmark = document["benchmark"]
        if benchmark is not None:
            require(isinstance(benchmark.get("trials", []), list)
                    and all(isinstance(row, dict) for row in benchmark.get("trials", []))
                    and isinstance(benchmark.get("summary", {}), dict), "benchmark")
            aggregates = benchmark.get("summary", {}).get("aggregates", {})
            require(isinstance(aggregates, dict) and all(isinstance(value, dict) for value in aggregates.values()),
                    "benchmark")
        capture = document["capture"]
        if capture is not None:
            graphs = capture.get("graphs", [])
            require(isinstance(graphs, list) and isinstance(capture.get("provenance", {}), dict), "capture")
            for graph in graphs:
                require(isinstance(graph, dict) and isinstance(graph.get("nodes", []), list), "capture")
                ids = set()
                for node in graph.get("nodes", []):
                    require(isinstance(node, dict) and type(node.get("id")) in (str, int), "capture")
                    key = str(node["id"])
                    require(key not in ids, "capture")
                    ids.add(key)
                    require(isinstance(node.get("inputs", []), list)
                            and all(type(value) in (str, int) or isinstance(value, dict)
                                    for value in node.get("inputs", [])), "capture")
            phase = document["state"].get("graph_phase")
            require(phase is None or phase < len(graphs), "graph phase")
            if phase is not None:
                graph_ids = {str(node["id"]) for node in graphs[phase].get("nodes", [])}
                selected = document["state"].get("graph_selected")
                trace = document["state"].get("graph_trace")
                require(not selected or selected in graph_ids, "graph selection")
                require(trace is None or set(trace) <= graph_ids, "graph trace")
