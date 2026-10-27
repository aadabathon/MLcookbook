"""Kitchen editor and selected-node inspector. No analytical imports or execution."""
from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.kitchen_canvas import NODE_LABELS, NODE_PORTS, KitchenCanvas
from gui.preprocessing_dialog import PreprocessingDialog
from gui.report_view import ReportView

COMPARISON_PARAMS = [
    ("alpha", "Ridge penalty", 0.0001, 10000, 4), ("n_estimators", "Forest trees", 1, 1000, None),
    ("max_depth", "Depth limit", 0, 100, None), ("min_samples_leaf", "Min leaf rows", 1, 1_000_000, None),
    ("test_size", "Test fraction", 0.05, 0.5, 2), ("random_state", "Random seed", 0, 2147483647, None),
    ("cv_folds", "Training CV folds", 2, 20, None),
]


class KitchenScreen(QWidget):
    changed = Signal(dict)
    selection_changed = Signal(object)
    capture_requested = Signal(str)
    run_requested = Signal(object)
    cancel_requested = Signal()
    save_requested = Signal(str)
    open_requested = Signal(str)

    def __init__(self, templates, parent=None):
        super().__init__(parent)
        self.templates = deepcopy(templates)
        self.editor = {"graph": {"nodes": [], "edges": [], "version": 1}, "layout": {}, "selected": None}
        self._undo, self._redo = [], []
        self._states, self._context, self._busy, self._building = {}, {}, False, False
        self._status_message = ""
        self._has_active = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SPACE_LG, theme.SPACE_MD, theme.SPACE_LG, theme.SPACE_MD)
        title = QHBoxLayout()
        heading = QLabel("Kitchen")
        heading.setObjectName("ScreenTitle")
        title.addWidget(heading)
        title.addStretch()
        self.open_button = QPushButton("Open Kitchen…")
        self.open_button.clicked.connect(self._open)
        self.save_button = QPushButton("Save Kitchen…")
        self.save_button.clicked.connect(self._save)
        title.addWidget(self.open_button)
        title.addWidget(self.save_button)
        outer.addLayout(title)
        self.editor_tools = QWidget()
        tools = QHBoxLayout(self.editor_tools)
        tools.setContentsMargins(0, 0, 0, 0)
        self.node_kind = QComboBox()
        for key, label in NODE_LABELS.items():
            self.node_kind.addItem(label, key)
        self.node_kind.setAccessibleName("Operation to add")
        tools.addWidget(self.node_kind)
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_node)
        tools.addWidget(self.add_button)
        for text, method in (("Connect…", self._connect_dialog), ("Delete", self.delete_selected),
                             ("Undo", self.undo), ("Redo", self.redo)):
            button = QPushButton(text)
            button.clicked.connect(method)
            tools.addWidget(button)
            if text == "Undo":
                self.undo_button = button
            elif text == "Redo":
                self.redo_button = button
        tools.addStretch()
        self.fit_button = QPushButton("Fit graph")
        tools.addWidget(self.fit_button)
        outer.addWidget(self.editor_tools)
        self.splitter = QSplitter()
        self.canvas = KitchenCanvas()
        self.canvas.setMinimumWidth(300)
        self.canvas.selected.connect(self._selected)
        self.canvas.moved.connect(self._moved)
        self.canvas.connect_requested.connect(self.connect_nodes)
        self.canvas.delete_requested.connect(self.delete_selected)
        self.canvas.message.connect(self.set_status)
        self.fit_button.clicked.connect(self.canvas.fit_graph)
        self.splitter.addWidget(self.canvas)
        self.inspector = QTabWidget()
        self.inspector.setObjectName("KitchenInspector")
        self.inspector.setMinimumWidth(330)
        self.inspector.setMaximumWidth(600)
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inspector.addTab(self.settings_scroll, "Settings")
        self.output = ReportView({})
        self.inspector.addTab(self.output, "Output")
        inspector_panel = QWidget()
        inspector_layout = QVBoxLayout(inspector_panel)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        self.result_state = QLabel()
        self.result_state.setObjectName("ResultState")
        self.result_state.setWordWrap(True)
        self.result_state.setTextFormat(Qt.TextFormat.PlainText)
        inspector_layout.addWidget(self.result_state)
        self.warning_button = QPushButton()
        self.warning_button.setObjectName("WarningSummary")
        self.warning_button.clicked.connect(self._show_warnings)
        self.warning_button.hide()
        self._warnings = []
        inspector_layout.addWidget(self.warning_button)
        inspector_layout.addWidget(self.inspector)
        self.splitter.addWidget(inspector_panel)
        self.splitter.setSizes([600, 410])
        outer.addWidget(self.splitter, stretch=1)
        footer = QHBoxLayout()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setMinimumWidth(0)
        footer.addWidget(self.status, stretch=1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedSize(70, 6)
        self.progress.setTextVisible(False)
        self.progress.hide()
        footer.addWidget(self.progress)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip("Stops between nodes. A model already fitting finishes first.")
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.cancel_button.hide()
        footer.addWidget(self.cancel_button)
        self.run_selected_button = QPushButton("Run to selected")
        self.run_selected_button.clicked.connect(self._run_selected)
        footer.addWidget(self.run_selected_button)
        self.run_button = QPushButton("Run workflow")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self._run_all)
        footer.addWidget(self.run_button)
        outer.addLayout(footer)
        for sequence, method in ((QKeySequence.StandardKey.Undo, self.undo), (QKeySequence.StandardKey.Redo, self.redo)):
            action = QAction(self)
            action.setShortcut(QKeySequence(sequence))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(method)
            self.addAction(action)
        self._build_settings()
        self._update_actions()

    def state(self):
        return deepcopy(self.editor)

    def set_editor(self, state, *, reset_history=False):
        self.validate_editor(state)
        self.editor = deepcopy(state)
        if reset_history:
            self._undo.clear()
            self._redo.clear()
        self.canvas.set_graph(self.editor["graph"], self.editor["layout"], self.editor["selected"], self._states)
        self._build_settings()
        self._update_actions()

    @staticmethod
    def validate_editor(state):
        """Reject API-only settings the controls cannot restore exactly."""
        for node in state["graph"]["nodes"]:
            if node["kind"] == "comparison":
                for key, _, low, high, decimals in COMPARISON_PARAMS:
                    value = node["settings"][key]
                    value = 0 if key == "max_depth" and value is None else value
                    if not low <= value <= high or (decimals is not None and value != round(value, decimals)):
                        raise ValueError(f"{node['id']}: {key} cannot be restored exactly by the Kitchen editor. "
                                         f"Supported range: {low}–{high}" + (f", {decimals} decimal places." if decimals is not None else "."))

    def _commit(self, state, *, redraw=True, internal=False):
        if (self._busy and not internal) or state == self.editor:
            return
        self._undo.append(self.state())
        self._undo = self._undo[-50:]
        self._redo.clear()
        self.editor = deepcopy(state)
        if redraw:
            self.canvas.set_graph(self.editor["graph"], self.editor["layout"], self.editor["selected"], self._states)
            self._build_settings()
        self._update_actions()
        self.changed.emit(self.state())

    def install_snapshot(self, node_id, identifier):
        state = self.state()
        for node in state["graph"]["nodes"]:
            if node["id"] == node_id and node["kind"] == "dataset":
                node["settings"]["snapshot"] = identifier
        self._commit(state, internal=True)

    def undo(self):
        if self._undo and not self._busy:
            self._redo.append(self.state())
            self.set_editor(self._undo.pop())
            self.changed.emit(self.state())

    def redo(self):
        if self._redo and not self._busy:
            self._undo.append(self.state())
            self.set_editor(self._redo.pop())
            self.changed.emit(self.state())

    def add_node(self):
        state, kind = self.state(), self.node_kind.currentData()
        if len(state["graph"]["nodes"]) >= 100:
            self.set_status("This Kitchen supports at most 100 nodes.")
            return
        ids = {n["id"] for n in state["graph"]["nodes"]}
        index = max([int(n["id"][1:]) for state in [self.editor, *self._undo, *self._redo]
                     for n in state["graph"]["nodes"] if n["id"].startswith("n") and n["id"][1:].isdigit()] or [0]) + 1
        while f"n{index}" in ids:
            index += 1
        key = f"n{index}"
        state["graph"]["nodes"].append({"id": key, "kind": kind, "settings": deepcopy(self.templates[kind])})
        state["layout"][key] = [((index - 1) % 2) * 260, ((index - 1) // 2) * 155]
        state["selected"] = key
        self._context = {}
        self._commit(state)
        self.canvas.ensureVisible(self.canvas.nodes[key])

    def _selected(self, key):
        if self.editor["selected"] != key:
            self.editor["selected"] = key
            self._context = {}
            self._build_settings()
            self._update_actions()
            self.selection_changed.emit(key)

    def _moved(self, positions):
        state = self.state()
        state["layout"] = positions
        self._commit(state, redraw=False)

    def connect_nodes(self, edge):
        if self._busy:
            return
        nodes = {n["id"]: n for n in self.editor["graph"]["nodes"]}
        if edge["source"] not in nodes or edge["target"] not in nodes:
            return
        output = NODE_PORTS[nodes[edge["source"]]["kind"]][1]
        incoming = NODE_PORTS[nodes[edge["target"]]["kind"]][0]
        if not output or output != incoming or edge["source_port"] != output or edge["target_port"] != incoming:
            self.set_status("Incompatible ports: dataset → dataset, result → report.")
            return
        if any(e["target"] == edge["target"] for e in self.editor["graph"]["edges"]):
            self.set_status("That input is already connected. Select its connection and Delete it first.")
            return
        children = {}
        for existing in self.editor["graph"]["edges"]:
            children.setdefault(existing["source"], []).append(existing["target"])
        pending, seen = [edge["target"]], set()
        while pending:
            key = pending.pop()
            if key == edge["source"]:
                self.set_status("This connection would create a cycle.")
                return
            if key not in seen:
                seen.add(key)
                pending.extend(children.get(key, []))
        state = self.state()
        state["graph"]["edges"].append(deepcopy(edge))
        self._commit(state)
        self.set_status("")

    def _connect_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Connect nodes")
        form = QFormLayout(dialog)
        source, target = QComboBox(), QComboBox()
        for node in self.editor["graph"]["nodes"]:
            incoming, output = NODE_PORTS[node["kind"]]
            label = f"{node['id']} · {NODE_LABELS[node['kind']]}"
            if output:
                source.addItem(label + " / " + output, (node["id"], output))
            if incoming:
                target.addItem(label + " / " + incoming, (node["id"], incoming))
        form.addRow("From output", source)
        form.addRow("To input", target)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(source.count() and target.count()))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            a, b = source.currentData(), target.currentData()
            self.connect_nodes({"source": a[0], "source_port": a[1], "target": b[0], "target_port": b[1]})
        dialog.deleteLater()

    def delete_selected(self):
        state, key = self.state(), self.editor["selected"]
        if key:
            state["graph"]["nodes"] = [n for n in state["graph"]["nodes"] if n["id"] != key]
            state["graph"]["edges"] = [e for e in state["graph"]["edges"] if key not in (e["source"], e["target"])]
            state["layout"].pop(key, None)
            state["selected"] = None
        else:
            indices = self.canvas.selected_edges()
            state["graph"]["edges"] = [e for i, e in enumerate(state["graph"]["edges"]) if i not in indices]
        self._context = {}
        self._commit(state)

    def _node(self):
        return next((n for n in self.editor["graph"]["nodes"] if n["id"] == self.editor["selected"]), None)

    def set_context(self, context, *, report=None, state="unrun"):
        if context != self._context:
            self._context = deepcopy(context)
            self._build_settings()
        self.result_state.setText(state.capitalize() + (" · showing the last completed output" if state in ("stale", "failed", "cancelled", "blocked") and report else "") +
                                  (": " + context["error"][:180] if state == "failed" and context.get("error") else ""))
        self.result_state.setToolTip(context.get("error", ""))
        self.result_state.setVisible(self.editor["selected"] is not None and state != "completed")
        self._warnings = list((report or {}).get("warnings", []))
        self.warning_button.setText(f"{len(self._warnings)} warning{'s' if len(self._warnings) != 1 else ''} · Review")
        self.warning_button.setVisible(bool(self._warnings))
        index = self.inspector.currentIndex()
        old = self.output
        self.output = ReportView(report or {})
        self.inspector.removeTab(1)
        self.inspector.insertTab(1, self.output, "Output")
        self.inspector.setCurrentIndex(index)
        old.deleteLater()

    def set_states(self, states):
        self._states = deepcopy(states)
        self.canvas.set_states(states)

    def _build_settings(self):
        self._building = True
        self.capture_button = None
        old = self.settings_scroll.takeWidget()
        if old:
            old.deleteLater()
        panel, self.controls = QWidget(), {}
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        node = self._node()
        if not node:
            label = QLabel("Add an operation or select a node.")
            label.setWordWrap(True)
            layout.addWidget(label)
        else:
            heading = QLabel(f"{node['id']} · {NODE_LABELS[node['kind']]}")
            heading.setTextFormat(Qt.TextFormat.PlainText)
            layout.addWidget(heading)
            s, columns = node["settings"], self._context.get("columns", [])
            if node["kind"] == "dataset":
                self.capture_button = QPushButton("Use active dataset")
                self.capture_button.setToolTip("Capture the ordinary dataset shown in Data. Later imports do not replace this snapshot.")
                self.capture_button.setEnabled(self._has_active and not self._busy)
                self.capture_button.clicked.connect(self._capture)
                layout.addWidget(self.capture_button)
                identity = QLabel(self._context.get("identity", "No snapshot selected"))
                identity.setWordWrap(True)
                identity.setTextFormat(Qt.TextFormat.PlainText)
                layout.addWidget(identity)
            elif node["kind"] in ("select", "comparison"):
                if node["kind"] == "comparison":
                    form = QFormLayout()
                    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
                    target = QComboBox()
                    target.addItem("Choose target…", "")
                    for column in columns:
                        target.addItem(column["name"], column["name"])
                    if s["target"] and target.findData(s["target"]) < 0:
                        target.addItem(s["target"] + " (missing)", s["target"])
                    target.setCurrentIndex(target.findData(s["target"]))
                    target.currentIndexChanged.connect(self._settings_edited)
                    self.controls["target"] = target
                    form.addRow("Target", target)
                    layout.addLayout(form)
                self._column_controls(layout, node, columns)
                if node["kind"] == "comparison":
                    form = QFormLayout()
                    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
                    for key, label, low, high, decimals in COMPARISON_PARAMS:
                        widget = QSpinBox() if decimals is None else QDoubleSpinBox()
                        if decimals is not None:
                            widget.setDecimals(decimals)
                            widget.setSingleStep(0.05 if key == "test_size" else 0.1)
                        widget.setRange(low, high)
                        widget.setValue(s[key] or 0)
                        if key == "max_depth":
                            widget.setSpecialValueText("Unlimited")
                        widget.valueChanged.connect(self._settings_edited)
                        self.controls[key] = widget
                        form.addRow(label, widget)
                    layout.addLayout(form)
                    preprocessing = QPushButton("Preprocessing && validation…")
                    preprocessing.clicked.connect(self._preprocessing)
                    layout.addWidget(preprocessing)
            else:
                button = QPushButton("Inspect completed report")
                button.clicked.connect(self._inspect_output)
                layout.addWidget(button)
        layout.addStretch()
        self.settings_scroll.setWidget(panel)
        panel.setEnabled(not self._busy)
        self._building = False

    def _column_controls(self, layout, node, columns):
        s = node["settings"]
        key = "columns" if node["kind"] == "select" else "features"
        names = [c["name"] for c in columns]
        listed = columns + [{"name": name, "type": "unsupported"} for name in s[key] if name not in names]
        table = QTableWidget(len(listed), 2)
        table.setHorizontalHeaderLabels(["Column" if key == "columns" else "Feature", "Role"])
        table.setToolTip("Roles describe statistical meaning; they do not convert values. Exclude IDs and target leakage.")
        table.setAccessibleDescription(table.toolTip())
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setMinimumHeight(160)
        for row, column in enumerate(listed):
            name = column["name"]
            item = QTableWidgetItem(name)
            item.setToolTip(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name in s[key] else Qt.CheckState.Unchecked)
            table.setItem(row, 0, item)
            if key == "columns":
                role = QComboBox()
                role.addItems(["numeric", "categorical", "boolean", "datetime", "text", "unsupported"])
                role.setCurrentText(s["roles"].get(name, column["type"]))
                role.currentTextChanged.connect(self._settings_edited)
                table.setCellWidget(row, 1, role)
            else:
                table.setItem(row, 1, QTableWidgetItem(column["type"]))
        table.cellChanged.connect(self._settings_edited)
        self.controls[key] = table
        layout.addWidget(table)
        if key == "features":
            self._target_checks(s)

    def _target_checks(self, settings):
        table = self.controls["features"]
        table.blockSignals(True)
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            name = item.data(Qt.ItemDataRole.UserRole)
            flags = item.flags() | Qt.ItemFlag.ItemIsEnabled
            item.setFlags(flags & ~Qt.ItemFlag.ItemIsEnabled if name == settings["target"] else flags)
            item.setCheckState(Qt.CheckState.Checked if name in settings["features"] else Qt.CheckState.Unchecked)
        table.blockSignals(False)

    def _settings_edited(self, *_):
        if self._building or self._busy:
            return
        state, node = self.state(), self._node()
        if not node or node["kind"] not in ("select", "comparison"):
            return
        s = deepcopy(node["settings"])
        key = "columns" if node["kind"] == "select" else "features"
        table = self.controls[key]
        checked = [table.item(row, 0).data(Qt.ItemDataRole.UserRole) for row in range(table.rowCount())
                   if table.item(row, 0).checkState() == Qt.CheckState.Checked]
        s[key] = [name for name in s[key] if name in checked] + [name for name in checked if name not in s[key]]
        if key == "columns":
            s["roles"] = {table.item(row, 0).data(Qt.ItemDataRole.UserRole): table.cellWidget(row, 1).currentText()
                          for row in range(table.rowCount()) if table.item(row, 0).data(Qt.ItemDataRole.UserRole) in checked}
        else:
            s["target"] = self.controls["target"].currentData()
            s["features"] = [name for name in s["features"] if name != s["target"]]
            for setting, widget in self.controls.items():
                if setting not in ("target", "features"):
                    s[setting] = widget.value() or None if setting == "max_depth" else widget.value()
            self._target_checks(s)
        for candidate in state["graph"]["nodes"]:
            if candidate["id"] == node["id"]:
                candidate["settings"] = s
        self._commit(state, redraw=False)

    def _preprocessing(self):
        node = self._node()
        dialog = PreprocessingDialog(node["settings"]["preprocessing"], node["settings"]["cv_folds"], self)
        # The shared recipe dialog has a short preset list. Retain valid custom
        # configuration and the Kitchen's 2–20 folds rather than silently defaulting.
        for key, combo in dialog.combos.items():
            current = node["settings"]["preprocessing"][key]
            index = next((i for i in range(combo.count()) if combo.itemData(i) == current and
                          type(combo.itemData(i)) is type(current)), -1)
            if index < 0:
                combo.addItem(str(current), current)
                index = combo.count() - 1
            combo.setCurrentIndex(index)
        folds = node["settings"]["cv_folds"]
        if dialog.cv.findData(folds) < 0:
            dialog.cv.addItem(f"{folds} folds", folds)
        dialog.cv.setCurrentIndex(dialog.cv.findData(folds))
        if dialog.exec() == QDialog.DialogCode.Accepted:
            preprocessing, folds = dialog.values()
            if folds is None or folds < 2:
                self.set_status("Model comparison requires at least two training CV folds.")
            else:
                state = self.state()
                for candidate in state["graph"]["nodes"]:
                    if candidate["id"] == node["id"]:
                        candidate["settings"].update(preprocessing=preprocessing, cv_folds=folds)
                self._commit(state)
        dialog.deleteLater()

    def _capture(self):
        if self.editor["selected"]:
            self.capture_requested.emit(self.editor["selected"])

    def _inspect_output(self):
        self.inspector.setCurrentIndex(1)

    def _show_warnings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Warnings for the selected output")
        dialog.resize(620, 400)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        notes = QVBoxLayout(content)
        for warning in self._warnings:
            label = QLabel(warning)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            notes.addWidget(label)
        notes.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
        dialog.deleteLater()

    def _run_selected(self):
        self.run_requested.emit(self.editor["selected"])

    def _run_all(self):
        self.run_requested.emit(None)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Kitchen", "", "Kitchen workflows (*.mlck)")
        if path:
            self.open_requested.emit(path)

    def _save(self):
        dialog = QFileDialog(self, "Save Kitchen with snapshots and completed results", "workflow.mlck", "Kitchen workflows (*.mlck)")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("mlck")
        if dialog.exec():
            self.save_requested.emit(dialog.selectedFiles()[0])
        dialog.deleteLater()

    def set_active_available(self, available):
        self._has_active = available
        if self.capture_button is not None:
            self.capture_button.setEnabled(available and not self._busy)

    def set_status(self, message):
        self._status_message = message
        self.status.setText(" ".join(message.split())[:180])
        self.status.setToolTip(message)

    def set_busy(self, busy):
        self._busy = busy
        self.editor_tools.setEnabled(not busy)
        self.canvas.setEnabled(not busy)
        self.settings_scroll.setEnabled(not busy)
        if self.settings_scroll.widget():
            self.settings_scroll.widget().setEnabled(not busy)
        self.open_button.setEnabled(not busy)
        self.save_button.setEnabled(not busy)
        if not busy:
            self.set_cancellable(False)
        self.progress.setVisible(busy)
        self._update_actions()

    def set_cancellable(self, cancellable):
        self.cancel_button.setVisible(cancellable)
        self.cancel_button.setEnabled(cancellable)

    def _update_actions(self):
        self.undo_button.setEnabled(bool(self._undo) and not self._busy)
        self.redo_button.setEnabled(bool(self._redo) and not self._busy)
        self.run_button.setEnabled(bool(self.editor["graph"]["nodes"]) and not self._busy)
        self.run_selected_button.setEnabled(self.editor["selected"] is not None and not self._busy)
