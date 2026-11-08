"""Optional Python editor, console and genuine notebook presentation.

Runtime ownership and all dataset/file actions remain in the coordinator. This
screen connects a Jupyter frontend to a supplied connection file, never to app
objects. Optional QtConsole/WebEngine imports happen only when requested.
"""

from __future__ import annotations

import keyword
import re
from pathlib import Path
from urllib.parse import urlsplit

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui import theme


class _PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.rules = []
        for pattern, color in [
            (r"\b(?:" + "|".join(keyword.kwlist) + r")\b", theme.UI_ACCENT),
            (r"\b\d+(?:\.\d+)?\b", theme.PLOT_ALT),
            (r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", theme.WARNING),
            (r"#.*$", theme.TEXT_MUTED),
        ]:
            style = QTextCharFormat()
            style.setForeground(QColor(color))
            self.rules.append((re.compile(pattern), style))

    def highlightBlock(self, text):
        for pattern, style in self.rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), style)


class ProgrammingScreen(QWidget):
    action_requested = Signal(str, dict)

    def __init__(self, state: dict | None = None, parent=None):
        super().__init__(parent)
        self.console = None
        self.client = None
        self.web = None
        self._connection_file = ""
        self._kernel_dead = False
        self._script_path = ""
        self._notebook_url = ""
        self._notebook_path = ""
        self._workspace = ""
        self._active_available = False
        self._busy = False
        self._return_valid = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SPACE_LG, theme.SPACE_MD, theme.SPACE_LG, theme.SPACE_MD)
        outer.setSpacing(theme.SPACE_SM)
        header = QHBoxLayout()
        title = QLabel("Programming")
        title.setObjectName("PageTitle")
        header.addWidget(title, stretch=1)
        self.start_button = QPushButton("Start Python")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self._start)
        header.addWidget(self.start_button)
        self.interrupt_button = QPushButton("Interrupt")
        self.interrupt_button.setToolTip("Interrupt the console/script kernel. Restart Python if a native operation remains blocked. Notebook kernels use JupyterLab's controls.")
        self.interrupt_button.clicked.connect(lambda: self.action_requested.emit("interrupt", {}))
        self.restart_button = QPushButton("Restart")
        self.restart_button.setToolTip("Restart the console/script kernel. Notebook kernels use JupyterLab's controls.")
        self.restart_button.clicked.connect(lambda: self.action_requested.emit("restart", {}))
        header.addWidget(self.interrupt_button)
        header.addWidget(self.restart_button)
        outer.addLayout(header)

        environment = QHBoxLayout()
        environment.addWidget(QLabel("Python"))
        self.python = QLineEdit()
        self.python.setPlaceholderText("Select python.exe from an environment with mlcookbook[programming]")
        self.python.setMinimumWidth(0)
        environment.addWidget(self.python, stretch=1)
        self.python_browse = QPushButton("Browse…")
        self.python_browse.clicked.connect(self._choose_python)
        environment.addWidget(self.python_browse)
        environment.addWidget(QLabel("Folder"))
        self.folder = QLineEdit()
        self.folder.setMinimumWidth(0)
        environment.addWidget(self.folder, stretch=1)
        self.folder_browse = QPushButton("Browse…")
        self.folder_browse.clicked.connect(self._choose_folder)
        environment.addWidget(self.folder_browse)
        outer.addLayout(environment)
        self.environment_label = QLabel("Python starts only when requested. Code runs with this environment's normal permissions.")
        self.environment_label.setObjectName("HelpText")
        self.environment_label.setWordWrap(True)
        self.environment_label.setMinimumWidth(0)
        outer.addWidget(self.environment_label)

        self.splitter = QSplitter()
        self.splitter.setOrientation(Qt.Orientation.Vertical)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("ProgrammingTabs")
        script = QWidget()
        script_layout = QVBoxLayout(script)
        script_layout.setContentsMargins(0, theme.SPACE_SM, 0, 0)
        actions = QHBoxLayout()
        self.new_button = QPushButton("New script")
        self.new_button.clicked.connect(self._new_script)
        self.open_button = QPushButton("Open…")
        self.open_button.clicked.connect(self._open_script)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save_script)
        self.run_button = QPushButton("Run file")
        self.run_button.clicked.connect(lambda: self._run(False))
        self.selection_button = QPushButton("Run selection")
        self.selection_button.clicked.connect(lambda: self._run(True))
        for button in (self.new_button, self.open_button, self.save_button, self.run_button, self.selection_button):
            actions.addWidget(button)
        self.script_name = QLabel("Untitled.py")
        self.script_name.setObjectName("HelpText")
        self.script_name.setMinimumWidth(0)
        self.script_name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        actions.addWidget(self.script_name, stretch=1)
        script_layout.addLayout(actions)
        self.editor = QPlainTextEdit()
        self.editor.setObjectName("ProgrammingEditor")
        self.editor.setFont(QFont("Consolas", theme.BODY_POINT_SIZE))
        self.editor.setStyleSheet(f'QPlainTextEdit#ProgrammingEditor {{ font-family: "Consolas", "Cascadia Code", monospace; font-size: 14px; color: {theme.TEXT}; background: {theme.SURFACE}; }}')
        self.editor.setTabStopDistance(self.editor.fontMetrics().horizontalAdvance(" ") * 4)
        self.highlighter = _PythonHighlighter(self.editor.document())
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setPlaceholderText("Write Python here. Send a dataset to inspect df and data in the console.")
        self.editor.document().modificationChanged.connect(self._modified)
        script_layout.addWidget(self.editor, stretch=1)
        self.tabs.addTab(script, "Script")

        notebook = QWidget()
        notebook_layout = QVBoxLayout(notebook)
        notebook_layout.setContentsMargins(0, theme.SPACE_SM, 0, 0)
        notebook_actions = QHBoxLayout()
        self.new_notebook = QPushButton("New notebook…")
        self.new_notebook.clicked.connect(lambda: self._notebook_action(True))
        self.open_notebook = QPushButton("Open notebook…")
        self.open_notebook.clicked.connect(lambda: self._notebook_action(False))
        notebook_actions.addWidget(self.new_notebook)
        notebook_actions.addWidget(self.open_notebook)
        note = QLabel("Notebook kernels are independent. Run the input cell; use workbench.publish(df) to return a table.")
        note.setObjectName("HelpText")
        note.setWordWrap(True)
        note.setMinimumWidth(0)
        notebook_actions.addWidget(note, stretch=1)
        notebook_layout.addLayout(notebook_actions)
        self.web_host = QWidget()
        self.web_layout = QVBoxLayout(self.web_host)
        self.web_layout.setContentsMargins(0, 0, 0, 0)
        self.notebook_placeholder = QLabel("Open or create a notebook to start local JupyterLab.")
        self.notebook_placeholder.setObjectName("HelpText")
        self.web_layout.addWidget(self.notebook_placeholder)
        notebook_layout.addWidget(self.web_host, stretch=1)
        self.tabs.addTab(notebook, "Notebook")
        self.splitter.addWidget(self.tabs)

        console_panel = QWidget()
        console_layout = QVBoxLayout(console_panel)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_actions = QHBoxLayout()
        self.console_toggle = QPushButton("▾  Console")
        self.console_toggle.setObjectName("SectionToggle")
        self.console_toggle.setCheckable(True)
        self.console_toggle.setChecked(True)
        self.console_toggle.toggled.connect(self._toggle_console)
        console_actions.addWidget(self.console_toggle)
        self.send_button = QPushButton("Send active dataset")
        self.send_button.clicked.connect(lambda: self.action_requested.emit("send_data", {}))
        console_actions.addWidget(self.send_button)
        self.variable = QLineEdit("df")
        self.variable.setMaximumWidth(110)
        self.variable.setToolTip("DataFrame variable to publish from the console kernel")
        console_actions.addWidget(self.variable)
        self.return_button = QPushButton("Return variable")
        self.return_button.clicked.connect(lambda: self.action_requested.emit("publish", {"variable": self.variable.text()}))
        console_actions.addWidget(self.return_button)
        self.preview_button = QPushButton("Preview published table")
        self.preview_button.clicked.connect(self._preview_return)
        console_actions.addWidget(self.preview_button)
        self.accept_button = QPushButton("Use dataset")
        self.accept_button.clicked.connect(lambda: self.action_requested.emit("accept_return", {}))
        self.accept_button.setEnabled(False)
        console_actions.addWidget(self.accept_button)
        self.tabs.currentChanged.connect(self._refresh_actions)
        console_actions.addStretch()
        console_layout.addLayout(console_actions)
        self.return_preview = QLabel("")
        self.return_preview.setObjectName("HelpText")
        self.return_preview.setWordWrap(True)
        self.return_preview.hide()
        console_layout.addWidget(self.return_preview)
        self.return_table = QTableWidget()
        self.return_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.return_table.setAlternatingRowColors(True)
        self.return_table.verticalHeader().hide()
        self.return_table.setMinimumWidth(0)
        self.return_table.setMaximumHeight(180)
        self.return_table.hide()
        console_layout.addWidget(self.return_table)
        self.console_host = QWidget()
        self.console_layout = QVBoxLayout(self.console_host)
        self.console_layout.setContentsMargins(0, 0, 0, 0)
        self.console_placeholder = QLabel("Start Python to open the console. Scripts and console use the same kernel.")
        self.console_placeholder.setObjectName("HelpText")
        self.console_layout.addWidget(self.console_placeholder)
        console_layout.addWidget(self.console_host, stretch=1)
        self.splitter.addWidget(console_panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        outer.addWidget(self.splitter, stretch=1)
        self.status = QLabel("")
        self.status.setObjectName("HelpText")
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(0)
        outer.addWidget(self.status)
        self.save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self.editor, activated=self._save_script)
        self.save_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.selection_shortcut = QShortcut(QKeySequence("Shift+Return"), self.editor, activated=lambda: self._run(True))
        self.selection_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        if state:
            self.restore_session(state)
        self._refresh_actions()

    def _choose_python(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Python interpreter", self.python.text(), "Python (python.exe python3 python);;All files (*)")
        if path:
            self.python.setText(path)

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Programming folder", self.folder.text())
        if path:
            self.folder.setText(path)

    def _start(self):
        if self._connection_file:
            self.action_requested.emit("stop", {})
        else:
            self.action_requested.emit("start", {"python": self.python.text(), "workspace": self.folder.text()})

    def _modified(self, changed):
        self.script_name.setText((Path(self._script_path).name if self._script_path else "Untitled.py") + (" *" if changed else ""))

    def _new_script(self):
        if self.can_leave_script():
            self.set_script("", "")

    def _open_script(self):
        if not self.can_leave_script():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Python script", self.folder.text(), "Python scripts (*.py)")
        if path:
            self.action_requested.emit("open_script", {"path": path})

    def _save_script(self):
        path = self._script_path
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save Python script", str(Path(self.folder.text()) / "experiment.py"), "Python scripts (*.py)")
        if path:
            if not path.lower().endswith(".py"):
                path += ".py"
            self.action_requested.emit("save_script", {"path": path, "text": self.editor.toPlainText()})

    def _run(self, selection):
        text = self.editor.textCursor().selectedText().replace("\u2029", "\n") if selection else self.editor.toPlainText()
        if text.strip():
            self.action_requested.emit("execute", {"code": text, "path": self._script_path, "selection": selection})

    def _notebook_action(self, create):
        if create:
            path, _ = QFileDialog.getSaveFileName(self, "Create notebook", str(Path(self.folder.text()) / "experiment.ipynb"), "Jupyter notebooks (*.ipynb)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Open notebook", self.folder.text(), "Jupyter notebooks (*.ipynb)")
        if path:
            self.action_requested.emit("new_notebook" if create else "open_notebook", {"path": path})

    def _preview_return(self):
        payload = {"context": "console"}
        if self.tabs.currentIndex() == 1:
            payload = {"context": "notebook", "notebook": self._notebook_path}
        self.action_requested.emit("preview_return", payload)

    def set_script(self, text: str, path: str):
        self._script_path = path
        self.editor.setPlainText(text)
        self.editor.document().setModified(False)
        self._modified(False)

    def mark_saved(self, path: str):
        self._script_path = path
        self.editor.document().setModified(False)
        self._modified(False)

    def set_active_available(self, available: bool):
        self._active_available = available
        self._refresh_actions()

    def set_runtime(self, info: dict):
        environment = info.get("environment", {})
        self._workspace = info.get("workspace", self.folder.text())
        self.folder.setText(self._workspace)
        if environment:
            self.python.setText(environment["python"])
            packages = environment.get("packages", {})
            self.environment_label.setText("Python " + environment.get("version", "") + " · " +
                                           " · ".join(f"{name} {packages[name]}" for name in ("mlcookbook", "pandas", "ipykernel") if name in packages))
        connection = info.get("connection_file", "")
        if connection and connection != self._connection_file:
            self.detach_runtime()
            try:
                from qtconsole.client import QtKernelClient
                from qtconsole.rich_jupyter_widget import RichJupyterWidget
                self.console = RichJupyterWidget(parent=self.console_host)
                # The coordinator owns recovery. Avoid QtConsole's default
                # "restarting" message when it does not own a kernel manager.
                self.console.custom_restart = True
                self.console.font_family = "Consolas"
                self.console.font_size = theme.BODY_POINT_SIZE
                self.console.syntax_style = "monokai"
                self.console.style_sheet = f"QTextEdit, QPlainTextEdit {{ background: {theme.SURFACE}; color: {theme.TEXT}; }} .in-prompt {{ color: {theme.UI_ACCENT}; }} .out-prompt {{ color: {theme.PLOT_ALT}; }} .error {{ color: {theme.WARNING}; }}"
                self.client = QtKernelClient(connection_file=connection)
                self.client.load_connection_file()
                self.client.start_channels()
                self.console.kernel_client = self.client
                self.client.hb_channel.kernel_died.connect(
                    lambda delay, client=self.client: self._kernel_died(delay) if self.client is client else None
                )
                self.client.hb_channel.time_to_dead = 3.0
                self.client.hb_channel.unpause()
                self.console.executing.connect(lambda *_: self.action_requested.emit("execution_started", {}))
                self.console.executed.connect(self._execution_finished)
                self.console.exit_requested.connect(lambda *_: self.action_requested.emit("stop", {}))
                self.console_layout.addWidget(self.console, stretch=1)
                self.console_placeholder.hide()
                self._connection_file = connection
                self._kernel_dead = False
            except ImportError:
                self.set_status("The application needs its programming extra installed to display QtConsole.")
        elif not connection:
            self.detach_runtime()
        self._refresh_actions()

    def _execution_finished(self, message):
        content = message.get("content", message) if isinstance(message, dict) else {}
        details = {"status": content.get("status", "ok")}
        if details["status"] == "error":
            details.update({name: content.get(name, "") for name in ("ename", "evalue")})
            details["traceback"] = content.get("traceback", [])
        self.action_requested.emit("execution_finished", details)

    def _kernel_died(self, delay):
        if not self._connection_file or self._kernel_dead:
            return
        self._kernel_dead = True
        self.set_status("The Python kernel stopped. Restart Python to continue; editor contents are retained.")
        self._refresh_actions()
        self.action_requested.emit("kernel_died", {"reason": "heartbeat", "seconds_since_heartbeat": float(delay)})

    def execute_code(self, code: str):
        if self.console is None or self._kernel_dead:
            self.set_status("Start Python first.")
            return
        self.console_toggle.setChecked(True)
        self.console.execute(code, hidden=False, interactive=False)

    def detach_runtime(self):
        self._connection_file = ""
        if self.console is not None:
            self.console.kernel_client = None
        if self.client is not None:
            self.client.hb_channel.pause()
            self.client.stop_channels()
            self.client.deleteLater()
            self.client = None
        if self.console is not None:
            self.console.hide()
            self.console_layout.removeWidget(self.console)
            self.console.deleteLater()
            self.console = None
        self._kernel_dead = False
        self.console_placeholder.show()

    def set_notebook_url(self, url: str, *, notebook_path: str = ""):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
            raise ValueError("The notebook must use the app-owned local server.")
        if self.web is None:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            self.web = QWebEngineView(self.web_host)
            self.web_layout.addWidget(self.web, stretch=1)
            self.notebook_placeholder.hide()
        self._notebook_url = url
        self._notebook_path = notebook_path
        self.web.setUrl(QUrl(url))
        self.tabs.setCurrentIndex(1)
        self.console_toggle.setChecked(False)
        self._refresh_actions()

    def set_return_preview(self, view: dict | None):
        self._return_valid = view is not None
        self.return_preview.setVisible(view is not None)
        self.return_table.setVisible(view is not None)
        if view is not None:
            columns = view.get("columns", [])
            names = [column["name"] if isinstance(column, dict) else str(column) for column in columns]
            notebook = view.get("source", {}).get("notebook")
            origin = "Notebook " + Path(notebook).name if notebook else "Console table"
            self.return_preview.setText(f"{origin} · {view['rows']} rows · {len(columns)} columns · preview of first 20 rows")
            rows = view.get("preview", [])[:20]
            self.return_table.clear()
            self.return_table.setColumnCount(len(names))
            self.return_table.setHorizontalHeaderLabels(names)
            self.return_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                for column_index, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    item.setToolTip(str(value))
                    self.return_table.setItem(row_index, column_index, item)
        self._refresh_actions()

    def clear_notebook(self):
        """Retire the old server's document when switching environments."""
        self._notebook_url = ""
        self._notebook_path = ""
        if self.web is not None:
            self.web.stop()
            self.web.hide()
            self.web_layout.removeWidget(self.web)
            self.web.deleteLater()
            self.web = None
        self.notebook_placeholder.show()

    def _toggle_console(self, opened):
        self.console_host.setVisible(opened)
        self.console_toggle.setText(("▾  " if opened else "▸  ") + "Console")

    def set_busy(self, busy: bool):
        self._busy = busy
        self._refresh_actions()

    def _refresh_actions(self):
        running = bool(self._connection_file)
        notebook = self.tabs.currentIndex() == 1
        self.preview_button.setText("Preview notebook table" if notebook else "Preview console table")
        self.preview_button.setToolTip("Last notebook opened with this app: " + (self._notebook_path or "none")
                                       if notebook else "Preview the current console's published table")
        for widget in (self.python, self.folder, self.python_browse, self.folder_browse):
            widget.setEnabled(not self._busy and not running)
        self.start_button.setText("Stop Python" if running else "Start Python")
        self.start_button.setEnabled(not self._busy)
        self.restart_button.setEnabled(not self._busy and running)
        for widget in (self.run_button, self.selection_button, self.return_button):
            widget.setEnabled(not self._busy and running and not self._kernel_dead)
        self.interrupt_button.setEnabled(running and not self._kernel_dead)
        self.send_button.setEnabled(not self._busy and running and not self._kernel_dead and self._active_available)
        self.preview_button.setEnabled(not self._busy)
        self.new_notebook.setEnabled(not self._busy)
        self.open_notebook.setEnabled(not self._busy)
        self.accept_button.setEnabled(not self._busy and self._return_valid)

    def set_status(self, text: str):
        self.status.setText(text)

    def can_leave_script(self) -> bool:
        if not self.editor.document().isModified():
            return True
        answer = QMessageBox.question(self, "Unsaved Python script", "Save changes to this script?",
                                      QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                      QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            # Coordinator handles the small text save synchronously and calls
            # mark_saved only on success; failed/cancelled saves retain edits.
            self._save_script()
            return not self.editor.document().isModified()
        return True

    def can_leave(self) -> bool:
        if not self.can_leave_script():
            return False
        if self.web is not None:
            answer = QMessageBox.question(self, "Leave notebook", "Save notebook edits in JupyterLab before leaving. Leave this view?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                          QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return False
        return True

    def session_state(self) -> dict:
        return {"python": self.python.text(), "workspace": self.folder.text(),
                "script_path": self._script_path, "script": self.editor.toPlainText(),
                "modified": self.editor.document().isModified(), "tab": self.tabs.currentIndex(),
                "console_open": self.console_toggle.isChecked(), "sizes": self.splitter.sizes()}

    def restore_session(self, state: dict):
        self.python.setText(state.get("python", ""))
        self.folder.setText(state.get("workspace", ""))
        self.set_script(state.get("script", ""), state.get("script_path", ""))
        self.editor.document().setModified(bool(state.get("modified", False)))
        self.tabs.setCurrentIndex(state.get("tab", 0))
        self.console_toggle.setChecked(state.get("console_open", True))
        if state.get("sizes"):
            self.splitter.setSizes(state["sizes"])
