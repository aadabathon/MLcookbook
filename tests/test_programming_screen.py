"""Programming presentation keeps notebook shortcuts and runtime state separate."""

import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.screens.programming import ProgrammingScreen  # noqa: E402


@pytest.fixture
def screen():
    application = QApplication.instance() or QApplication([])
    widget = ProgrammingScreen()
    yield widget
    widget.detach_runtime()
    widget.deleteLater()
    application.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_return_preview_has_rows_and_enables_accept_after_job(screen):
    screen.set_busy(True)
    screen.set_return_preview({"rows": 2, "columns": [{"name": "identifier"}, {"name": "value"}],
                               "preview": [["001", "2.5"], ["002", "missing"]]})
    assert not screen.accept_button.isEnabled()
    assert screen.return_table.rowCount() == 2
    assert screen.return_table.item(0, 0).text() == "001"
    assert screen.return_table.item(1, 1).text() == "missing"
    screen.set_busy(False)
    assert screen.accept_button.isEnabled()
    screen.set_return_preview(None)
    assert not screen.accept_button.isEnabled()


def test_editor_shortcuts_do_not_capture_notebook_save_or_execution(screen):
    assert screen.save_shortcut.parent() is screen.editor
    assert screen.selection_shortcut.parent() is screen.editor
    assert screen.save_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    assert screen.selection_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut


def test_published_preview_names_the_console_or_last_app_opened_notebook(screen):
    actions = []
    screen.action_requested.connect(lambda action, value: actions.append((action, value)))
    screen.preview_button.click()
    assert actions[-1] == ("preview_return", {"context": "console"})
    screen._notebook_path = "folder/saved.ipynb"
    screen.tabs.setCurrentIndex(1)
    screen.preview_button.click()
    assert actions[-1] == ("preview_return", {"context": "notebook", "notebook": "folder/saved.ipynb"})
    assert "saved.ipynb" in screen.preview_button.toolTip()
    screen.set_return_preview({"rows": 1, "columns": ["x"], "preview": [[1]],
                               "source": {"notebook": "folder/saved.ipynb"}})
    assert "saved.ipynb" in screen.return_preview.text()


def test_hide_show_does_not_retire_live_connection(screen):
    screen._connection_file = "retained-console.json"
    screen.hide()
    screen.show()
    screen.hide()
    assert screen._connection_file == "retained-console.json"


def test_start_button_becomes_stop_and_environment_can_be_changed_after_stop(screen):
    actions = []
    screen.action_requested.connect(lambda action, value: actions.append((action, value)))
    screen._connection_file = "connected.json"
    screen._refresh_actions()
    assert screen.start_button.text() == "Stop Python"
    assert screen.start_button.isEnabled()
    assert not screen.python.isEnabled()
    screen.start_button.click()
    assert actions[-1] == ("stop", {})
    screen.set_runtime({})
    assert screen.start_button.text() == "Start Python"
    assert screen.python.isEnabled()


def test_execution_error_forwards_diagnostics_without_program_text(screen):
    actions = []
    screen.action_requested.connect(lambda action, value: actions.append((action, value)))
    screen._execution_finished({"content": {"status": "error", "ename": "ValueError", "evalue": "deliberate",
                                             "traceback": ["ValueError: deliberate"], "code": "private program"}})
    action, details = actions[-1]
    assert action == "execution_finished"
    assert details == {"status": "error", "ename": "ValueError", "evalue": "deliberate", "traceback": ["ValueError: deliberate"]}
    screen._execution_finished({"content": {"status": "ok"}})
    assert actions[-1] == ("execution_finished", {"status": "ok"})


def test_kernel_death_reports_once_and_keeps_restart_available(screen):
    actions = []
    screen.action_requested.connect(lambda action, value: actions.append((action, value)))
    screen._connection_file = "retained-console.json"
    screen._kernel_died(3.0)
    screen._kernel_died(6.0)
    assert actions == [("kernel_died", {"reason": "heartbeat", "seconds_since_heartbeat": 3.0})]
    assert screen.restart_button.isEnabled()
    assert not screen.run_button.isEnabled()
    assert not screen.interrupt_button.isEnabled()
    assert "Restart" in screen.status.text()


def test_real_qtconsole_heartbeat_reports_user_code_crash(screen, tmp_path):
    pytest.importorskip("qtconsole")
    pytest.importorskip("jupyterlab")
    from mlcookbook.programming import ProgrammingSession

    session = ProgrammingSession(sys.executable, tmp_path)
    actions = []
    screen.action_requested.connect(lambda action, value: actions.append((action, value)))
    try:
        screen.set_runtime(session.start_kernel())
        assert not screen.client.hb_channel._pause
        screen.execute_code("import os; os._exit(9)")
        deadline = time.monotonic() + 15
        application = QApplication.instance()
        while not screen._kernel_dead and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(0.01)
        assert screen._kernel_dead
        assert any(action == "kernel_died" for action, _ in actions)
        assert screen.restart_button.isEnabled()
    finally:
        screen.detach_runtime()
        session.close()
