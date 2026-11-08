"""The native programming UI exchanges explicit ordinary dataset copies."""
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.skipif(importlib.util.find_spec("qtconsole") is None, reason="optional programming extra")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402
from mlcookbook import Dataset  # noqa: E402


def finish(controller, qt):
    deadline = time.monotonic() + 50
    while (controller.job is not None or controller._programming_executing) and time.monotonic() < deadline:
        qt.processEvents()
        time.sleep(.005)
    qt.processEvents()
    assert controller.job is None and not controller._programming_executing


def test_console_script_and_explicit_return_keep_source_independent(tmp_path):
    qt = QApplication.instance() or QApplication([])
    controller = app.MLCookbookApp()
    try:
        original = Dataset(pd.DataFrame({"id": ["001", "002", "003"], "value": [1., 2., 3.]}, index=[9, 9, 9]))
        view = app._dataset_view(original, "ordinary")
        view["source"] = {"kind": "snapshot"}
        controller._use_import((original, "ordinary", view))
        controller.show_programming()
        screen = controller.screen
        assert screen.console is None and controller.programming_session is None
        controller.programming_action("start", {"python": sys.executable, "workspace": str(tmp_path)})
        finish(controller, qt)
        assert screen.console is not None, screen.status.text()
        controller.programming_action("send_data", {})
        finish(controller, qt)
        screen.execute_code("df['doubled'] = df['value'] * 2")
        finish(controller, qt)
        screen.execute_code("raise ValueError('expected console error')")
        finish(controller, qt)
        assert controller.activity.view()[-1]["outcome"] == "Failed"
        assert "expected console error" in controller.activity.view()[-1]["error"]
        controller.programming_action("publish", {"variable": "df"})
        finish(controller, qt)
        assert controller.programming_return[0].df.doubled.tolist() == [2., 4., 6.]
        assert controller.imported[0] is original and "doubled" not in original.df
        controller.programming_action("accept_return", {})
        assert controller.imported[0].df.index.tolist() == [9, 9, 9]
        assert controller.imported[0].df.id.tolist() == ["001", "002", "003"]
        assert controller.imported[2]["source"]["kind"] == "programming"
        script = tmp_path / "practice.py"
        controller.programming_action("save_script", {"path": str(script), "text": "answer = 42\n"})
        assert script.read_text(encoding="utf-8") == "answer = 42\n"
        controller.programming_action("restart", {})
        finish(controller, qt)
        assert screen.console is not None
        session = controller.programming_session
        session.execute("assert 'answer' not in globals()")
        screen.execute_code("import os; os._exit(12)")
        finish(controller, qt)
        assert controller.activity.view()[-1]["outcome"] == "Failed"
        assert not controller._programming_executing
        assert controller.window.navigation_buttons["data"].isEnabled()
        controller.programming_action("restart", {})
        finish(controller, qt)
        assert controller.screen.console is not None
        controller.navigate("data")
        assert controller.key == "data" and controller.programming_session is session
        controller.navigate("programming")
        assert controller.screen.console is not None
    finally:
        controller.wait_for_job()
        qt.processEvents()
        controller.close_programming()
        controller.window.close()
        controller.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_job_failure_retains_traceback_after_navigation():
    qt = QApplication.instance() or QApplication([])
    controller = app.MLCookbookApp()
    try:
        def fail():
            raise ValueError("A deliberate failure")
        controller._start_job(fail, lambda value: None, "Testing failure history")
        finish(controller, qt)
        controller.show_front_page()
        record = controller.activity.view()[-1]
        assert record["outcome"] == "Failed" and "Traceback" in record["details"]
        assert "A deliberate failure" in record["error"]
        assert controller.window.activity_panel.operations.topLevelItemCount() == 1
    finally:
        controller.close_programming()
        controller.window.close()
        controller.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_reopened_notebook_explicit_return_preserves_console_and_active_input(tmp_path, monkeypatch):
    from mlcookbook.programming import DatasetBridge, ProgrammingSession

    original = Dataset(pd.DataFrame({"id": ["001", "002"], "value": [2., 3.]}, index=[7, 7]))
    prior = ProgrammingSession(sys.executable, tmp_path)
    monkeypatch.setattr(prior, "execute", lambda code: {})
    prior.send_dataset(original, source={"kind": "fixture", "name": "saved notebook input"})
    notebook = prior.create_notebook("saved.ipynb")
    notebook_bytes = notebook.read_bytes()
    prior.close()
    qt = QApplication.instance() or QApplication([])
    controller = app.MLCookbookApp()
    current = ProgrammingSession(sys.executable, tmp_path)
    try:
        active = Dataset(pd.DataFrame({"ordinary": [11., 12.]}))
        controller._use_import((active, "ordinary", app._dataset_view(active, "ordinary")))
        controller.show_programming()
        controller.programming_session = current
        monkeypatch.setattr(current, "execute", lambda *args, **kwargs: pytest.fail("Reopening must not execute notebook cells"))
        monkeypatch.setattr(current, "start_notebook_server", lambda: {})
        monkeypatch.setattr(current, "notebook_url", lambda path: "http://127.0.0.1:18888/lab/tree/saved.ipynb")
        opened = []
        monkeypatch.setattr(controller.screen, "set_notebook_url", lambda url, **kwargs: opened.append((url, kwargs)))
        controller.programming_action("open_notebook", {"path": str(notebook)})
        finish(controller, qt)
        assert controller.programming_notebook == notebook
        assert opened[0][1] == {"notebook_path": str(notebook)}
        assert notebook.read_bytes() == notebook_bytes
        DatasetBridge(current.bridge_directory, current.snapshot_id).publish(pd.DataFrame({"console": [99]}))
        bridge = DatasetBridge(prior.bridge_directory, prior.snapshot_id)
        frame = bridge.load()[0].df.copy()
        frame["doubled"] = frame.value * 2
        bridge.publish(frame)
        controller.programming_action("preview_return", {"context": "notebook", "notebook": str(notebook)})
        finish(controller, qt)
        returned = controller.programming_return
        assert returned[0].df.doubled.tolist() == [4., 6.]
        assert returned[2]["source"]["snapshot_id"] == prior.snapshot_id != current.snapshot_id
        assert returned[2]["source"]["input_source"]["name"] == "saved notebook input"
        assert "saved.ipynb" in controller.screen.status.text()
        assert controller.imported[0] is active and "doubled" not in original.df
        controller.programming_action("accept_return", {})
        assert controller.imported[0] is returned[0]
        assert current.read_published_dataset()[0].df.console.tolist() == [99]
        assert notebook.read_bytes() == notebook_bytes
    finally:
        controller.wait_for_job()
        qt.processEvents()
        controller.close_programming()
        controller.window.close()
        controller.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
