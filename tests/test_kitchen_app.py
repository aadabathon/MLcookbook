"""The native Kitchen composes real operations without disturbing recipe sessions."""
import sys
import time
from pathlib import Path
from threading import Event

import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from gui import theme
from gui.plot import PlotWidget
from gui.screens.kitchen import KitchenScreen


@pytest.fixture(scope="module")
def qt():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(qt):
    previous = qt.styleSheet()
    qt.setStyleSheet(theme.stylesheet())
    controller = app.MLCookbookApp()
    controller.window.resize(1080, 720)
    controller.window.show()
    yield controller
    controller.wait_for_job()
    qt.processEvents()
    controller.window.close()
    controller.window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt.setStyleSheet(previous)


def finish(controller, qt):
    deadline = time.monotonic() + 30
    while controller.job is not None and time.monotonic() < deadline:
        qt.processEvents()
        time.sleep(0.005)
    assert controller.job is None
    qt.processEvents()


def add(screen, kind):
    screen.node_kind.setCurrentIndex(screen.node_kind.findData(kind))
    screen.add_button.click()
    return screen.editor["selected"]


def select(screen, key):
    screen.canvas.scene().clearSelection()
    screen.canvas.nodes[key].setSelected(True)


def setup_graph(controller, qt, tmp_path):
    path = tmp_path / "own.csv"
    pd.DataFrame({"x": range(40), "y": [i * 3 + i % 3 for i in range(40)]}).to_csv(path, index=False)
    controller.load_file(str(path), ",", "utf-8", "")
    finish(controller, qt)
    controller.navigate("kitchen")
    finish(controller, qt)
    screen = controller.screen
    assert isinstance(screen, KitchenScreen)
    source = add(screen, "dataset")
    screen.capture_button.click()
    finish(controller, qt)
    assert screen.capture_button.isEnabled()
    selected = add(screen, "select")
    screen.connect_nodes({"source": source, "source_port": "data", "target": selected, "target_port": "data"})
    for row in range(screen.controls["columns"].rowCount()):
        screen.controls["columns"].item(row, 0).setCheckState(Qt.CheckState.Checked)
    comparison = add(screen, "comparison")
    screen.connect_nodes({"source": selected, "source_port": "data", "target": comparison, "target_port": "data"})
    screen.controls["target"].setCurrentIndex(screen.controls["target"].findData("y"))
    screen.controls["features"].item(0, 0).setCheckState(Qt.CheckState.Checked)
    screen.controls["n_estimators"].setValue(3)
    screen.controls["cv_folds"].setValue(2)
    report = add(screen, "report")
    screen.connect_nodes({"source": comparison, "source_port": "result", "target": report, "target_port": "result"})
    return path, (source, selected, comparison, report)


def test_native_graph_run_edit_undo_save_and_source_free_reopen(controller, qt, tmp_path, monkeypatch):
    import mlcookbook.kitchen as kitchen

    path, keys = setup_graph(controller, qt, tmp_path)
    screen = controller.screen
    original, calls = kitchen.compare_regressors, []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(kitchen, "compare_regressors", counted)
    screen.run_button.click()
    finish(controller, qt)
    assert len(calls) == 1
    assert set(screen._states.values()) == {"completed"}
    assert screen.warning_button.isVisibleTo(screen) and not screen.result_state.isVisibleTo(screen)
    assert not controller.window.save_experiment_button.isEnabled()
    assert len(screen.output.findChildren(PlotWidget)) == 2
    select(screen, keys[2])
    screen.inspector.setCurrentIndex(1)
    screen.controls["alpha"].setValue(2.0)
    assert screen._states[keys[2]] == screen._states[keys[3]] == "stale"
    assert screen.result_state.isVisibleTo(screen) and "last completed" in screen.result_state.text()
    screen.undo()
    assert screen._states[keys[2]] == "completed" and screen.controls["alpha"].value() == 1.0
    screen.redo()
    assert screen._states[keys[3]] == "stale" and len(calls) == 1
    saved = tmp_path / "work.mlck"
    controller.save_kitchen(str(saved))
    finish(controller, qt)
    before = controller.kitchen.run.to_dict()
    path.unlink()
    ordinary = controller.imported
    controller.navigate("data")
    controller.open_kitchen(str(saved))
    finish(controller, qt)
    assert isinstance(controller.screen, KitchenScreen)
    assert controller.kitchen.run.to_dict() == before and len(calls) == 1
    assert controller.screen._states[keys[0]] == controller.screen._states[keys[1]] == "completed"
    assert controller.screen._states[keys[3]] == "stale"
    assert controller.imported is ordinary
    assert controller.screen.controls["alpha"].value() == 2.0
    assert "without execution" in controller.screen.status.text()
    old_doc, old_screen = controller.kitchen, controller.screen
    bad = tmp_path / "bad.mlck"
    bad.write_text("broken", encoding="utf-8")
    controller.open_kitchen(str(bad))
    finish(controller, qt)
    assert controller.kitchen is old_doc and controller.screen is old_screen
    assert "Unable to complete" in controller.screen.status.text()


def test_job_locks_canvas_routes_and_cancels_only_after_current_model(controller, qt, tmp_path, monkeypatch):
    import mlcookbook.kitchen as kitchen

    _, keys = setup_graph(controller, qt, tmp_path)
    entered, release = Event(), Event()
    original = kitchen.compare_regressors

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(kitchen, "compare_regressors", held)
    controller.screen.run_button.click()
    deadline = time.monotonic() + 5
    while not entered.is_set() and time.monotonic() < deadline:
        qt.processEvents()
        time.sleep(0.005)
    try:
        assert entered.is_set()
        screen = controller.screen
        assert not screen.canvas.isEnabled() and not screen.add_button.isEnabled()
        assert not screen.save_button.isEnabled()
        assert all(not b.isEnabled() for b in controller.window.navigation_buttons.values())
        screen.cancel_button.click()
        assert "current model operation" in screen.status.text()
    finally:
        release.set()
    finish(controller, qt)
    assert controller.kitchen.run.attempts[keys[2]]["state"] == "completed"
    assert controller.kitchen.run.attempts[keys[3]]["state"] == "cancelled"
    assert keys[2] in controller.kitchen.run.records and keys[3] not in controller.kitchen.run.records
    assert controller.screen.canvas.isEnabled() and controller.screen.save_button.isEnabled()


def test_canvas_port_gesture_keyboard_delete_undo_and_layout_survive_navigation(controller, qt):
    controller.navigate("kitchen")
    finish(controller, qt)
    screen = controller.screen
    first, second = add(screen, "dataset"), add(screen, "select")
    screen.canvas.fit_graph()
    qt.processEvents()
    source = screen.canvas.mapFromScene(screen.canvas.nodes[first].ports[True].scenePos())
    target = screen.canvas.mapFromScene(screen.canvas.nodes[second].ports[False].scenePos())
    QTest.mouseClick(screen.canvas.viewport(), Qt.MouseButton.LeftButton, pos=source)
    QTest.mouseClick(screen.canvas.viewport(), Qt.MouseButton.LeftButton, pos=target)
    assert len(controller.kitchen.graph.edges) == 1
    screen.canvas.nodes[first].setPos(31, 42)
    screen.canvas.moved.emit(screen.canvas.positions())
    select(screen, first)
    screen.canvas.setFocus()
    QTest.keyClick(screen.canvas, Qt.Key.Key_Delete)
    assert len(controller.kitchen.graph.nodes) == 1 and not controller.kitchen.graph.edges
    screen.undo()
    assert len(controller.kitchen.graph.nodes) == 2 and len(controller.kitchen.graph.edges) == 1
    controller.navigate("data")
    controller.navigate("kitchen")
    assert controller.screen.canvas.nodes[first].pos() == QPoint(31, 42)
    assert controller.screen.undo_button.isEnabled()


def test_ports_reject_type_mismatch_and_cycles_without_mutating_graph(controller, qt):
    controller.navigate("kitchen")
    finish(controller, qt)
    screen = controller.screen
    a, b, report = add(screen, "select"), add(screen, "select"), add(screen, "report")
    screen.connect_nodes({"source": a, "source_port": "data", "target": report, "target_port": "result"})
    assert not controller.kitchen.graph.edges and "Incompatible" in screen.status.text()
    screen.connect_nodes({"source": a, "source_port": "data", "target": b, "target_port": "data"})
    screen.connect_nodes({"source": b, "source_port": "data", "target": a, "target_port": "data"})
    assert len(controller.kitchen.graph.edges) == 1 and "cycle" in screen.status.text()


def test_cold_kitchen_initialization_stays_in_worker_and_retries_from_data(controller, qt, monkeypatch):
    from PySide6.QtCore import QThread

    entered, release = Event(), Event()
    previous = controller.screen
    previous.markers.setText("?, -999")
    controller._runtime_initialized = False
    calls, workers = [], []

    def prepare():
        calls.append(True)
        workers.append(QThread.currentThread())
        entered.set()
        assert release.wait(10)
        if len(calls) == 1:
            raise RuntimeError("initialization test failure")

    monkeypatch.setattr(app, "_prepare_runtime", prepare)
    controller.navigate("kitchen")
    assert entered.wait(2)
    try:
        assert controller.screen is previous and controller.job is not None
        assert "Preparing Kitchen" in previous.status.text()
        assert not previous.load_button.isEnabled()
        assert all(not button.isEnabled() for button in controller.window.navigation_buttons.values())
    finally:
        release.set()
        finish(controller, qt)
    assert controller.screen is previous and "initialization test failure" in previous.status.text()
    assert previous.load_button.isEnabled()
    controller.navigate("kitchen")
    finish(controller, qt)
    assert isinstance(controller.screen, KitchenScreen) and controller._runtime_initialized
    assert len(calls) == 2 and all(worker != qt.thread() for worker in workers)
    assert controller.data_options["markers"] == "?, -999"
    assert not controller.screen.editor["graph"]["nodes"] and not controller.screen.run_button.isEnabled()


def test_fit_graph_keeps_nodes_readable_and_retired_capture_controls_are_not_reused(controller, qt, tmp_path):
    setup_graph(controller, qt, tmp_path)
    screen = controller.screen
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt.processEvents()
    assert screen.capture_button is None
    controller._refresh_shell()  # must not access the retired dataset inspector's button
    screen.fit_button.click()
    qt.processEvents()
    for item in screen.canvas.nodes.values():
        bounds = screen.canvas.mapFromScene(item.sceneBoundingRect()).boundingRect()
        assert bounds.width() >= 145


def test_preprocessing_dialog_retains_two_folds_and_unrun_settings(controller, qt, tmp_path):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDialog

    _, keys = setup_graph(controller, qt, tmp_path)
    screen = controller.screen
    select(screen, keys[2])
    seen = []

    def accept():
        dialog = qt.activeModalWidget()
        assert isinstance(dialog, QDialog)
        seen.append(dialog.cv.currentData())
        dialog.combos["scaling"].setCurrentIndex(dialog.combos["scaling"].findData("robust"))
        dialog.accept()

    QTimer.singleShot(0, accept)
    screen._preprocessing()
    assert seen == [2]
    settings = next(n.settings for n in controller.kitchen.graph.nodes if n.id == keys[2])
    assert settings["preprocessing"]["scaling"] == "robust" and settings["cv_folds"] == 2


def test_open_rejects_unrepresentable_api_settings_instead_of_clamping_them(controller, qt, tmp_path):
    from dataclasses import replace

    from mlcookbook.kitchen_io import save_kitchen

    _, keys = setup_graph(controller, qt, tmp_path)
    original, screen = controller.kitchen, controller.screen
    changed = replace(original.graph, nodes=[replace(n, settings=n.settings | {"alpha": 1.234567}) if n.id == keys[2]
                                             else n for n in original.graph.nodes])
    path = tmp_path / "api-only.mlck"
    save_kitchen(path, replace(original, graph=changed))
    controller.open_kitchen(str(path))
    finish(controller, qt)
    assert controller.kitchen is original and controller.screen is screen
    assert "cannot be restored exactly" in screen.status.text()


def test_kitchen_open_preserves_study_sessions_and_ordinary_data(controller, qt, tmp_path, monkeypatch):
    from tests.test_wearable import make_export

    _, keys = setup_graph(controller, qt, tmp_path)
    graph_file = tmp_path / "graph.mlck"
    controller.save_kitchen(str(graph_file))
    finish(controller, qt)
    ordinary = controller.imported
    controller.navigate("wearable-study")
    study = make_export(tmp_path / "study")
    controller.load_wearable_study(str(study / "export.json"))
    finish(controller, qt)
    assert controller.imported is ordinary and not controller.screen.request()["params"]["final_test"]
    controller.screen.param_widgets["random_state"].setValue(17)
    monkeypatch.setattr(app, "_run_report", lambda *a, **kw: pytest.fail("Navigation/open must not evaluate a study"))
    controller.open_kitchen(str(graph_file))
    finish(controller, qt)
    assert controller.imported is ordinary and controller.kitchen.selected == keys[3]
    assert all("__wearable_split" not in data.df.columns for data in controller.kitchen.snapshots.values())
    controller.navigate("wearable-study")
    assert controller.screen.source["kind"] == "wearable_study"
    assert controller.screen.request()["params"]["random_state"] == 17
    assert not controller.screen.request()["params"]["final_test"]


def test_large_canvas_positions_remain_reachable_and_fit_can_be_zoomed(controller, qt):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QWheelEvent

    controller.navigate("kitchen")
    finish(controller, qt)
    screen = controller.screen
    first, second = add(screen, "dataset"), add(screen, "select")
    item = screen.canvas.nodes[second]
    item.setPos(18000, -14000)
    screen.canvas.moved.emit(screen.canvas.positions())
    assert screen.canvas.sceneRect().contains(item.sceneBoundingRect())
    screen.canvas.fit_graph()
    qt.processEvents()
    assert screen.canvas.transform().m11() < 0.1
    event = QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier,
                        Qt.ScrollPhase.ScrollUpdate, False)
    QApplication.sendEvent(screen.canvas.viewport(), event)
    assert screen.canvas.transform().m11() >= 0.1
    screen.canvas.ensureVisible(item)
    qt.processEvents()
    assert screen.canvas.viewport().rect().contains(screen.canvas.mapFromScene(item.sceneBoundingRect().center()))
    assert first in screen.canvas.nodes


@pytest.mark.parametrize("action", ["capture", "save", "open"])
def test_non_cancellable_file_jobs_do_not_show_or_enable_cancel(controller, qt, tmp_path, monkeypatch, action):
    import mlcookbook.kitchen_io as storage

    _, keys = setup_graph(controller, qt, tmp_path)
    saved = tmp_path / "file.mlck"
    controller.save_kitchen(str(saved))
    finish(controller, qt)
    entered, release = Event(), Event()
    owner, name = (app, "_dataset_view") if action == "capture" else (storage, "save_kitchen" if action == "save" else "load_kitchen")
    original = getattr(owner, name)

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, held)
    if action == "capture":
        select(controller.screen, keys[0])
        controller.screen.capture_button.click()
    elif action == "save":
        controller.save_kitchen(str(saved))
    else:
        controller.open_kitchen(str(saved))
    assert entered.wait(2)
    try:
        assert controller.screen.cancel_button.isHidden() and not controller.screen.cancel_button.isEnabled()
        controller.screen.cancel_button.click()
        assert not controller.job.isInterruptionRequested()
    finally:
        release.set()
        finish(controller, qt)
