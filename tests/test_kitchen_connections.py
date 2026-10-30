"""Native connection gestures retain exact typed ports and editor history."""
import sys
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QDialogButtonBox, QGraphicsPathItem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui import theme
from gui.kitchen_canvas import NODE_PORTS
from gui.screens.kitchen import KitchenScreen
from mlcookbook.kitchen import default_settings


@pytest.fixture(scope="module")
def qt():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def screen(qt):
    previous = qt.styleSheet()
    qt.setStyleSheet(theme.stylesheet())
    screen = KitchenScreen({kind: default_settings(kind) for kind in NODE_PORTS})
    screen.resize(1600, 900)
    screen.show()
    screen.activateWindow()
    qt.processEvents()
    yield screen
    screen.close()
    screen.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt.setStyleSheet(previous)


def edge(source, source_port, target, target_port):
    return {"source": source, "source_port": source_port, "target": target, "target_port": target_port}


def install(screen, qt, *kinds, edges=()):
    keys = [f"n{index + 1}" for index in range(len(kinds))]
    screen.set_editor({
        "graph": {"version": 1, "nodes": [
            {"id": key, "kind": kind, "settings": default_settings(kind)} for key, kind in zip(keys, kinds)
        ], "edges": list(edges)},
        "layout": {key: [(index % 2) * 320, (index // 2) * 210] for index, key in enumerate(keys)},
        "selected": None,
    }, reset_history=True)
    screen.canvas.fit_graph()
    qt.processEvents()
    return keys


def point(screen, key, port=None, *, output=False):
    node = screen.canvas.nodes[key]
    if port is None:
        position = node.mapToScene(node.rect().center())
    else:
        position = (node.output_ports if output else node.input_ports)[port].scenePos()
    result = screen.canvas.mapFromScene(position)
    assert screen.canvas.viewport().rect().contains(result), (key, port, result)
    return result


def click(screen, qt, key, port=None, *, output=False):
    QTest.mouseClick(screen.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point(screen, key, port, output=output))
    qt.processEvents()


def drag(screen, qt, start, end):
    viewport = screen.canvas.viewport()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(viewport, (start + end) / 2, delay=1)
    QTest.mouseMove(viewport, end, delay=1)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
    qt.processEvents()


def no_preview(screen):
    committed = {line for line, _ in screen.canvas.edges}
    assert not [item for item in screen.canvas.scene().items()
                if isinstance(item, QGraphicsPathItem) and item not in committed
                and item.isVisible() and not item.path().isEmpty()]


def move_through_viewport(screen, qt, position, *, buttons=Qt.MouseButton.NoButton):
    viewport = screen.canvas.viewport()
    event = QMouseEvent(QEvent.Type.MouseMove, QPointF(position), QPointF(viewport.mapToGlobal(position)),
                        Qt.MouseButton.NoButton, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(viewport, event)
    qt.processEvents()


def test_live_hover_snaps_preview_to_exact_ports_and_hides_it_over_blank(screen, qt):
    source, target = install(screen, qt, "dataset", "select")
    click(screen, qt, source, "data", output=True)
    move_through_viewport(screen, qt, point(screen, target, "data") + QPoint(-14, 0))
    canvas = screen.canvas
    output = canvas.nodes[source].output_ports["data"]
    incoming = canvas.nodes[target].input_ports["data"]
    assert canvas._target_port is incoming and incoming.highlighted
    assert canvas._link is not None and canvas._link[1].isVisible()
    path = canvas._link[1].path()
    assert path.pointAtPercent(0) == output.scenePos()
    assert path.pointAtPercent(1) == incoming.scenePos()
    blank = canvas.viewport().rect().bottomRight() - QPoint(25, 25)
    move_through_viewport(screen, qt, blank)
    assert canvas._link is not None and canvas._target_port is None
    no_preview(screen)
    assert not screen.editor["graph"]["edges"] and not screen._undo


def test_live_held_drag_shows_temporary_wire_and_blank_release_cancels(screen, qt):
    source, _ = install(screen, qt, "dataset", "select")
    canvas = screen.canvas
    viewport = canvas.viewport()
    start = point(screen, source, "data", output=True)
    blank = viewport.rect().bottomRight() - QPoint(25, 25)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    move_through_viewport(screen, qt, blank, buttons=Qt.MouseButton.LeftButton)
    assert canvas._dragging and canvas._link is not None and canvas._link[1].isVisible()
    assert canvas._target_port is None
    path = canvas._link[1].path()
    assert path.pointAtPercent(0) == canvas.nodes[source].output_ports["data"].scenePos()
    assert path.pointAtPercent(1) == canvas.mapToScene(blank)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=blank)
    qt.processEvents()
    assert canvas._link is None
    no_preview(screen)
    assert not screen.editor["graph"]["edges"] and not screen._undo


@pytest.mark.parametrize("gesture", ["click", "drag"])
def test_connect_tool_infers_unique_types_without_moving_modules(screen, qt, gesture):
    source, target = install(screen, qt, "dataset", "add_column")
    positions = deepcopy(screen.editor["layout"])
    QTest.mouseClick(screen.connect_button, Qt.MouseButton.LeftButton)
    assert screen.connect_button.isChecked() and screen.canvas.connect_mode
    if gesture == "click":
        click(screen, qt, source)
        click(screen, qt, target)
    else:
        drag(screen, qt, point(screen, source), point(screen, target))
    assert screen.editor["graph"]["edges"] == [edge(source, "data", target, "data")]
    assert screen.editor["layout"] == positions == screen.canvas.positions()
    assert len(screen._undo) == 1 and not screen._redo
    no_preview(screen)


def test_connect_tool_allows_output_fanout_with_single_step_undo_and_redo(screen, qt):
    source, first, second = install(screen, qt, "dataset", "select", "select")
    screen.connect_button.click()
    for target in (first, second):
        click(screen, qt, source)
        click(screen, qt, target)
    expected = [edge(source, "data", first, "data"), edge(source, "data", second, "data")]
    assert screen.editor["graph"]["edges"] == expected
    assert len(screen._undo) == 2
    screen.canvas.setFocus()
    QTest.keyClick(screen.canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    qt.processEvents()
    assert screen.editor["graph"]["edges"] == expected[:1]
    QTest.keyClick(screen.canvas, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    qt.processEvents()
    assert screen.editor["graph"]["edges"] == expected


@pytest.mark.parametrize("kind", ["combine", "correlation"])
def test_ambiguous_target_requires_port_then_infers_only_remaining_free_input(screen, qt, kind):
    first, target, second = install(screen, qt, "map", kind, "map")
    screen.connect_button.click()
    click(screen, qt, first)
    click(screen, qt, target)
    assert not screen.editor["graph"]["edges"] and not screen._undo
    click(screen, qt, target, "right")
    assert screen.editor["graph"]["edges"] == [edge(first, "vector", target, "right")]
    click(screen, qt, second)
    click(screen, qt, target)
    assert screen.editor["graph"]["edges"][-1] == edge(second, "vector", target, "left")
    assert len(screen._undo) == 2


def test_multi_output_source_requires_explicit_missing_or_value_port(screen, qt):
    source, target = install(screen, qt, "column", "filter")
    screen.connect_button.click()
    click(screen, qt, source)
    assert not screen.editor["graph"]["edges"] and not screen._undo
    click(screen, qt, source, "missing", output=True)
    click(screen, qt, target, "mask")
    assert screen.editor["graph"]["edges"] == [edge(source, "missing", target, "mask")]


def test_clicking_another_output_restarts_the_pending_connection(screen, qt):
    first, second, target = install(screen, qt, "map", "map", "summary")
    positions = deepcopy(screen.editor["layout"])
    click(screen, qt, first, "vector", output=True)
    click(screen, qt, second, "vector", output=True)
    click(screen, qt, target, "vector")
    assert screen.editor["graph"]["edges"] == [edge(second, "vector", target, "vector")]
    assert screen.editor["layout"] == positions and len(screen._undo) == 1
    no_preview(screen)


@pytest.mark.parametrize("gesture", ["click", "drag"])
def test_precise_ports_work_without_connect_mode_and_preserve_exact_missing_output(screen, qt, gesture):
    source, target = install(screen, qt, "column", "filter")
    assert not screen.canvas.connect_mode
    if gesture == "click":
        click(screen, qt, source, "missing", output=True)
        click(screen, qt, target, "mask")
    else:
        drag(screen, qt, point(screen, source, "missing", output=True), point(screen, target, "mask"))
    assert screen.editor["graph"]["edges"] == [edge(source, "missing", target, "mask")]
    assert len(screen._undo) == 1
    no_preview(screen)


@pytest.mark.parametrize("zoom", [0.45, 1.6])
@pytest.mark.parametrize("distance,connected", [(14, True), (18, False)])
def test_port_snap_tolerance_uses_viewport_pixels_at_different_zooms(screen, qt, zoom, distance, connected):
    source, target = install(screen, qt, "dataset", "select")
    canvas = screen.canvas
    canvas.resetTransform()
    canvas.scale(zoom, zoom)
    canvas.centerOn(canvas.nodes[source].sceneBoundingRect().united(canvas.nodes[target].sceneBoundingRect()).center())
    qt.processEvents()
    start = point(screen, source, "data", output=True)
    end = point(screen, target, "data") + QPoint(-distance, 0)
    drag(screen, qt, start, end)
    expected = [edge(source, "data", target, "data")] if connected else []
    assert screen.editor["graph"]["edges"] == expected
    assert len(screen._undo) == int(connected)


def test_explicit_incompatible_port_does_not_snap_to_nearby_compatible_input(screen, qt):
    source, target = install(screen, qt, "map", "add_column")
    screen.canvas.resetTransform()
    screen.canvas.scale(0.4, 0.4)
    screen.canvas.centerOn(screen.canvas.nodes[source].sceneBoundingRect().united(
        screen.canvas.nodes[target].sceneBoundingRect()).center())
    qt.processEvents()
    incoming = point(screen, target, "data")
    assert (incoming - point(screen, target, "vector")).manhattanLength() < 16
    drag(screen, qt, point(screen, source, "vector", output=True), incoming)
    assert not screen.editor["graph"]["edges"] and not screen._undo
    no_preview(screen)


@pytest.mark.parametrize("case", ["type", "occupied", "self", "cycle"])
def test_invalid_precise_gestures_preserve_graph_layout_and_history(screen, qt, case):
    if case == "type":
        source, target = install(screen, qt, "map", "select")
        output, incoming = "vector", "data"
    elif case == "occupied":
        _, source, target = install(screen, qt, "dataset", "dataset", "select",
                                    edges=[edge("n1", "data", "n3", "data")])
        output, incoming = "data", "data"
    elif case == "self":
        source, = install(screen, qt, "select")
        target, output, incoming = source, "data", "data"
    else:
        target, source = install(screen, qt, "select", "select", edges=[edge("n1", "data", "n2", "data")])
        output, incoming = "data", "data"
    before = screen.state()
    click(screen, qt, source, output, output=True)
    click(screen, qt, target, incoming)
    assert screen.editor["graph"] == before["graph"]
    assert screen.editor["layout"] == before["layout"]
    assert not screen._undo and not screen._redo
    assert screen.status.text()
    no_preview(screen)


def test_dragging_to_blank_cancels_without_leaving_a_wire_or_armed_connection(screen, qt):
    source, target = install(screen, qt, "dataset", "select")
    blank = screen.canvas.viewport().rect().bottomRight() - QPoint(25, 25)
    drag(screen, qt, point(screen, source, "data", output=True), blank)
    assert not screen.editor["graph"]["edges"] and not screen._undo
    no_preview(screen)
    click(screen, qt, target, "data")
    assert not screen.editor["graph"]["edges"]


@pytest.mark.parametrize("cancel", ["escape", "tool_off", "busy", "hide", "redraw"])
def test_pending_connection_cancels_on_editor_lifecycle_changes(screen, qt, cancel):
    source, target = install(screen, qt, "dataset", "select")
    screen.connect_button.click()
    click(screen, qt, source, "data", output=True)
    blank = screen.canvas.viewport().rect().bottomRight() - QPoint(25, 25)
    QTest.mouseMove(screen.canvas.viewport(), blank)
    qt.processEvents()
    no_preview(screen)
    if cancel == "escape":
        QTest.keyClick(screen.canvas, Qt.Key.Key_Escape)
        assert not screen.canvas.connect_mode and not screen.connect_button.isChecked()
    elif cancel == "tool_off":
        screen.connect_button.click()
        assert not screen.canvas.connect_mode
    elif cancel == "busy":
        screen.set_busy(True)
        assert not screen.connect_button.isEnabled() and not screen.canvas.isEnabled()
        screen.connect_nodes(edge(source, "data", target, "data"))
        assert not screen.editor["graph"]["edges"]
        screen.set_busy(False)
    elif cancel == "hide":
        screen.hide()
        qt.processEvents()
        screen.show()
        screen.canvas.fit_graph()
    else:
        screen.set_editor(screen.state())
    qt.processEvents()
    click(screen, qt, target, "data")
    assert not screen.editor["graph"]["edges"] and not screen._undo
    no_preview(screen)


def test_default_node_drag_and_background_pan_remain_available(screen, qt):
    source, _ = install(screen, qt, "dataset", "select")
    start = point(screen, source)
    drag(screen, qt, start, start + QPoint(35, 24))
    assert screen.editor["layout"][source] != [0, 0]
    assert screen.editor["layout"] == screen.canvas.positions()
    assert len(screen._undo) == 1 and not screen.editor["graph"]["edges"]
    positions = deepcopy(screen.editor["layout"])
    canvas = screen.canvas
    before = (canvas.horizontalScrollBar().value(), canvas.verticalScrollBar().value())
    blank = canvas.viewport().rect().bottomRight() - QPoint(25, 25)
    drag(screen, qt, blank, blank - QPoint(45, 30))
    after = (canvas.horizontalScrollBar().value(), canvas.verticalScrollBar().value())
    assert before != after
    assert screen.editor["layout"] == positions and len(screen._undo) == 1


def test_armed_source_highlights_only_free_compatible_inputs(screen, qt):
    source, free, incompatible, occupied = install(screen, qt, "dataset", "select", "map", "select",
                                                   edges=[edge("n1", "data", "n4", "data")])
    click(screen, qt, source, "data", output=True)
    assert screen.canvas.nodes[free].input_ports["data"].highlighted
    assert not screen.canvas.nodes[incompatible].input_ports["vector"].highlighted
    assert not screen.canvas.nodes[occupied].input_ports["data"].highlighted
    QTest.keyClick(screen.canvas, Qt.Key.Key_Escape)
    assert all(not port.highlighted for node in screen.canvas.nodes.values()
               for ports in (node.input_ports, node.output_ports) for port in ports.values())


def test_cycle_and_self_inputs_are_excluded_from_highlights(screen, qt):
    ancestor, source, free = install(screen, qt, "select", "select", "select",
                                     edges=[edge("n1", "data", "n2", "data")])
    click(screen, qt, source, "data", output=True)
    assert not screen.canvas.nodes[ancestor].input_ports["data"].highlighted
    assert not screen.canvas.nodes[source].input_ports["data"].highlighted
    assert screen.canvas.nodes[free].input_ports["data"].highlighted


def test_explicit_connection_dialog_remains_a_precise_fallback(screen, qt):
    source, target = install(screen, qt, "column", "filter")
    errors = []

    def choose():
        dialog = qt.activeModalWidget()
        try:
            assert dialog is not None
            output, incoming = dialog.findChildren(QComboBox)
            output.setCurrentIndex(next(i for i in range(output.count()) if output.itemData(i) == (source, "missing")))
            incoming.setCurrentIndex(next(i for i in range(incoming.count()) if incoming.itemData(i) == (target, "mask")))
            assert output.currentData() == (source, "missing")
            assert incoming.currentData() == (target, "mask")
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
        except BaseException as exc:
            errors.append(exc)
            if dialog is not None:
                dialog.reject()

    QTimer.singleShot(0, choose)
    screen.canvas.setFocus()
    QTest.keyClick(screen.canvas, Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier)
    qt.processEvents()
    assert not errors
    assert screen.editor["graph"]["edges"] == [edge(source, "missing", target, "mask")]
    assert len(screen._undo) == 1


def test_busy_editor_disables_keyboard_dialog_and_refuses_direct_dialog_entry(screen, qt, monkeypatch):
    install(screen, qt, "dataset", "select")
    screen.set_busy(True)
    assert not screen.connect_action.isEnabled()
    monkeypatch.setattr(QDialog, "exec", lambda self: pytest.fail("A busy editor must not open connection dialogs"))
    screen._connect_dialog()
    QTest.keyClick(screen.canvas, Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier)
    qt.processEvents()
    assert not screen.editor["graph"]["edges"] and not screen._undo
