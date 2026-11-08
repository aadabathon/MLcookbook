"""Research navigation and explicit reading actions preserve analytical sessions."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402


@pytest.fixture(scope="module")
def qt():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(qt):
    controller = app.MLCookbookApp()
    yield controller
    controller.wait_for_job()
    controller.window.close()
    controller.window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_research_keeps_data_options_and_recipe_sessions(controller, qt):
    from mlcookbook.research import timeline_entries

    controller.screen.markers.setText("?, -999")
    marker = object()
    controller.sessions["retained"] = marker
    controller.workspace_sessions["retained"] = marker
    controller.example_sessions["retained"] = marker
    controller.kitchen = marker
    controller.window.navigation_buttons["research"].click()
    assert controller.key == "research" and controller.job is None
    assert controller.window.navigation_buttons["research"].isChecked()
    assert not controller.window.save_experiment_button.isEnabled()
    selected = next(entry for entry in timeline_entries() if "transformer" in entry["paper_ids"])
    controller.screen.restore_session({"query": "attention", "era": "", "papers_only": True,
                                       "selected": selected["id"], "scroll": 0})
    expected = controller.screen.session_state()
    controller.navigate("data")
    assert controller.screen.markers.text() == "?, -999"
    controller.navigate("research")
    qt.processEvents()
    assert controller.screen.session_state() == expected
    assert controller.imported is None and controller.kitchen is marker
    assert all(sessions["retained"] is marker for sessions in
               (controller.sessions, controller.workspace_sessions, controller.example_sessions))


def test_local_reading_requires_explicit_action_and_current_catalog_file(controller, monkeypatch, tmp_path):
    from mlcookbook.research import paper_specs

    paper = paper_specs()[0]
    path = tmp_path / paper["filename"]
    path.write_bytes(b"%PDF-1.4\n")
    opened = []
    monkeypatch.setattr(app.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    controller.research_directory = tmp_path
    controller.navigate("research")
    assert opened == []
    controller.screen.paper_requested.emit(paper["id"])
    assert len(opened) == 1 and opened[0].isLocalFile()
    assert Path(opened[0].toLocalFile()) == path.resolve()
    path.unlink()
    controller.screen.paper_requested.emit(paper["id"])
    assert len(opened) == 1 and controller.screen._status_message
    assert not next(item for item in controller.screen.papers if item["id"] == paper["id"])["available"]
    controller.screen.paper_requested.emit("../outside.pdf")
    assert len(opened) == 1


def test_sources_are_catalog_links_and_browser_failures_are_visible(controller, monkeypatch):
    from mlcookbook.research import timeline_entries

    opened = []
    monkeypatch.setattr(app.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    controller.navigate("research")
    controller.screen.source_requested.emit("file:///C:/Windows/notepad.exe")
    controller.screen.source_requested.emit("https://example.invalid/unlisted")
    assert opened == []
    source = timeline_entries()[0]["sources"][0]["url"]
    controller.screen.source_requested.emit(source)
    assert len(opened) == 1 and opened[0].toString() == source
    monkeypatch.setattr(app.QDesktopServices, "openUrl", lambda url: False)
    controller.screen.source_requested.emit(source)
    assert "browser" in controller.screen._status_message.lower()


def test_choose_library_refreshes_only_research(controller, monkeypatch, tmp_path):
    from mlcookbook.research import paper_specs

    (tmp_path / paper_specs()[0]["filename"]).write_bytes(b"%PDF-1.4\n")
    controller.navigate("research")
    controller.screen.search.setText("attention")
    state = controller.screen.session_state()
    monkeypatch.setattr(app.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    controller.screen.library_requested.emit()
    assert controller.research_directory == tmp_path
    assert sum(paper["available"] for paper in controller.screen.papers) == 1
    assert controller.screen.session_state() == state
    assert controller.imported is None and controller.job is None
    monkeypatch.setattr(app.QFileDialog, "getExistingDirectory", lambda *args: "")
    controller.screen.library_requested.emit()
    assert controller.research_directory == tmp_path


@pytest.mark.parametrize("zoom", [1.0, 1.2])
def test_full_timeline_pan_restores_after_installing_page_in_visible_shell(controller, monkeypatch, qt, zoom):
    """A real shell resize must not overwrite the saved graph's distant pan."""
    def flush():
        qt.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qt.processEvents()

    opened = []
    monkeypatch.setattr(app.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    controller.window.resize(1280, 860)
    controller.window.show()
    flush()
    controller.navigate("research")
    flush()
    controller.screen.restore_session({"query": "", "era": "", "track": "", "view": "timeline",
                                       "papers_only": False, "selected": "transformer", "scroll": 0})
    flush()
    graph = controller.screen.graphic
    graph.scale(zoom, zoom)
    node = graph.node_items["transformer"]
    graph.centerOn(node)
    flush()
    point = graph.mapFromScene(node.mapToScene(node.action_rect.center()))
    assert graph.viewport().rect().contains(point)
    QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, pos=point)
    flush()
    expected = controller.screen.session_state()
    assert expected["graphic"]["x"] > 5000 and expected["graphic"]["y"] > 0
    assert len(opened) == 1
    controller.navigate("data")
    flush()
    controller.navigate("research")
    for _ in range(3):
        flush()
    assert controller.screen.session_state() == expected
    assert len(opened) == 1
    assert not controller.screen._restore_pending
    # A later hide/show must not replay that old saved position over a new pan.
    controller.screen.graphic.horizontalScrollBar().setValue(expected["graphic"]["x"] - 200)
    changed = controller.screen.graphic.view_state()
    controller.screen.hide()
    controller.screen.show()
    flush()
    assert controller.screen.graphic.view_state() == changed


def test_research_actions_and_navigation_lock_during_existing_jobs(controller, qt, monkeypatch):
    opened = []
    monkeypatch.setattr(app.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    controller.navigate("research")
    previous = controller.screen
    completed = []
    controller._start_job(lambda: "ready", completed.append, "Checking…")
    controller.navigate("data")
    controller.open_research_paper("transformer")
    controller.choose_research_library()
    assert controller.screen is previous and opened == []
    assert all(not button.isEnabled() for button in controller.window.navigation_buttons.values())
    deadline = time.monotonic() + 10
    while controller.job is not None and time.monotonic() < deadline:
        qt.processEvents()
        time.sleep(.005)
    assert controller.job is None and completed == ["ready"]
    assert previous.search.isEnabled()
    assert all(button.isEnabled() for button in controller.window.navigation_buttons.values())


def test_research_is_offline_and_opens_without_scientific_or_inference_imports():
    code = textwrap.dedent("""
        import importlib.abc
        import sys
        forbidden = ('numpy', 'pandas', 'scipy', 'sklearn', 'pyarrow', 'torch', 'transformers', 'mlcookbook.core')
        class ForbidHeavyImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
                    raise AssertionError('Research imported analytical runtime: ' + fullname)
        sys.meta_path.insert(0, ForbidHeavyImports())
        import socket
        socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Network access'))
        import app
        from PySide6.QtCore import QEvent
        application = app.QApplication([])
        controller = app.MLCookbookApp()
        controller.navigate('research')
        assert controller.key == 'research' and controller.job is None
        assert controller.screen.timeline.count() > 25
        controller.navigate('data')
        controller.window.close()
        controller.window.deleteLater()
        app.QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert not any(module == name or module.startswith(name + '.')
                       for module in sys.modules for name in forbidden)
        print('Research opens offline without scientific imports')
    """)
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Research opens offline" in completed.stdout


def test_timeline_card_opens_checked_local_paper_then_online_source(controller, monkeypatch, tmp_path, qt):
    from mlcookbook.research import paper_specs, resolve_library, timeline_entries

    paper = next(item for item in paper_specs() if item["id"] == "transformer")
    entry = next(item for item in timeline_entries() if "transformer" in item["paper_ids"])
    path = tmp_path / paper["filename"]
    path.write_bytes(b"%PDF-1.4\n")
    opened = []
    monkeypatch.setattr(app.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    controller.research_directory = tmp_path
    controller.navigate("research")
    controller.window.show()
    controller.screen.search.setText(paper["title"])
    qt.processEvents()
    assert opened == []

    def click_reading():
        graph = controller.screen.graphic
        node = graph.node_items[entry["id"]]
        graph.centerOn(node)
        qt.processEvents()
        point = graph.mapFromScene(node.mapToScene(node.action_rect.center()))
        assert graph.viewport().rect().contains(point)
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, pos=point)
        qt.processEvents()

    click_reading()
    assert len(opened) == 1 and opened[0].isLocalFile()
    assert Path(opened[0].toLocalFile()) == path.resolve()
    path.unlink()
    controller.screen.set_library(resolve_library(tmp_path), str(tmp_path))
    click_reading()
    assert len(opened) == 2 and not opened[1].isLocalFile()
    assert opened[1].toString() == entry["sources"][0]["url"]
    assert controller.imported is None and controller.job is None
