"""Research navigation preserves chronology and emits only explicit reading actions."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import theme  # noqa: E402
from gui.screens.research import ResearchScreen  # noqa: E402


@pytest.fixture(scope="module")
def qt():
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(theme.stylesheet())
    return application


@pytest.fixture
def catalog():
    entries = [
        {"id": "logic", "year": -350, "date": "c. 350 BCE", "era": "Foundations", "track": "Computing",
         "title": "Formal logic", "people": "Aristotle", "summary": "Reasoning from stated premises.",
         "significance": "A historical precursor to symbolic reasoning, rather than computational AI.",
         "connections": ["Explicit assumptions in statistical analysis."],
         "sources": [{"title": "Prior Analytics", "url": "https://example.org/logic"}], "paper_ids": []},
        {"id": "dartmouth", "year": 1956, "date": "1956", "era": "Early AI", "track": "AI",
         "title": "Dartmouth summer research project", "people": "John McCarthy and colleagues",
         "summary": "A proposal organizes work around artificial intelligence.",
         "significance": "The proposal dates from 1955; the workshop took place in 1956.",
         "connections": [], "sources": [{"title": "Dartmouth proposal", "url": "https://example.org/dartmouth"}],
         "paper_ids": []},
        {"id": "transformer", "year": 2017, "date": "2017", "era": "Language models", "track": "Deep learning",
         "title": "Attention Is All You Need", "people": "Ashish Vaswani and colleagues",
         "summary": "Attention replaces recurrence in a sequence transduction model.",
         "significance": "Parallel matrix operations change training and inference workloads.",
         "connections": ["Graph inspection exposes attention shapes and activation storage."],
         "sources": [{"title": "Original Transformer paper", "url": "https://example.org/transformer"}],
         "paper_ids": ["transformer"]},
        {"id": "gqa", "year": 2023, "date": "2023", "era": "Language models", "track": "Deep learning",
         "title": "Grouped-query attention", "people": "Joshua Ainslie and colleagues",
         "summary": "Share key/value heads across groups of query heads.",
         "significance": "A memory and quality tradeoff for decoder inference.",
         "connections": ["KV-cache bandwidth matters for accelerator design."],
         "sources": [{"title": "GQA paper", "url": "https://example.org/gqa"}], "paper_ids": ["gqa"]},
    ]
    papers = [
        {"id": "transformer", "title": "Attention Is All You Need", "year": 2017, "kind": "Paper",
         "filename": "attention.pdf", "path": "C:/reading/attention.pdf", "available": True,
         "url": "https://example.org/transformer", "summary": "Original Transformer architecture."},
        {"id": "gqa", "title": "GQA", "year": 2023, "kind": "Paper", "filename": "gqa.pdf",
         "path": "C:/reading/gqa.pdf", "available": False, "url": "https://example.org/gqa",
         "summary": "Grouped-query attention."},
    ]
    return entries, papers


@pytest.fixture
def screen(qt, catalog):
    widget = ResearchScreen(*catalog)
    widget.views.setCurrentIndex(1)
    widget.resize(880, 650)
    widget.show()
    qt.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def selected(widget):
    item = widget.timeline.currentItem()
    return item.data(Qt.ItemDataRole.UserRole) if item is not None else None


def choose(widget, key):
    for row in range(widget.timeline.count()):
        if widget.timeline.item(row).data(Qt.ItemDataRole.UserRole) == key:
            widget.timeline.setCurrentRow(row)
            return
    raise AssertionError(key)


def test_chronology_keyboard_and_plain_data_are_retained(screen, catalog, qt):
    entries, _ = catalog
    assert [screen.timeline.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(screen.timeline.count())] == [entry["id"] for entry in entries]
    assert selected(screen) == "logic"
    screen.timeline.setFocus()
    QTest.keyClick(screen.timeline, Qt.Key.Key_Down)
    assert selected(screen) == "dartmouth"
    entries[0]["title"] = "Edited outside the screen"
    assert screen.entries[0]["title"] == "Formal logic"


def test_search_era_and_linked_readings_are_combined(screen):
    screen.search.setText("KV-cache accelerator")
    assert screen.timeline.count() == 1 and selected(screen) == "gqa"
    screen.era.setCurrentIndex(screen.era.findData("Early AI"))
    assert screen.timeline.count() == 0 and screen.empty.isVisible()
    screen.era.setCurrentIndex(0)
    screen.search.clear()
    screen.papers_only.setChecked(True)
    assert [screen.timeline.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(screen.timeline.count())] == ["transformer", "gqa"]
    assert not screen.empty.isVisible()


def test_sources_and_local_pdfs_require_explicit_actions(screen):
    urls, files, directories = [], [], []
    screen.source_requested.connect(urls.append)
    screen.paper_requested.connect(files.append)
    screen.library_requested.connect(lambda: directories.append(True))
    choose(screen, "transformer")
    assert urls == files == directories == []
    screen.source_buttons[0].click()
    screen.paper_buttons[0].click()
    screen.library_button.click()
    assert urls == ["https://example.org/transformer"]
    assert files == ["transformer"] and directories == [True]
    choose(screen, "gqa")
    assert not screen.paper_buttons[0].isEnabled()
    screen.paper_buttons[0].click()
    assert files == ["transformer"]
    screen.source_buttons[0].click()
    assert urls[-1] == "https://example.org/gqa"


def test_filtering_temporarily_hides_but_retains_selected_milestone(screen):
    choose(screen, "transformer")
    screen.search.setText("Aristotle")
    assert selected(screen) == "logic"
    assert screen.session_state()["selected"] == "transformer"
    screen.search.clear()
    assert selected(screen) == "transformer"


def test_clicking_filtered_fallback_remembers_explicit_selection(screen, qt):
    choose(screen, "transformer")
    screen.search.setText("Aristotle")
    qt.processEvents()
    assert selected(screen) == "logic"
    item_rect = screen.timeline.visualItemRect(screen.timeline.item(0))
    QTest.mouseClick(screen.timeline.viewport(), Qt.MouseButton.LeftButton, pos=item_rect.center())
    screen.search.clear()
    assert selected(screen) == "logic" and screen.session_state()["selected"] == "logic"


def test_navigation_state_restores_search_selection_and_scroll(screen, catalog, qt):
    choose(screen, "gqa")
    screen.era.setCurrentIndex(screen.era.findData("Language models"))
    screen.papers_only.setChecked(True)
    screen.search.setText("attention")
    state = deepcopy(screen.session_state())
    restored = ResearchScreen(*catalog)
    try:
        restored.resize(880, 650)
        restored.restore_session(state)
        restored.show()
        qt.processEvents()
        assert restored.session_state() == state
        assert selected(restored) == "gqa"
    finally:
        restored.close()
        restored.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_long_timeline_scroll_restores_after_layout(catalog, qt):
    entries, papers = deepcopy(catalog)
    entries = [{**entries[0], "id": f"logic-{index}", "date": str(index)} for index in range(20)]
    first = ResearchScreen(entries, papers)
    restored = ResearchScreen(entries, papers)
    try:
        first.views.setCurrentIndex(1)
        restored.views.setCurrentIndex(1)
        first.resize(880, 650)
        first.show()
        qt.processEvents()
        first.timeline.setCurrentRow(6)
        first.timeline.verticalScrollBar().setValue(180)
        state = first.session_state()
        assert state["scroll"] == 180
        restored.resize(880, 650)
        restored.restore_session(state)
        restored.show()
        qt.processEvents()
        assert restored.session_state() == state
    finally:
        for widget in (first, restored):
            widget.close()
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_library_refresh_and_busy_state_preserve_filters_and_actions(screen, catalog):
    choose(screen, "gqa")
    screen.search.setText("grouped")
    papers = deepcopy(catalog[1])
    papers[1]["available"] = True
    screen.set_library(papers, "C:/new-readings")
    assert selected(screen) == "gqa" and screen.search.text() == "grouped"
    assert screen.paper_buttons[0].isEnabled()
    assert screen.library_summary.text().startswith("2 of 2")
    assert screen.library_summary.toolTip() == "C:/new-readings"
    screen.set_busy(True)
    assert not screen.search.isEnabled() and not screen.library_button.isEnabled()
    assert not screen.source_buttons[0].isEnabled() and not screen.paper_buttons[0].isEnabled()
    screen.set_busy(False)
    assert screen.search.isEnabled() and screen.paper_buttons[0].isEnabled()
    screen.set_status("A reading moved. Choose its new folder.")
    assert screen.status.isVisible() and screen._status_message == screen.status.text()


@pytest.mark.parametrize("width", [850, 1100])
def test_timeline_and_long_details_remain_within_their_viewports(screen, qt, width):
    choose(screen, "transformer")
    screen.resize(width, 650)
    qt.processEvents()
    assert screen.timeline.horizontalScrollBar().maximum() == 0
    assert screen.detail_scroll.horizontalScrollBar().maximum() == 0
    assert screen.detail.width() <= screen.detail_scroll.viewport().width()
    assert screen.timeline.viewport().width() > 230
    for child in screen.detail.findChildren(QLabel):
        if child.isVisible():
            assert child.width() <= screen.detail_scroll.viewport().width()


def click_graph_card(screen, qt, key, *, action=False):
    """Exercise the actual scene hit target through the native viewport."""
    screen.views.setCurrentIndex(0)
    qt.processEvents()
    graph = screen.graphic
    node = graph.node_items[key]
    graph.centerOn(node)
    qt.processEvents()
    local = node.action_rect.center() if action else node.boundingRect().topLeft() + node.boundingRect().center() / 2
    position = graph.mapFromScene(node.mapToScene(local))
    assert graph.viewport().rect().contains(position)
    QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, pos=position)
    qt.processEvents()


def test_graph_cards_select_milestones_and_stay_in_sync_with_reading_list(screen, qt):
    urls, files = [], []
    screen.source_requested.connect(urls.append)
    screen.paper_requested.connect(files.append)
    click_graph_card(screen, qt, "transformer")
    assert selected(screen) == "transformer"
    assert screen.graphic.selected_id == "transformer"
    assert screen.session_state()["selected"] == "transformer"
    assert "Attention Is All You Need" in [label.text() for label in screen.detail.findChildren(QLabel)]
    assert urls == files == []
    choose(screen, "gqa")
    assert screen.graphic.selected_id == "gqa"


def test_graph_paper_actions_open_local_readings_or_catalog_sources_only(screen, qt):
    urls, files = [], []
    screen.source_requested.connect(urls.append)
    screen.paper_requested.connect(files.append)
    click_graph_card(screen, qt, "transformer", action=True)
    assert files == ["transformer"] and urls == []
    click_graph_card(screen, qt, "gqa", action=True)
    assert urls == ["https://example.org/gqa"] and files == ["transformer"]
    click_graph_card(screen, qt, "logic", action=True)
    assert urls[-1] == "https://example.org/logic"
    screen.set_busy(True)
    click_graph_card(screen, qt, "logic", action=True)
    assert urls == ["https://example.org/gqa", "https://example.org/logic"]
    assert files == ["transformer"]


def test_graph_keyboard_navigation_and_open_action_respect_busy_state(screen, qt):
    screen.views.setCurrentIndex(0)
    urls, files = [], []
    screen.source_requested.connect(urls.append)
    screen.paper_requested.connect(files.append)
    graph = screen.graphic
    graph.setFocus()
    QTest.keyClick(graph, Qt.Key.Key_End)
    assert selected(screen) == graph.selected_id == "gqa"
    QTest.keyClick(graph, Qt.Key.Key_Up)
    assert selected(screen) == graph.selected_id == "transformer"
    QTest.keyClick(graph, Qt.Key.Key_Return)
    assert files == ["transformer"] and urls == []
    QTest.keyClick(graph, Qt.Key.Key_Home)
    assert selected(screen) == graph.selected_id == "logic"
    screen.set_busy(True)
    QTest.keyClick(graph, Qt.Key.Key_Return)
    QTest.keyClick(graph, Qt.Key.Key_End)
    assert files == ["transformer"] and urls == []
    assert selected(screen) == graph.selected_id == "logic"


def test_graph_filtering_and_empty_state_share_the_visible_chronology(screen, qt):
    screen.views.setCurrentIndex(0)
    choose(screen, "transformer")
    screen.search.setText("Aristotle")
    assert set(screen.graphic.node_items) == {"logic"}
    assert screen.graphic.selected_id == selected(screen) == "logic"
    assert screen.session_state()["selected"] == "transformer"
    screen.search.setText("a query without any catalog match")
    assert not screen.graphic.node_items
    assert selected(screen) is None and screen.empty.isVisible()
    screen.search.clear()
    assert selected(screen) == screen.graphic.selected_id == "transformer"
    screen.papers_only.setChecked(True)
    assert set(screen.graphic.node_items) == {"transformer", "gqa"}


def test_filtering_keeps_only_the_selected_tab_page_visible(screen, qt):
    screen.views.setCurrentIndex(0)
    screen.search.setText("attention")
    qt.processEvents()
    assert screen.graphic.isVisible() and not screen.timeline.isVisible()
    screen.views.setCurrentIndex(1)
    screen.search.clear()
    qt.processEvents()
    assert screen.timeline.isVisible() and not screen.graphic.isVisible()
    screen.views.setCurrentIndex(0)
    screen.search.setText("a query without any catalog match")
    screen.search.clear()
    qt.processEvents()
    assert screen.graphic.isVisible() and not screen.timeline.isVisible()


def test_graph_view_and_selection_restore_without_opening_any_reading(screen, catalog, qt):
    click_graph_card(screen, qt, "transformer")
    screen.graphic.scale(1.15, 1.15)
    state = deepcopy(screen.session_state())
    assert state["view"] == "timeline"
    restored = ResearchScreen(*catalog)
    urls, files = [], []
    restored.source_requested.connect(urls.append)
    restored.paper_requested.connect(files.append)
    try:
        restored.resize(880, 650)
        restored.restore_session(state)
        restored.show()
        qt.processEvents()
        assert restored.views.currentIndex() == 0
        assert selected(restored) == restored.graphic.selected_id == "transformer"
        assert restored.graphic.transform().m11() == pytest.approx(screen.graphic.transform().m11())
        assert restored.graphic.view_state() == state["graphic"]
        assert urls == files == []
    finally:
        restored.close()
        restored.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_track_filter_combines_with_search_and_restores_with_navigation(screen, catalog, qt):
    screen.track.setCurrentIndex(screen.track.findData("Deep learning"))
    assert set(screen.graphic.node_items) == {"transformer", "gqa"}
    screen.search.setText("grouped")
    assert set(screen.graphic.node_items) == {"gqa"}
    state = screen.session_state()
    assert state["track"] == "Deep learning"
    restored = ResearchScreen(*catalog)
    try:
        restored.restore_session(state)
        assert restored.track.currentData() == "Deep learning"
        assert set(restored.graphic.node_items) == {"gqa"}
        restored.search.clear()
        restored.era.setCurrentIndex(restored.era.findData("Early AI"))
        assert not restored.graphic.node_items
    finally:
        restored.close()
        restored.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_dense_same_year_cards_keep_reading_targets_separate(qt, catalog):
    from gui.research_timeline import ResearchTimeline

    entries, papers = deepcopy(catalog)
    dense = [{**entries[2], "id": f"paper-{index}", "year": 2020, "date": "2020"}
             for index in range(24)]
    graph = ResearchTimeline(dense)
    try:
        nodes = list(graph.node_items.values())
        assert len(nodes) == 24
        for index, node in enumerate(nodes):
            assert node.boundingRect().contains(node.action_rect)
            for following in nodes[index + 1:]:
                assert not node.sceneBoundingRect().intersects(following.sceneBoundingRect())
        assert graph.sceneRect().contains(nodes[-1].sceneBoundingRect())
    finally:
        graph.close()
        graph.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_graph_context_readings_never_replace_the_original_source_action(qt, catalog):
    from gui.research_timeline import ResearchTimeline

    entries, papers = deepcopy(catalog)
    # A later reading can explain this milestone without being its original paper.
    entries[0]["paper_ids"] = ["transformer"]
    graph = ResearchTimeline(entries)
    graph.set_library(papers)
    urls, files = [], []
    graph.source_requested.connect(urls.append)
    graph.paper_requested.connect(files.append)
    try:
        graph.open_entry("logic")
        assert urls == ["https://example.org/logic"] and files == []
    finally:
        graph.close()
        graph.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_zoom_controls_and_library_refresh_preserve_the_current_view(screen, catalog, qt):
    click_graph_card(screen, qt, "gqa")
    initial = screen.graphic.transform().m11()
    screen.zoom_in.click()
    qt.processEvents()
    assert screen.graphic.transform().m11() > initial
    before = screen.graphic.view_state()
    papers = deepcopy(catalog[1])
    papers[1]["available"] = True
    screen.set_library(papers, "C:/new-library")
    qt.processEvents()
    after = screen.graphic.view_state()
    assert after["zoom"] == pytest.approx(before["zoom"])
    assert after["x"] == pytest.approx(before["x"], abs=2)
    assert after["y"] == pytest.approx(before["y"], abs=2)
    assert selected(screen) == "gqa"
    screen.set_busy(True)
    screen.zoom_out.click()
    screen.fit_button.click()
    assert screen.graphic.transform().m11() == pytest.approx(after["zoom"])


def test_overview_navigates_without_opening_papers_and_locks_while_busy(screen, qt):
    screen.views.setCurrentIndex(0)
    screen.graphic.select_entry("logic")
    qt.processEvents()
    urls, files = [], []
    screen.source_requested.connect(urls.append)
    screen.paper_requested.connect(files.append)
    before = screen.graphic.horizontalScrollBar().value()
    QTest.mouseClick(screen.overview, Qt.MouseButton.LeftButton,
                     pos=QPoint(screen.overview.width() - 14, screen.overview.height() // 2))
    qt.processEvents()
    assert screen.graphic.horizontalScrollBar().value() > before
    assert urls == files == []
    screen.set_busy(True)
    locked = screen.graphic.view_state()
    QTest.mouseClick(screen.overview, Qt.MouseButton.LeftButton, pos=QPoint(14, 30))
    assert screen.graphic.view_state() == locked
    assert not screen.zoom_in.isEnabled() and not screen.zoom_out.isEnabled()
    assert not screen.fit_button.isEnabled()
