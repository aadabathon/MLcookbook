"""Research navigation preserves chronology and emits only explicit reading actions."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, Qt
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
        {"id": "logic", "year": -350, "date": "c. 350 BCE", "era": "Foundations",
         "title": "Formal logic", "people": "Aristotle", "summary": "Reasoning from stated premises.",
         "significance": "A historical precursor to symbolic reasoning, rather than computational AI.",
         "connections": ["Explicit assumptions in statistical analysis."],
         "sources": [{"title": "Prior Analytics", "url": "https://example.org/logic"}], "paper_ids": []},
        {"id": "dartmouth", "year": 1956, "date": "1956", "era": "Early AI",
         "title": "Dartmouth summer research project", "people": "John McCarthy and colleagues",
         "summary": "A proposal organizes work around artificial intelligence.",
         "significance": "The proposal dates from 1955; the workshop took place in 1956.",
         "connections": [], "sources": [{"title": "Dartmouth proposal", "url": "https://example.org/dartmouth"}],
         "paper_ids": []},
        {"id": "transformer", "year": 2017, "date": "2017", "era": "Language models",
         "title": "Attention Is All You Need", "people": "Ashish Vaswani and colleagues",
         "summary": "Attention replaces recurrence in a sequence transduction model.",
         "significance": "Parallel matrix operations change training and inference workloads.",
         "connections": ["Graph inspection exposes attention shapes and activation storage."],
         "sources": [{"title": "Original Transformer paper", "url": "https://example.org/transformer"}],
         "paper_ids": ["transformer"]},
        {"id": "gqa", "year": 2023, "date": "2023", "era": "Language models",
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
