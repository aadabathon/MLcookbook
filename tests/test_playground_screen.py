"""The data playground edits plain queries and renders completed snapshots."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def qt():
    from PySide6.QtWidgets import QApplication

    from gui import theme

    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(theme.stylesheet())
    return application


@pytest.fixture
def specs():
    return [
        {"key": "usgs", "title": "USGS earthquakes", "description": "Recorded seismic events.",
         "url": "https://earthquake.usgs.gov/fdsnws/event/1/",
         "fields": [
             {"key": "start_date", "label": "Start date", "kind": "date", "default": "2024-01-01"},
             {"key": "end_date", "label": "End date", "kind": "date", "default": "2024-02-01"},
             {"key": "region", "label": "Region", "kind": "choice", "default": "world",
              "options": [("Worldwide", "world"), ("California", "california")]},
             {"key": "min_magnitude", "label": "Minimum magnitude", "kind": "float", "default": 4.0,
              "min": 0, "max": 10, "decimals": 1}],
         "columns": [{"key": "id", "label": "Event identifier", "unit": "", "role": "unsupported"},
                     {"key": "magnitude", "label": "Magnitude", "unit": "", "role": "numeric"},
                     {"key": "depth", "label": "Depth", "unit": "km", "role": "numeric"}],
         "default_columns": ["id", "magnitude", "depth"], "max_limit": 5000, "default_limit": 500},
        {"key": "food", "title": "Open Food Facts", "description": "Packaged foods and nutrients.",
         "url": "https://openfoodfacts.github.io/documentation/",
         "fields": [{"key": "category", "label": "Category", "kind": "text", "default": "breakfast-cereals"},
                    {"key": "complete_only", "label": "Complete measurements", "kind": "bool", "default": False}],
         "columns": [{"key": "product", "label": "Product", "unit": "", "role": "categorical"},
                     {"key": "sugar", "label": "Sugar", "unit": "g / 100 g", "role": "numeric"}],
         "default_columns": ["product", "sugar"], "max_limit": 500, "default_limit": 100},
        {"key": "census", "title": "Census ACS", "description": "Geographic estimates.",
         "url": "https://www.census.gov/data/developers/",
         "fields": [{"key": "year", "label": "Year", "kind": "int", "min": 2010, "max": 2024, "default": 2024},
                    {"key": "api_key", "label": "API key", "kind": "text", "secret": True, "required": True,
                     "default": "", "help": "Register a Census API key."}],
         "columns": [{"key": "income", "label": "Median income", "unit": "USD", "role": "numeric"}],
         "default_columns": ["income"]},
    ]


@pytest.fixture
def screen(qt, specs):
    from PySide6.QtCore import QCoreApplication, QEvent

    from gui.screens.playground import DataPlaygroundScreen

    widget = DataPlaygroundScreen(specs)
    widget.resize(850, 760)
    widget.show()
    qt.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _select(screen, key):
    screen.source_combo.setCurrentIndex(screen.source_combo.findData(key))


def _view(screen):
    source, query = screen._safe_request()
    return {"name": "Recorded <events>", "rows": 40,
            "columns": [{"name": "id", "type": "unsupported", "unit": "", "missing": 0},
                        {"name": "magnitude", "type": "numeric", "unit": "", "missing": 0},
                        {"name": "depth", "type": "numeric", "unit": "km", "missing": 3}],
            "preview": [[f"{index:05d}", "4.12345", "1.23456789e-09"] for index in range(40)],
            "source_summary": "USGS · fetched 2026-10-28T12:00:00Z",
            "provenance": ["Query: 2024-01-01 to 2024-02-01", "Depth measured in km."],
            "warnings": ["Nearby events may be dependent."],
            "notes": ["Three missing depth values."], "issues": [],
            "playground_request": {"source": source, "query": query}}


def test_only_explicit_fetch_and_use_emit_requests(screen):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractItemView

    fetched, accepted = [], []
    screen.fetch_requested.connect(lambda *args: fetched.append(args))
    screen.use_requested.connect(lambda: accepted.append(True))
    assert screen.view is None and not screen.use_button.isEnabled()
    assert screen.result_state.isVisible()
    screen.fetch_button.click()
    assert fetched == [("usgs", {"start_date": "2024-01-01", "end_date": "2024-02-01", "region": "world",
                                  "min_magnitude": 4.0, "limit": 500, "columns": ["id", "magnitude", "depth"]})]
    supplied = _view(screen)
    original = deepcopy(supplied)
    screen.set_preview(supplied)
    assert supplied == original
    assert screen.use_button.isEnabled() and not screen.result_state.isVisible()
    assert not accepted
    screen.use_button.click()
    assert accepted == [True]
    assert screen.preview_table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert screen.preview_table.rowCount() == 20
    assert screen.preview_table.item(0, 0).text() == "00000"
    assert "km" in screen.preview_table.horizontalHeaderItem(2).toolTip()
    assert screen.identity.textFormat() == Qt.TextFormat.PlainText
    assert screen.identity.text().startswith("Recorded <events>")
    assert len(fetched) == 1


def test_download_emits_selected_format_only_for_completed_matching_preview(screen):
    downloaded, fetched, accepted = [], [], []
    screen.download_requested.connect(downloaded.append)
    screen.fetch_requested.connect(lambda *args: fetched.append(args))
    screen.use_requested.connect(lambda: accepted.append(True))
    assert screen.download_format.currentData() == "csv"
    assert not screen.download_button.isEnabled()
    screen._download()
    assert downloaded == []
    supplied = _view(screen)
    original = deepcopy(supplied)
    screen.set_preview(supplied)
    assert screen.download_button.isEnabled()
    screen.download_button.click()
    screen.download_format.setCurrentIndex(screen.download_format.findData("parquet"))
    screen.download_button.click()
    assert downloaded == ["csv", "parquet"]
    assert fetched == accepted == []
    assert screen.view == original
    screen.limit.setValue(screen.limit.value() + 1)
    assert not screen.download_button.isEnabled()
    screen._download()
    assert downloaded == ["csv", "parquet"]
    screen.limit.setValue(screen.limit.value() - 1)
    assert screen.download_button.isEnabled()
    screen.set_busy(True)
    assert not screen.download_button.isEnabled() and not screen.download_format.isEnabled()
    screen._download()
    assert downloaded == ["csv", "parquet"]
    screen.set_busy(False)
    assert screen.download_button.isEnabled() and screen.download_format.isEnabled()
    screen.set_preview({**supplied, "rows": 0})
    assert not screen.download_button.isEnabled()
    screen._download()
    screen.set_preview(None)
    assert not screen.download_button.isEnabled()
    assert downloaded == ["csv", "parquet"]


def test_download_destination_and_session_restore_have_no_download_side_effect(screen, qt, specs):
    from PySide6.QtCore import QCoreApplication, QEvent

    from gui.screens.playground import DataPlaygroundScreen

    screen.set_download_directory(r"C:\Projects\MLcookbook\data")
    assert screen.download_destination.text() == "Save to data/"
    assert screen.download_destination.toolTip() == r"C:\Projects\MLcookbook\data"
    assert r"C:\Projects\MLcookbook\data" in screen.download_button.toolTip()
    screen.set_preview(_view(screen))
    screen.download_format.setCurrentIndex(screen.download_format.findData("parquet"))
    state = screen.session_state()
    assert "download_format" not in state
    restored = DataPlaygroundScreen(specs)
    downloaded = []
    restored.download_requested.connect(downloaded.append)
    try:
        restored.restore_session(state)
        restored.set_preview(screen.view)
        qt.processEvents()
        assert restored.download_format.currentData() == "csv"
        assert restored.download_button.isEnabled()
        assert downloaded == []
    finally:
        restored.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_edits_and_source_switch_preserve_preview_but_require_matching_query(screen):
    screen.set_preview(_view(screen))
    before = deepcopy(screen.view)
    screen.field_widgets["min_magnitude"].setValue(5)
    assert screen.preview_stale and not screen.use_button.isEnabled()
    assert screen.result_state.isVisible() and "Settings changed" in screen.result_state.text()
    assert screen.view == before
    screen.field_widgets["min_magnitude"].setValue(4)
    assert not screen.preview_stale and screen.use_button.isEnabled()
    _select(screen, "food")
    assert screen.preview_stale and not screen.use_button.isEnabled()
    assert screen.limit.maximum() == 500
    screen.field_widgets["category"].setText("chocolates")
    screen.limit.setValue(75)
    _select(screen, "usgs")
    assert screen.field_widgets["min_magnitude"].value() == 4
    assert screen.limit.value() == 500 and screen.limit.maximum() == 5000
    assert not screen.preview_stale and screen.use_button.isEnabled()
    _select(screen, "food")
    assert screen.field_widgets["category"].text() == "chocolates" and screen.limit.value() == 75
    assert screen.view == before


def test_invalid_query_does_not_fetch_or_discard_preview(screen):
    from PySide6.QtCore import QDate, Qt

    fetched = []
    screen.fetch_requested.connect(lambda *args: fetched.append(args))
    screen.set_preview(_view(screen))
    original = deepcopy(screen.view)
    screen.field_widgets["start_date"].setDate(QDate(2025, 1, 1))
    screen.fetch_button.click()
    assert not fetched and "start date" in screen.status.text()
    screen.field_widgets["start_date"].setDate(QDate(2024, 1, 1))
    for row in range(screen.columns.rowCount()):
        screen.columns.item(row, 0).setCheckState(Qt.CheckState.Unchecked)
    screen.fetch_button.click()
    assert not fetched and "at least one column" in screen.status.text()
    assert screen.view == original and screen.preview_stale
    assert not screen.use_button.isEnabled()


def test_busy_cancellation_and_status_leave_completed_preview_intact(screen):
    fetched, accepted, cancelled = [], [], []
    screen.fetch_requested.connect(lambda *args: fetched.append(args))
    screen.use_requested.connect(lambda: accepted.append(True))
    screen.cancel_requested.connect(lambda: cancelled.append(True))
    screen.set_preview(_view(screen))
    original = deepcopy(screen.view)
    screen.set_busy(True)
    assert not screen.fetch_button.isEnabled() and not screen.use_button.isEnabled()
    assert not screen.source_combo.isEnabled() and not screen.query_panel.isEnabled()
    assert screen.progress.isVisible() and screen.cancel_button.isVisible()
    screen._fetch()
    screen._use()
    assert not fetched and not accepted
    screen.cancel_button.click()
    assert cancelled == [True] and not screen.cancel_button.isEnabled()
    screen.set_busy(False)
    assert screen.fetch_button.isEnabled() and screen.use_button.isEnabled()
    assert not screen.progress.isVisible() and not screen.cancel_button.isVisible()
    assert screen.view == original
    screen.set_busy(True)
    screen.set_cancellable(False)
    assert not screen.cancel_button.isVisible()
    screen._cancel()
    assert cancelled == [True]
    screen.set_busy(False)
    message = "Fetch failed: " + "example " * 80
    screen.set_status(message)
    assert len(screen.status.text()) <= 180
    assert screen._status_message == screen.status.toolTip() == message


def test_warning_access_and_provenance_do_not_fetch(screen):
    fetched = []
    screen.fetch_requested.connect(lambda *args: fetched.append(args))
    screen.set_preview(_view(screen))
    assert screen.warning_button.isVisible() and "2" in screen.warning_button.text()
    assert not screen.warnings.isVisible() and not screen.provenance.isVisible()
    screen.warning_button.click()
    assert screen.warnings.isVisible() and "Nearby events" in screen.warnings.text()
    screen.provenance_toggle.click()
    assert screen.provenance.isVisible() and "2026-10-28" in screen.provenance.text()
    assert "Depth measured in km" in screen.provenance.text()
    assert not fetched
    screen.set_preview(None)
    assert not screen.preview_panel.isVisible() and not screen.use_button.isEnabled()
    assert screen.preview_table.rowCount() == screen.preview_table.columnCount() == 0


def test_sessions_restore_all_sources_without_fetch_or_credentials(screen, qt, specs):
    from PySide6.QtCore import QCoreApplication, QEvent, Qt
    from PySide6.QtWidgets import QLineEdit

    from gui.screens.playground import DataPlaygroundScreen

    screen.set_preview(_view(screen))
    screen.field_widgets["min_magnitude"].setValue(5.5)
    screen.columns.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    screen._column_toggles["usgs"].click()
    _select(screen, "food")
    screen.field_widgets["category"].setText("chocolates")
    screen.field_widgets["complete_only"].setChecked(True)
    screen.limit.setValue(33)
    _select(screen, "census")
    secret = screen.field_widgets["api_key"]
    assert secret.echoMode() == QLineEdit.EchoMode.Password
    secret.setText("do-not-save-this-key")
    screen.field_widgets["year"].setValue(2023)
    state = screen.session_state()
    assert "do-not-save-this-key" not in json.dumps(state)
    assert "api_key" not in state["queries"]["census"]
    restored = DataPlaygroundScreen(specs)
    restored.resize(850, 760)
    restored.show()
    fetched = []
    restored.fetch_requested.connect(lambda *args: fetched.append(args))
    try:
        restored.restore_session(state)
        qt.processEvents()
        assert restored.source_key == "census"
        assert restored.field_widgets["api_key"].text() == ""
        assert restored.field_widgets["year"].value() == 2023
        restored.fetch_button.click()
        assert not fetched and "API key".lower() in restored.status.text().lower()
        _select(restored, "food")
        assert restored.field_widgets["category"].text() == "chocolates"
        assert restored.field_widgets["complete_only"].isChecked() and restored.limit.value() == 33
        _select(restored, "usgs")
        assert restored.field_widgets["min_magnitude"].value() == 5.5
        assert restored.columns.item(0, 0).checkState() == Qt.CheckState.Unchecked
        assert restored._column_toggles["usgs"].isChecked()
        assert not fetched
    finally:
        restored.close()
        restored.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_offline_census_preview_can_be_used_without_transient_key(screen):
    _select(screen, "census")
    screen.field_widgets["api_key"].setText("key")
    view = {"name": "Census", "rows": 3, "columns": [{"name": "income", "type": "numeric", "unit": "USD"}],
            "preview": [["123"]], "playground_request": {"source": "census", "query": screen._safe_request()[1]}}
    screen.set_preview(view)
    screen.field_widgets["api_key"].clear()
    assert not screen.preview_stale and screen.use_button.isEnabled()


def test_content_and_actions_fit_narrow_workspace(screen, qt):
    screen.set_preview(_view(screen))
    screen._column_toggles["usgs"].click()
    screen.warning_button.click()
    screen.provenance_toggle.click()
    screen.resize(550, 650)
    qt.processEvents()
    screen.grab()
    assert screen.content.width() <= screen.scroll.viewport().width()
    for widget in (screen.query_panel, screen.preview_panel, screen.description, screen.preview_table):
        assert widget.width() <= screen.scroll.viewport().width()
    assert screen.fetch_button.geometry().right() < screen.use_button.geometry().left()
    assert screen.use_button.geometry().right() < screen.width()
    assert screen.source_combo.geometry().right() < screen.docs_button.geometry().left()
    for widget in (screen.download_destination, screen.download_format, screen.download_button):
        assert widget.geometry().right() < screen.width()
    assert screen.download_destination.geometry().right() < screen.download_format.geometry().left()
    assert screen.download_format.geometry().right() < screen.download_button.geometry().left()
    assert screen.download_button.geometry().top() > screen.use_button.geometry().bottom()
