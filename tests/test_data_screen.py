"""The ordinary-data page renders supplied snapshots without changing them."""

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

    return QApplication.instance() or QApplication([])


@pytest.fixture
def screen(qt):
    from PySide6.QtCore import QCoreApplication, QEvent

    from gui.screens.data import DataScreen

    widget = DataScreen()
    widget.resize(850, 700)
    widget.show()
    qt.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _view():
    return {"name": "my <study>.parquet", "rows": 40,
            "source_summary": "Saved experiment: held-out work.mlcb\nOriginal CSV is provenance only.",
            "columns": [{"name": "identifier", "type": "categorical", "missing": 0},
                        {"name": "mixedCase", "type": "numeric", "missing": 3}],
            "preview": [[f"{index:05d}", "1.23456789e-09"] for index in range(40)],
            "notes": ["Leading zeros retained."], "issues": ["Review identifier before using it as a predictor."]}


def test_empty_import_first_and_clear_do_not_leave_previous_data(screen):
    assert screen.empty.isVisible() and screen.load_button.isEnabled()
    assert screen.empty.text() == "No dataset open"
    assert not screen.dataset_panel.isVisible()
    screen.set_dataset(_view())
    assert screen.dataset_panel.isVisible() and not screen.empty.isVisible()
    screen.set_dataset(None)
    assert screen.empty.isVisible() and not screen.source.text()
    assert screen.preview_table.rowCount() == screen.columns_table.rowCount() == 0


def test_data_has_one_identity_with_context_help_on_the_columns(screen, qt):
    from PySide6.QtWidgets import QLabel

    assert not screen.findChildren(QLabel, "PageSubtitle")
    supplied = _view()
    supplied.pop("source_summary")
    screen.set_dataset(supplied)
    qt.processEvents()
    assert screen.identity.isVisible() and not screen.empty.isVisible()
    assert not screen.source.isVisible()
    assert "Adjust roles" in screen.columns_table.toolTip()
    assert screen.columns_table.accessibleDescription() == screen.columns_table.toolTip()
    supplied["source_summary"] = "Snapshot from work.mlcb"
    screen.set_dataset(supplied)
    assert screen.source.isVisible() and screen.source.text() == supplied["source_summary"]


def test_recipe_catalog_keeps_title_and_actions_without_a_generic_subtitle(qt):
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QLabel

    from gui.front_page import FrontPage
    from gui.recipe_card import RecipeCard
    from gui.recipes import RECIPES

    screen = FrontPage()
    try:
        assert screen.findChild(QLabel, "PageTitle").text() == "Recipes"
        assert not screen.findChildren(QLabel, "PageSubtitle")
        assert len(screen.findChildren(RecipeCard)) == len(RECIPES)
    finally:
        screen.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_source_roles_preview_strings_and_checks_stay_read_only(screen, qt):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractItemView

    supplied = _view()
    original = deepcopy(supplied)
    screen.set_dataset(supplied)
    qt.processEvents()
    assert supplied == original
    assert screen.identity.text().startswith("my <study>.parquet")
    assert screen.identity.textFormat() == Qt.TextFormat.PlainText
    assert screen.source.text() == supplied["source_summary"]
    assert screen.source.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert screen.preview_table.rowCount() == 20
    assert screen.preview_table.item(0, 0).text() == "00000"
    assert screen.preview_table.item(19, 1).text() == "1.23456789e-09"
    assert screen.columns_table.item(0, 1).text() == "categorical"
    assert screen.columns_table.item(1, 2).text() == "3"
    for table in (screen.preview_table, screen.columns_table):
        assert table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert not screen.checks.isVisible()
    screen.checks_toggle.click()
    assert screen.checks.isVisible()
    assert all(message in screen.checks.text() for message in supplied["notes"] + supplied["issues"])


def test_import_options_restore_and_file_request_preserve_exact_choices(screen, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    options = {"delimiter": "|", "encoding": "utf-16", "markers": "?, -999", "expanded": True}
    screen.restore_import_options(options)
    assert screen.import_options() == options
    chosen = []
    screen.load_requested.connect(lambda *args: chosen.append(args))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("C:/my data/study.csv", ""))
    screen.load_button.click()
    assert chosen == [("C:/my data/study.csv", "|", "utf-16", "?, -999")]
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    screen.load_button.click()
    assert len(chosen) == 1


def test_busy_blocks_import_and_preserves_view_and_options(screen, qt):
    screen.set_dataset(_view())
    before = deepcopy(screen.view)
    options = screen.import_options()
    screen.set_status("Reading dataset…")
    screen.set_busy(True)
    qt.processEvents()
    assert not screen.load_button.isEnabled() and not screen.import_panel.isEnabled()
    assert screen.progress.isVisible()
    assert screen.status.text() == "Reading dataset…"
    screen.set_busy(False)
    assert screen.load_button.isEnabled() and not screen.progress.isVisible()
    assert screen.view == before and screen.import_options() == options


def test_csv_labels_do_not_overlap_controls_in_a_narrow_workspace(screen, qt):
    from PySide6.QtWidgets import QLabel

    screen.resize(650, 700)
    screen.import_toggle.setChecked(True)
    qt.processEvents()
    screen.grab()
    labels = {label.text(): label for label in screen.import_controls.findChildren(QLabel)}
    assert labels["Delimiter"].geometry().right() < screen.delimiter.geometry().left()
    assert screen.delimiter.geometry().right() < labels["Encoding"].geometry().left()
    assert labels["Encoding"].geometry().right() < screen.encoding.geometry().left()
