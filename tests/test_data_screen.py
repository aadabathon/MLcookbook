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


def _workflow_discovery(target=None, features=None):
    return {"columns": [
        {"name": "identifier", "role": "categorical", "dtype": "string", "suggested": False,
         "reason": "Looks like a row identifier; initially excluded."},
        {"name": "mixedCase", "role": "numeric", "dtype": "float64", "suggested": True,
         "reason": "Numeric storage; review its meaning."},
    ], "context": {"target": target, "features": ["mixedCase"] if features is None else features,
                   "group_column": None, "time_column": None},
        "workflows": [
        {"key": "association-scan", "label": "Association scan", "intent": "relationships", "task": "unsupervised",
         "available": False, "reason": "Select at least two columns.", "warnings": []},
        {"key": "linear-regression", "label": "Linear regression · OLS / ridge / lasso", "intent": "regression",
         "task": "regression", "available": target == "mixedCase", "reason": "Ready." if target else "Select a numeric target.", "warnings": []},
        {"key": "pca", "label": "Principal component analysis", "intent": "structure", "task": "unsupervised",
         "available": True, "reason": "Choose settings in the recipe, then run explicitly.", "warnings": []},
    ], "warnings": []}


def test_sqlite_is_an_explicit_action_and_busy_locks_it(screen):
    calls = []
    screen.sqlite_requested.connect(lambda: calls.append(True))
    screen.sqlite_button.click()
    assert calls == [True]
    screen.set_busy(True)
    assert not screen.sqlite_button.isEnabled()
    screen.sqlite_button.click()
    assert calls == [True]


def test_workflows_render_plain_requirements_and_keep_identifiers_explicit(screen, qt):
    from PySide6.QtCore import Qt

    view = _view()
    view["workflows"] = _workflow_discovery()
    original = deepcopy(view)
    screen.set_dataset(view)
    qt.processEvents()
    assert view == original
    assert screen.workflow_panel.isVisible()
    assert screen.workflow_columns.item(0, 0).checkState() == Qt.CheckState.Unchecked
    assert "identifier" in screen.workflow_columns.item(0, 0).toolTip()
    assert not screen.workflow_open.isEnabled()
    assert screen.workflow_reason.text() == "Select at least two columns."
    assert "review its meaning" in screen.columns_table.item(1, 1).toolTip()


def test_workflow_handoff_has_exact_choices_and_waits_for_fresh_core_validation(screen):
    from PySide6.QtCore import Qt

    screen.set_dataset(_view())
    screen.set_workflows(_workflow_discovery())
    edits, opens = [], []
    screen.workflow_context_changed.connect(lambda context: edits.append(context))
    screen.workflow_requested.connect(lambda key, context: opens.append((key, context)))
    screen.workflow_intent.setCurrentIndex(screen.workflow_intent.findData("regression"))
    screen.workflow_target.setCurrentIndex(screen.workflow_target.findData("mixedCase"))
    screen.workflow_columns.item(0, 0).setCheckState(Qt.CheckState.Checked)
    assert edits[-1] == {"target": "mixedCase", "features": ["identifier"], "group_column": None, "time_column": None}
    assert not screen.workflow_open.isEnabled()
    screen.set_workflows(_workflow_discovery("mixedCase", ["identifier"]))
    screen.workflow_open.click()
    assert opens == [("linear-regression", {**edits[-1], "task": "regression"})]
    screen.workflow_columns.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    screen._workflow_method_changed()
    assert not screen.workflow_open.isEnabled()  # A method change cannot revive stale applicability.
    screen._open_workflow()
    assert len(opens) == 1


def test_workflow_state_restores_separately_from_import_options_and_source_roles(screen):
    from PySide6.QtCore import Qt

    screen.set_dataset(_view())
    screen.set_workflows(_workflow_discovery())
    before = screen.import_options()
    source = screen.columns_table.item(1, 1).text()
    state = {"intent": "regression", "target": "mixedCase", "features": ["identifier"],
             "group_column": "identifier", "time_column": None, "method": "linear-regression",
             "inputs_expanded": True, "context_expanded": True}
    changes = []
    screen.workflow_context_changed.connect(changes.append)
    screen.restore_workflow_state(state)
    assert screen.workflow_state() == state
    assert len(changes) == 1
    assert changes[0] == {"target": "mixedCase", "features": [], "group_column": "identifier", "time_column": None}
    assert screen.workflow_columns.item(0, 0).checkState() == Qt.CheckState.Unchecked
    assert "evaluation context" in screen.workflow_columns.item(0, 0).toolTip()
    assert screen.import_options() == before
    assert screen.columns_table.item(1, 1).text() == source


def test_unsupervised_intent_does_not_hide_an_excluded_target(screen):
    screen.set_dataset(_view())
    screen.set_workflows(_workflow_discovery())
    screen.workflow_intent.setCurrentIndex(screen.workflow_intent.findData("regression"))
    screen.workflow_target.setCurrentIndex(screen.workflow_target.findData("mixedCase"))
    assert screen.workflow_context()["features"] == []
    screen.workflow_intent.setCurrentIndex(screen.workflow_intent.findData("structure"))
    assert screen.workflow_context() == {"target": None, "features": ["mixedCase"], "group_column": None, "time_column": None}


def test_workflow_controls_fit_narrow_content_and_busy_preserves_choices(screen, qt):
    screen.resize(650, 700)
    screen.set_dataset(_view())
    screen.set_workflows(_workflow_discovery())
    screen.workflow_columns_toggle.setChecked(True)
    screen.workflow_context_toggle.setChecked(True)
    qt.processEvents()
    before = screen.workflow_state()
    content = screen.scroll.widget()
    assert content.width() <= screen.scroll.viewport().width()
    assert screen.workflow_panel.width() <= screen.scroll.viewport().width()
    screen.set_busy(True)
    assert not screen.workflow_panel.isEnabled()
    assert screen.workflow_state() == before
    screen.set_busy(False)
    assert screen.workflow_panel.isEnabled()
