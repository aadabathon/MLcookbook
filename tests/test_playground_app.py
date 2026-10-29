"""Fetched previews enter ordinary recipes only through explicit acceptance."""

import json
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402 -- the desktop coordinator deliberately lives outside src/.
from gui.recipes import RECIPES  # noqa: E402


@pytest.fixture(scope="module")
def qt():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(qt):
    controller = app.MLCookbookApp()
    controller.window.resize(1080, 760)
    controller.window.show()
    qt.processEvents()
    yield controller
    controller.wait_for_job()
    qt.processEvents()
    controller.window.close()
    controller.window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def finish(controller, qt):
    deadline = time.monotonic() + 30
    while controller.job is not None and time.monotonic() < deadline:
        qt.processEvents()
        time.sleep(.005)
    qt.processEvents()
    assert controller.job is None


def open_playground(controller, qt):
    controller.navigate("playground")
    finish(controller, qt)
    assert controller.key == "playground"
    return controller.screen


def open_recipe(controller, key, qt):
    controller.open_recipe(next(recipe for recipe in RECIPES if recipe.key == key))
    finish(controller, qt)
    return controller.screen


def gathered(source, query):
    from mlcookbook.core.dataset import Dataset

    frame = pd.DataFrame({"magnitude": [2 + index / 20 for index in range(60)],
                          "depth": [5 + index * .7 + index % 3 for index in range(60)]})
    metadata = {
        "source": source, "title": "USGS earthquake snapshot", "source_url": "https://earthquake.usgs.gov/",
        "query": {key: deepcopy(value) for key, value in query.items() if key != "api_key"},
        "retrieved_at": "2026-10-28T15:30:00+00:00", "n_rows": len(frame), "n_columns": len(frame.columns),
        "units": {"magnitude": "magnitude", "depth": "km"},
        "roles": {"magnitude": "numeric", "depth": "numeric"},
        "request_urls": ["https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"],
        "warnings": ["Catalogue detection thresholds differ between regions."], "truncated": False,
    }
    return SimpleNamespace(dataset=Dataset(frame), result=SimpleNamespace(to_dict=lambda: deepcopy(metadata)))


def install_fake(monkeypatch):
    import mlcookbook.gathering as gathering

    calls = []

    def fetch(source, query, *, cancelled=None, progress=None):
        calls.append((source, deepcopy(query), threading.get_ident()))
        if progress:
            progress("Fetched fixture rows")
        return gathered(source, query)

    monkeypatch.setattr(gathering, "fetch_dataset", fetch)
    return calls


def fetch_preview(controller, qt):
    controller.screen.fetch_button.click()
    finish(controller, qt)
    assert controller.playground_preview is not None
    return controller.playground_preview


def seed_ordinary(controller, qt, tmp_path):
    path = tmp_path / "ordinary.csv"
    pd.DataFrame({"x": range(40), "y": [index * 2 + index % 3 for index in range(40)]}).to_csv(path, index=False)
    controller.load_file(str(path), ",", "utf-8", "")
    finish(controller, qt)
    return controller.imported


def test_playground_navigation_never_fetches_and_retains_unrun_controls(controller, qt, monkeypatch):
    calls = install_fake(monkeypatch)
    assert controller.key == "data" and controller.playground_preview is None
    screen = open_playground(controller, qt)
    assert calls == []
    assert not controller.window.save_experiment_button.isEnabled()
    assert controller.window.navigation_buttons["playground"].isChecked()
    assert not screen.use_button.isEnabled()
    screen.limit.setValue(73)
    original = screen.request()
    controller.navigate("data")
    assert controller.imported is None
    screen = open_playground(controller, qt)
    assert screen.request() == original
    assert screen.limit.value() == 73 and calls == []
    assert controller.playground_preview is None


def test_preview_does_not_replace_ordinary_sessions_and_acceptance_preserves_kitchen(controller, qt, monkeypatch, tmp_path):
    ordinary = seed_ordinary(controller, qt, tmp_path)
    recipe = open_recipe(controller, "linear-regression", qt)
    recipe.param_widgets["alpha"].setValue(2.5)
    controller.navigate("kitchen")
    finish(controller, qt)
    # Capture an independent Kitchen snapshot using the same coordinator action
    # as the native capture control; later ordinary replacement must not alter it.
    from mlcookbook.core.dataset import Dataset
    from mlcookbook.kitchen import KitchenDocument

    kitchen_data = Dataset(ordinary[0].df)
    controller.kitchen = KitchenDocument(snapshots={"kept": kitchen_data},
                                         snapshot_info={"kept": {"name": "Earlier snapshot", "source": {}}})
    calls = install_fake(monkeypatch)
    screen = open_playground(controller, qt)
    sessions = dict(controller.sessions)
    preview = fetch_preview(controller, qt)
    assert len(calls) == 1 and calls[0][2] != threading.get_ident()
    assert controller.imported is ordinary and controller.sessions == sessions
    assert controller.kitchen.snapshots["kept"] is kitchen_data
    assert screen.use_button.isEnabled()
    assert screen.view["rows"] == 60
    assert "Catalogue detection" in " ".join(screen.view["notes"])
    screen.use_button.click()
    finish(controller, qt)
    assert controller.imported is not ordinary
    assert controller.imported[0] is not preview[0]
    pd.testing.assert_frame_equal(controller.imported[0].df, preview[0].df)
    assert controller.sessions == controller.workspace_sessions == controller.example_sessions == {}
    assert controller.kitchen.snapshots["kept"] is kitchen_data
    assert controller.playground_preview is preview
    controller.navigate("data")
    assert controller.screen.view["rows"] == 60
    assert "2026-10-28" in controller.screen.source.text()
    assert "USGS" in controller.window.dataset_name_label.toolTip()
    recipe = open_recipe(controller, "linear-regression", qt)
    assert recipe.dataset is controller.imported[0] and not recipe._report


def test_changed_query_cannot_accept_previous_preview_and_reverting_restores_it(controller, qt, monkeypatch):
    install_fake(monkeypatch)
    screen = open_playground(controller, qt)
    preview = fetch_preview(controller, qt)
    original_limit = screen.limit.value()
    screen.limit.setValue(original_limit + 1)
    assert screen.preview_stale and not screen.use_button.isEnabled()
    controller.accept_playground_dataset()
    finish(controller, qt)
    assert controller.imported is None and controller.playground_preview is preview
    controller.navigate("data")
    screen = open_playground(controller, qt)
    assert screen.preview_stale and not screen.use_button.isEnabled()
    screen.limit.setValue(original_limit)
    assert not screen.preview_stale and screen.use_button.isEnabled()
    screen.use_button.click()
    finish(controller, qt)
    assert controller.imported is not None


def test_valid_query_edit_clears_old_error_across_navigation(controller, qt, monkeypatch):
    install_fake(monkeypatch)
    screen = open_playground(controller, qt)
    fetch_preview(controller, qt)
    controller._set_status("Source was unavailable")
    screen.limit.setValue(screen.limit.value() + 1)
    assert screen.status.text() == ""
    controller.navigate("data")
    screen = open_playground(controller, qt)
    assert screen.status.text() == "" and screen.preview_stale


def test_navigation_restores_playground_scroll_after_layout(controller, qt, monkeypatch):
    install_fake(monkeypatch)
    screen = open_playground(controller, qt)
    fetch_preview(controller, qt)
    screen._column_toggles["usgs"].click()
    qt.processEvents()
    screen.scroll.verticalScrollBar().setValue(100)
    position = screen.scroll.verticalScrollBar().value()
    assert position > 0
    controller.navigate("data")
    screen = open_playground(controller, qt)
    qt.processEvents()
    assert screen.scroll.verticalScrollBar().value() == position


def test_failed_fetch_keeps_completed_preview_and_ordinary_recipe_edits(controller, qt, monkeypatch, tmp_path):
    ordinary = seed_ordinary(controller, qt, tmp_path)
    recipe = open_recipe(controller, "linear-regression", qt)
    recipe.param_widgets["alpha"].setValue(4)
    install_fake(monkeypatch)
    screen = open_playground(controller, qt)
    preview = fetch_preview(controller, qt)
    sessions = dict(controller.sessions)
    import mlcookbook.gathering as gathering

    def fail(*args, **kwargs):
        raise ValueError("Source temporarily unavailable")

    monkeypatch.setattr(gathering, "fetch_dataset", fail)
    screen.limit.setValue(screen.limit.value() + 1)
    screen.fetch_button.click()
    finish(controller, qt)
    assert controller.imported is ordinary and controller.sessions == sessions
    assert controller.playground_preview is preview
    assert screen.view["preview"] == preview[2]["preview"]
    assert screen.preview_stale and not screen.use_button.isEnabled()
    assert "Source temporarily unavailable" in screen.status.toolTip()
    assert screen.fetch_button.isEnabled()
    assert all(button.isEnabled() for button in controller.window.navigation_buttons.values())
    recipe = open_recipe(controller, "linear-regression", qt)
    assert recipe.param_widgets["alpha"].value() == 4


def test_cancellable_fetch_locks_routes_and_keeps_last_complete_preview(controller, qt, monkeypatch):
    install_fake(monkeypatch)
    screen = open_playground(controller, qt)
    preview = fetch_preview(controller, qt)
    import mlcookbook.gathering as gathering

    started, release = threading.Event(), threading.Event()

    def blocked(source, query, *, cancelled=None, progress=None):
        started.set()
        deadline = time.monotonic() + 10
        while not release.is_set() and time.monotonic() < deadline:
            if cancelled():
                raise ValueError("Fetch cancelled; previous preview retained")
            time.sleep(.005)
        raise ValueError("Test released fetch")

    monkeypatch.setattr(gathering, "fetch_dataset", blocked)
    screen.fetch_button.click()
    try:
        assert started.wait(5)
        qt.processEvents()
        assert controller.job is not None
        assert not screen.fetch_button.isEnabled() and not screen.use_button.isEnabled()
        assert screen.cancel_button.isEnabled()
        assert all(not button.isEnabled() for button in controller.window.navigation_buttons.values())
        assert not controller.window.open_experiment_button.isEnabled()
        controller.navigate("data")
        assert controller.screen is screen and controller.key == "playground"
        screen.cancel_button.click()
        finish(controller, qt)
        assert controller.playground_preview is preview and controller.imported is None
        assert "cancel" in screen.status.toolTip().lower()
        assert screen.fetch_button.isEnabled()
        assert all(button.isEnabled() for button in controller.window.navigation_buttons.values())
    finally:
        release.set()
        finish(controller, qt)


def test_api_snapshot_reopens_offline_with_query_units_and_warning_provenance(controller, qt, monkeypatch, tmp_path):
    calls = install_fake(monkeypatch)
    screen = open_playground(controller, qt)
    fetch_preview(controller, qt)
    screen.use_button.click()
    finish(controller, qt)
    expected = controller.imported[0].df.copy()
    source = deepcopy(controller.imported[2]["source"])
    recipe = open_recipe(controller, "linear-regression", qt)
    recipe.target.setCurrentIndex(recipe.target.findData("depth"))
    saved = tmp_path / "fetched.mlcb"
    controller.save_experiment(str(saved))
    finish(controller, qt)
    assert saved.is_file()
    controller.navigate("playground")
    finish(controller, qt)
    controller.open_experiment(str(saved))
    finish(controller, qt)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(controller.screen.dataset.df, expected)
    assert controller.screen.source["kind"] == "api"
    assert controller.screen.source["gathering"] == source["gathering"]
    assert controller.screen.source["gathering"]["units"]["depth"] == "km"
    assert controller.screen.source["gathering"]["query"]["limit"] > 0
    assert any("Catalogue detection" in note for note in controller.screen.view["notes"])
    assert not controller.screen._report
    controller.navigate("data")
    assert "2026-10-28" in controller.screen.source.text()
    assert "km" in controller.screen.source.text()


def test_census_secret_is_transient_across_navigation_and_saved_experiment(controller, qt, monkeypatch, tmp_path):
    calls = install_fake(monkeypatch)
    screen = open_playground(controller, qt)
    screen.source_combo.setCurrentIndex(screen.source_combo.findData("census"))
    screen.field_widgets["api_key"].setText("never-persist-this-key")
    fetch_preview(controller, qt)
    assert calls[-1][1]["api_key"] == "never-persist-this-key"
    screen.use_button.click()
    finish(controller, qt)
    controller.navigate("data")
    assert "never-persist-this-key" not in json.dumps(controller.playground_state)
    assert "never-persist-this-key" not in json.dumps(controller.imported[2])
    screen = open_playground(controller, qt)
    assert not screen.field_widgets["api_key"].text()
    assert not screen.preview_stale and screen.use_button.isEnabled()
    recipe = open_recipe(controller, "linear-regression", qt)
    assert recipe.source["kind"] == "api"
    saved = tmp_path / "no-secret.mlcb"
    controller.save_experiment(str(saved))
    finish(controller, qt)
    from mlcookbook.experiments import load_experiment

    loaded = load_experiment(saved)
    assert "never-persist-this-key" not in json.dumps(loaded.source)
