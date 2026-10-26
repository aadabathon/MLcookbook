"""Data-first shell navigation preserves recipe state and source boundaries."""

import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402 -- the coordinator deliberately lives at the repository root.
from gui.recipes import RECIPES  # noqa: E402


@pytest.fixture(scope="module")
def qt():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(qt):
    controller = app.MLCookbookApp()
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


def open_recipe(controller, key, qt):
    controller.open_recipe(next(recipe for recipe in RECIPES if recipe.key == key))
    finish(controller, qt)


def import_frame(controller, qt, frame, tmp_path, name="measurements.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    controller.show_data_page()
    controller.screen.load_requested.emit(str(path), ",", "utf-8-sig", "")
    finish(controller, qt)
    assert controller.imported is not None
    return path


def configure_regression(controller, qt):
    open_recipe(controller, "linear-regression", qt)
    screen = controller.screen
    screen.target.setCurrentIndex(screen.target.findData("y"))
    screen.run_button.click()
    finish(controller, qt)
    assert screen._report
    return screen


def test_data_import_before_recipe_never_constructs_or_runs_a_demo(controller, qt, tmp_path, monkeypatch):
    from gui.screens.data import DataScreen

    assert controller.key == "data" and isinstance(controller.screen, DataScreen)
    assert list(controller.window.navigation_buttons)[0] == "data"
    assert controller.window.navigation_buttons["data"].isChecked()
    assert not controller.window.save_experiment_button.isEnabled()
    path = tmp_path / "own measurements.csv"
    path.write_text("record_id;response;group\n001;1.25;a\n002;?;b\n003;3.5;a\n", encoding="utf-8")
    controller.screen.load_requested.emit(str(path), ";", "utf-8-sig", "?")
    finish(controller, qt)
    data, name, view = controller.imported
    assert name == path.name and data.df.record_id.tolist() == ["001", "002", "003"]
    assert data.df.response.isna().sum() == 1
    assert controller.screen.view["preview"] == view["preview"]
    assert controller.screen.view["source"]["path"] == str(path.resolve())

    def forbidden(*args, **kwargs):
        pytest.fail("Opening an imported-data recipe must not construct or train its example")

    _, demo_name, target, positive = app.DEMOS["linear-regression"]
    monkeypatch.setitem(app.DEMOS, "linear-regression", (forbidden, demo_name, target, positive))
    monkeypatch.setitem(app.REPORT_BUILDERS, "linear-regression", forbidden)
    controller.window.navigation_buttons["recipes"].click()
    assert controller.key == "recipes" and controller.screen is None
    open_recipe(controller, "linear-regression", qt)
    assert controller.screen.dataset is data
    assert not controller.screen.is_example and not controller.screen._report
    assert controller.screen.request()["target"] is None
    assert path.name in controller.screen.input_context.text()
    assert controller.window.dataset_name_label.toolTip() == path.name


def test_recipe_role_edits_and_stale_report_survive_data_and_recipe_navigation(controller, qt, linear_df, tmp_path):
    import_frame(controller, qt, linear_df, tmp_path)
    screen = configure_regression(controller, qt)
    x1 = next(index for index, column in enumerate(screen.view["columns"]) if column["name"] == "x1")
    screen.columns.cellWidget(x1, 1).setCurrentText("categorical")
    screen.param_widgets["alpha"].setValue(2)
    screen.sections["settings"].setChecked(True)
    expected = deepcopy(screen.session_state())
    assert "previous result" in screen.result_state.text()
    original = controller.imported[0]
    controller.window.navigation_buttons["data"].click()
    assert controller.key == "data"
    assert next(column for column in controller.screen.view["columns"] if column["name"] == "x1")["type"] == "numeric"
    assert controller.imported[0] is original
    open_recipe(controller, "pca", qt)
    assert controller.screen.dataset is original and not controller.screen._report
    controller.show_data_page()
    open_recipe(controller, "linear-regression", qt)
    actual = controller.screen.session_state()
    for key in ("request", "params", "report", "report_request", "sections"):
        assert actual[key] == expected[key]
    assert "previous result" in controller.screen.result_state.text()


def test_failed_data_import_preserves_dataset_sessions_and_navigation(controller, qt, linear_df, tmp_path):
    import_frame(controller, qt, linear_df, tmp_path)
    screen = configure_regression(controller, qt)
    screen.param_widgets["alpha"].setValue(3)
    expected = deepcopy(screen.session_state())
    controller.show_data_page()
    original = controller.imported
    path = tmp_path / "bad.csv"
    path.write_text("duplicate,duplicate\n1,2", encoding="utf-8")
    controller.screen.load_requested.emit(str(path), ",", "utf-8-sig", "")
    finish(controller, qt)
    assert controller.imported is original
    assert controller.key == "data"
    assert "duplicate" in controller.screen.status.text().lower()
    assert controller.screen.load_button.isEnabled()
    assert all(button.isEnabled() for button in controller.window.navigation_buttons.values())
    assert controller.window.navigation_buttons["data"].isChecked()
    open_recipe(controller, "linear-regression", qt)
    for key in ("request", "report", "report_request", "params"):
        assert controller.screen.session_state()[key] == expected[key]


def test_recipe_replacement_clears_an_earlier_data_import_error(controller, qt, linear_df, tmp_path):
    import_frame(controller, qt, linear_df, tmp_path)
    invalid = tmp_path / "duplicate.csv"
    invalid.write_text("duplicate,duplicate\n1,2", encoding="utf-8")
    controller.screen.load_requested.emit(str(invalid), ",", "utf-8-sig", "")
    finish(controller, qt)
    assert "Unable to complete" in controller.screen.status.text()
    previous_error = controller.data_status
    open_recipe(controller, "linear-regression", qt)
    replacement = tmp_path / "new measurements.csv"
    linear_df.head(23).to_csv(replacement, index=False)
    controller.screen.load_requested.emit(str(replacement), ",", "utf-8-sig", "")
    finish(controller, qt)
    assert controller.screen.dataset_name == replacement.name
    controller.show_data_page()
    assert controller.screen.view["name"] == replacement.name
    assert controller.screen.view["rows"] == 23
    assert controller.screen.status.text() == controller.data_status
    assert controller.data_status != previous_error
    assert "Unable to complete" not in controller.screen.status.text()
    assert "loaded" in controller.screen.status.text().lower()


def test_csv_source_summary_preserves_applied_options_separately_from_unrun_controls(controller, qt, tmp_path):
    path = tmp_path / "source options.csv"
    path.write_text("record_id;response;note\n001;1.25;café\n002;?;?\n", encoding="cp1252")
    controller.screen.load_requested.emit(str(path), ";", "cp1252", "?")
    finish(controller, qt)
    imported = controller.imported
    assert imported[0].df.note.iloc[0] == "café"
    assert imported[0].df.response.isna().sum() == 1
    screen = controller.screen
    screen.delimiter.setCurrentIndex(screen.delimiter.findData(","))
    screen.encoding.setCurrentIndex(screen.encoding.findData("utf-8-sig"))
    screen.markers.setText("MISSING")
    unrun = screen.import_options()
    controller.navigate("llm")
    controller.show_data_page()
    assert controller.imported is imported
    assert controller.screen.import_options() == unrun
    source = controller.screen.view["source"]
    assert source["delimiter"] == ";" and source["encoding"] == "cp1252"
    assert source["missing_markers"] == ["?"]
    summary = controller.screen.source.text()
    assert "semicolon delimiter" in summary and "cp1252 encoding" in summary
    assert "Extra missing markers: ?" in summary and "MISSING" not in summary
    assert str(path.resolve()) in summary


def test_successful_shell_replacement_invalidates_sessions_even_at_same_path(controller, qt, linear_df, tmp_path):
    path = import_frame(controller, qt, linear_df, tmp_path)
    configure_regression(controller, qt)
    controller.show_data_page()
    open_recipe(controller, "llm", qt)
    controller.screen.param_widgets["prompt"].setPlainText("Unsaved prompt for the first dataset")
    controller.show_data_page()
    assert {"linear-regression", "llm"}.issubset(controller.sessions)
    previous = controller.imported[0]
    linear_df.head(27).to_csv(path, index=False)
    controller.screen.load_requested.emit(str(path), ",", "utf-8-sig", "")
    finish(controller, qt)
    assert controller.imported[0] is not previous
    assert controller.sessions == {}
    assert controller.screen.view["rows"] == 27
    open_recipe(controller, "linear-regression", qt)
    assert len(controller.screen.dataset.df) == 27
    assert controller.screen.request()["target"] is None and not controller.screen._report


def test_failed_experiment_open_from_populated_data_retains_file_and_recipe_work(controller, qt, linear_df, tmp_path):
    import_frame(controller, qt, linear_df, tmp_path)
    screen = configure_regression(controller, qt)
    screen.param_widgets["alpha"].setValue(4)
    expected = deepcopy(screen.session_state())
    controller.show_data_page()
    original = controller.imported
    data_screen = controller.screen
    data_screen.markers.setText("?, -999")
    saved_sessions = dict(controller.sessions)
    invalid = tmp_path / "damaged.mlcb"
    invalid.write_text("This is not an experiment archive.", encoding="utf-8")
    controller.open_experiment(str(invalid))
    finish(controller, qt)
    assert controller.key == "data" and controller.screen is data_screen
    assert controller.imported is original and controller.sessions == saved_sessions
    assert data_screen.markers.text() == "?, -999"
    assert "Unable to complete" in data_screen.status.text()
    assert data_screen.load_button.isEnabled()
    assert controller.window.navigation_buttons["data"].isChecked()
    open_recipe(controller, "linear-regression", qt)
    for key in ("request", "report", "report_request", "params"):
        assert controller.screen.session_state()[key] == expected[key]


@pytest.mark.parametrize("fail", [False, True])
def test_background_import_locks_all_navigation_and_recovers_on_success_or_error(controller, qt, linear_df, tmp_path, monkeypatch, fail):
    from mlcookbook import io

    path = tmp_path / "blocked.csv"
    linear_df.to_csv(path, index=False)
    release = threading.Event()
    original = io.load_dataset_with_report

    def blocked(*args, **kwargs):
        assert release.wait(10), "Test failed to release its import worker"
        if fail:
            raise ValueError("Deliberate import failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(io, "load_dataset_with_report", blocked)
    screen = controller.screen
    try:
        screen.load_requested.emit(str(path), ",", "utf-8-sig", "")
        first_job = controller.job
        assert first_job is not None and not screen.load_button.isEnabled()
        assert not controller.window.open_experiment_button.isEnabled()
        assert not controller.window.save_experiment_button.isEnabled()
        assert not any(button.isEnabled() for button in controller.window.navigation_buttons.values())
        controller.navigate("llm")
        controller.show_front_page()
        controller.open_experiment(str(tmp_path / "absent.mlcb"))
        assert controller.job is first_job and controller.screen is screen and controller.key == "data"
    finally:
        release.set()
        finish(controller, qt)
    assert controller.key == "data" and controller.screen is screen
    assert screen.load_button.isEnabled() and controller.window.open_experiment_button.isEnabled()
    assert not controller.window.save_experiment_button.isEnabled()
    assert all(button.isEnabled() for button in controller.window.navigation_buttons.values())
    assert controller.window.navigation_buttons["data"].isChecked()
    assert (controller.imported is None) is fail
    if fail:
        assert "Deliberate import failure" in screen.status.text()


def test_data_import_options_survive_navigation_without_becoming_recipe_options(controller, qt, linear_df, tmp_path):
    import_frame(controller, qt, linear_df, tmp_path)
    screen = controller.screen
    screen.delimiter.setCurrentIndex(screen.delimiter.findData(";"))
    screen.encoding.setCurrentIndex(screen.encoding.findData("cp1252"))
    screen.markers.setText("?, -999")
    expected = screen.import_options()
    open_recipe(controller, "linear-regression", qt)
    assert controller.screen.delimiter.currentData() == ","
    controller.show_data_page()
    assert controller.screen.import_options() == expected


def test_saved_regular_experiment_updates_active_data_and_keeps_role_overrides_scoped(controller, qt, linear_df, tmp_path):
    from mlcookbook.core.results import to_jsonable

    source = import_frame(controller, qt, linear_df, tmp_path)
    screen = configure_regression(controller, qt)
    x1 = next(index for index, column in enumerate(screen.view["columns"]) if column["name"] == "x1")
    screen.columns.cellWidget(x1, 1).setCurrentText("categorical")
    expected = to_jsonable(screen.session_state())
    saved = tmp_path / "retained.mlcb"
    controller.save_experiment(str(saved))
    finish(controller, qt)
    import_frame(controller, qt, linear_df.head(20), tmp_path, "replacement.csv")
    controller.screen.delimiter.setCurrentIndex(controller.screen.delimiter.findData("\t"))
    controller.screen.markers.setText("?, -999")
    workspace_options = controller.screen.import_options()
    source.unlink()
    controller.open_experiment(str(saved))
    finish(controller, qt)
    assert controller.key == "linear-regression"
    assert controller.screen.request() == expected["request"]
    assert controller.screen._report == expected["report"]
    assert "saved dataset snapshot" in controller.screen.input_context.text()
    assert controller.window.navigation_buttons["recipes"].isChecked()
    restored = controller.imported[0]
    assert len(restored.df) == len(linear_df)
    controller.show_data_page()
    assert controller.imported[0] is restored
    assert controller.screen.import_options() == workspace_options
    assert next(column for column in controller.screen.view["columns"] if column["name"] == "x1")["type"] == "numeric"
    assert any("Saved dataset snapshot" in note for note in controller.screen.view["notes"])


@pytest.mark.parametrize("opening_from", ["data", "recipe"])
def test_study_import_and_saved_reopen_preserve_active_ordinary_dataset(controller, qt, linear_df, tmp_path, monkeypatch, opening_from):
    from tests.test_wearable import make_export

    import_frame(controller, qt, linear_df, tmp_path)
    ordinary = controller.imported
    open_recipe(controller, "linear-regression", qt)
    controller.screen.set_status("Review the ordinary recipe's selected columns before running.")
    recipe_status = controller.screen._status_message
    controller.navigate("wearable-study")
    assert not controller.screen.run_button.isEnabled()
    study = make_export(tmp_path / "study")
    controller.load_wearable_study(str(study / "export.json"))
    finish(controller, qt)
    assert controller.imported is ordinary
    assert controller.screen.source["kind"] == "wearable_study"
    assert not controller.screen.request()["params"]["final_test"]
    assert "Movement Study" in controller.screen.input_context.text()
    assert controller.window.dataset_name_label.toolTip() == ordinary[1]
    saved = tmp_path / "study.mlcb"
    controller.save_experiment(str(saved))
    finish(controller, qt)
    controller.show_data_page()
    assert controller.screen.view["name"] == ordinary[1]
    assert controller.screen.view["rows"] == len(linear_df)
    controller._set_status("Review the active dataset's source roles before choosing an analysis.")
    data_status = controller.data_status
    if opening_from == "recipe":
        open_recipe(controller, "linear-regression", qt)
        assert controller.screen._status_message == recipe_status

    def forbidden(*args, **kwargs):
        pytest.fail("Reopening a saved study must not evaluate a holdout")

    monkeypatch.setattr(app, "_run_report", forbidden)
    controller.open_experiment(str(saved))
    finish(controller, qt)
    assert controller.imported is ordinary
    assert controller.key == "wearable-study"
    assert controller.window.navigation_buttons["wearable-study"].isChecked()
    assert controller.screen.source["kind"] == "wearable_study"
    assert not controller.screen._report and not controller.screen.request()["params"]["final_test"]
    assert controller.window.dataset_name_label.toolTip() == ordinary[1]
    controller.show_data_page()
    assert controller.screen.status.text() == data_status
    open_recipe(controller, "linear-regression", qt)
    assert controller.screen._status_message == recipe_status
    assert controller.imported is ordinary


def test_typed_prompt_and_dataset_rows_have_distinct_effective_input(controller, qt, linear_df, tmp_path):
    path = import_frame(controller, qt, linear_df, tmp_path)
    ordinary = controller.imported
    controller.window.navigation_buttons["llm"].click()
    screen = controller.screen
    assert screen.input_context.text() == "Input: typed prompt"
    assert controller.window.dataset_name_label.toolTip() == path.name
    mode = screen.param_widgets["input_mode"]
    mode.setCurrentIndex(mode.findData("rows"))
    assert path.name in screen.input_context.text() and "typed prompt" not in screen.input_context.text()
    mode.setCurrentIndex(mode.findData("prompt"))
    assert screen.input_context.text() == "Input: typed prompt"
    assert controller.imported is ordinary and not screen._report and controller.job is None


def test_example_input_does_not_appear_as_an_active_ordinary_dataset(controller, qt):
    open_recipe(controller, "linear-regression", qt)
    controller.run_example()
    finish(controller, qt)
    assert controller.screen.is_example
    assert "built-in example" in controller.screen.input_context.text()
    assert controller.imported is None
    assert controller.window.dataset_name_label.toolTip() == "No dataset loaded"


def test_derived_dataset_sidebar_navigation_retains_the_llm_source_run(controller, qt, ollama_server):
    from tests.test_app_import import configure_text_workflow

    screen = configure_text_workflow(controller, ollama_server)
    original = screen.dataset
    screen.run_button.click()
    finish(controller, qt)
    run = deepcopy(screen._report["workflow_run"])
    controller.workflow_action("derive", None)
    finish(controller, qt)
    derived = controller.imported
    assert "score" in derived[0].df and "score" not in original.df
    controller.window.navigation_buttons["data"].click()
    assert controller.screen.view["source"]["kind"] == "llm_derived"
    assert controller.screen.view["rows"] == len(original.df)
    controller.window.navigation_buttons["llm"].click()
    assert controller.screen.dataset is original
    assert controller.screen._report["workflow_run"] == run
    assert controller.imported is derived
    controller.show_front_page()
    open_recipe(controller, "linear-regression", qt)
    assert controller.screen.dataset is derived[0]
    assert controller.screen.source["kind"] == "llm_derived"
