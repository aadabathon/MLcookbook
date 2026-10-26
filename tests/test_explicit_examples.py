"""Examples run by explicit choice and retain the ordinary-data workspace."""

import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QPushButton

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402 -- app.py is the intended desktop/core seam.
from gui.recipes import RECIPES  # noqa: E402

CLASSICAL = [key for key in app.DEMOS if key not in ("llm", "wearable-study")]


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


def open_recipe(controller, key):
    controller.open_recipe(next(recipe for recipe in RECIPES if recipe.key == key))


def import_frame(controller, qt, frame, tmp_path):
    path = tmp_path / "my measurements.csv"
    frame.to_csv(path, index=False)
    controller.show_data_page()
    controller.screen.load_requested.emit(str(path), ",", "utf-8-sig", "")
    finish(controller, qt)
    assert controller.imported is not None
    return path


def own_linear_result(controller, qt, linear_df, tmp_path):
    import_frame(controller, qt, linear_df, tmp_path)
    open_recipe(controller, "linear-regression")
    screen = controller.screen
    screen.target.setCurrentIndex(screen.target.findData("y"))
    screen.run_button.click()
    finish(controller, qt)
    assert screen._report and not screen.is_example
    return screen


def comparable_state(screen):
    state = screen.session_state()
    return {key: deepcopy(state[key]) for key in
            ("request", "params", "report", "report_request", "sections", "is_example")}


@pytest.mark.parametrize("key", CLASSICAL)
def test_classical_navigation_is_setup_only_without_preset_construction_or_training(controller, qt, monkeypatch, key):
    def forbidden(*args, **kwargs):
        pytest.fail("Navigation must not construct a preset or run an analysis")

    for recipe_key in CLASSICAL:
        _, name, target, positive = app.DEMOS[recipe_key]
        monkeypatch.setitem(app.DEMOS, recipe_key, (forbidden, name, target, positive))
        monkeypatch.setitem(app.REPORT_BUILDERS, recipe_key, forbidden)
    open_recipe(controller, key)
    assert controller.job is None
    assert not controller.screen._report
    assert not controller.screen.is_example
    assert not controller.screen.view["columns"]
    assert not controller.screen.run_button.isEnabled()
    assert not controller.window.save_experiment_button.isEnabled()
    assert controller.imported is None
    controller.show_data_page()
    open_recipe(controller, key)
    assert controller.job is None and not controller.screen._report
    assert controller.imported is None
    qt.processEvents()


def test_explicit_example_runs_defaults_and_reports_its_own_dataset(controller, qt, monkeypatch):
    calls = []
    original_frame, name, target, positive = app.DEMOS["linear-regression"]
    original_report = app.REPORT_BUILDERS["linear-regression"]

    def frame():
        calls.append("frame")
        return original_frame()

    def report(dataset, request, dataset_name):
        calls.append((deepcopy(request), dataset_name))
        return original_report(dataset, request, dataset_name)

    monkeypatch.setitem(app.DEMOS, "linear-regression", (frame, name, target, positive))
    monkeypatch.setitem(app.REPORT_BUILDERS, "linear-regression", report)
    open_recipe(controller, "linear-regression")
    assert not calls
    controller.screen.example_requested.emit()
    finish(controller, qt)
    screen = controller.screen
    assert calls[0] == "frame" and len(calls) == 2
    assert calls[1][0]["target"] == target and calls[1][1] == name
    assert screen.is_example and screen.source["kind"] == "demo"
    assert screen.dataset_name == name and name in screen._report["dataset"]
    assert "built-in example" in screen.input_context.text().lower()
    assert "Built-in example" in screen.result_state.text()
    assert controller.imported is None
    assert controller.window.dataset_name_label.accessibleName() == "No dataset loaded"
    expected = comparable_state(screen)
    controller.show_data_page()
    open_recipe(controller, "linear-regression")
    assert comparable_state(controller.screen) == expected and len(calls) == 2


def test_example_switch_and_navigation_preserve_own_data_roles_results_and_unrun_edits(controller, qt, linear_df, tmp_path):
    screen = own_linear_result(controller, qt, linear_df, tmp_path)
    row = next(index for index, column in enumerate(screen.view["columns"]) if column["name"] == "x1")
    screen.columns.cellWidget(row, 1).setCurrentText("categorical")
    screen.param_widgets["alpha"].setValue(2)
    screen.sections["settings"].setChecked(True)
    own = comparable_state(screen)
    workspace = controller.imported
    assert "previous result" in screen.result_state.text()
    controller.run_example()
    finish(controller, qt)
    assert controller.imported is workspace
    assert controller.screen.dataset is not workspace[0]
    assert controller.screen.is_example and controller.screen.request()["target"] == "progression"
    example = comparable_state(controller.screen)
    controller.navigate("data")
    assert controller.screen.view["name"] == workspace[1]
    assert next(column for column in controller.screen.view["columns"] if column["name"] == "x1")["type"] == "numeric"
    open_recipe(controller, "pca")
    assert controller.screen.dataset is workspace[0] and not controller.screen._report
    controller.show_front_page()
    open_recipe(controller, "linear-regression")
    assert comparable_state(controller.screen) == example and controller.imported is workspace
    controller.screen.workspace_requested.emit()
    assert controller.job is None and controller.imported is workspace
    assert comparable_state(controller.screen) == own
    assert "previous result" in controller.screen.result_state.text()
    assert controller.screen.dataset is workspace[0]


def test_example_rerun_uses_example_edits_and_never_applies_them_to_own_data(controller, qt, linear_df, tmp_path):
    screen = own_linear_result(controller, qt, linear_df, tmp_path)
    own = comparable_state(screen)
    workspace = controller.imported
    controller.run_example()
    finish(controller, qt)
    controller.screen.param_widgets["alpha"].setValue(3)
    expected_request = deepcopy(controller.screen.request())
    controller.run_example()
    finish(controller, qt)
    assert controller.screen.is_example
    assert controller.screen._report_request == expected_request
    assert controller.imported is workspace
    controller.use_workspace_data()
    assert comparable_state(controller.screen) == own


@pytest.mark.parametrize("phase", ["load", "run"])
def test_failed_example_preserves_completed_own_result_and_context(controller, qt, linear_df, tmp_path, monkeypatch, phase):
    screen = own_linear_result(controller, qt, linear_df, tmp_path)
    before = comparable_state(screen)
    workspace = controller.imported

    def fail(*args, **kwargs):
        raise ValueError("Example " + phase + " failed")

    if phase == "load":
        _, name, target, positive = app.DEMOS["linear-regression"]
        monkeypatch.setitem(app.DEMOS, "linear-regression", (fail, name, target, positive))
    else:
        monkeypatch.setitem(app.REPORT_BUILDERS, "linear-regression", fail)
    controller.run_example()
    finish(controller, qt)
    assert controller.screen is screen
    assert controller.imported is workspace and comparable_state(screen) == before
    assert not screen.is_example
    assert "Example " + phase + " failed" in screen.status.text()
    assert all(button.isEnabled() for button in controller.window.navigation_buttons.values())


def test_example_background_job_locks_navigation_and_context_switches(controller, qt, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original, name, target, positive = app.DEMOS["linear-regression"]

    def frame():
        entered.set()
        assert release.wait(10), "Test did not release example loading"
        return original()

    monkeypatch.setitem(app.DEMOS, "linear-regression", (frame, name, target, positive))
    open_recipe(controller, "linear-regression")
    screen = controller.screen
    try:
        controller.run_example()
        deadline = time.monotonic() + 5
        while not entered.is_set() and time.monotonic() < deadline:
            qt.processEvents()
            time.sleep(.005)
        assert entered.is_set() and controller.job is not None
        assert all(not button.isEnabled() for button in controller.window.navigation_buttons.values())
        assert not controller.window.open_experiment_button.isEnabled()
        controller.navigate("data")
        controller.use_workspace_data()
        assert controller.screen is screen and controller.key == "linear-regression"
    finally:
        release.set()
        finish(controller, qt)
    assert controller.screen.is_example and controller.screen._report
    assert all(button.isEnabled() for button in controller.window.navigation_buttons.values())


@pytest.mark.parametrize("with_workspace", [False, True])
def test_saved_example_reopens_offline_without_becoming_active_ordinary_data(controller, qt, linear_df, tmp_path, monkeypatch, with_workspace):
    if with_workspace:
        screen = own_linear_result(controller, qt, linear_df, tmp_path)
        screen.param_widgets["alpha"].setValue(4)
        own = comparable_state(screen)
    else:
        open_recipe(controller, "linear-regression")
    workspace = controller.imported
    controller.run_example()
    finish(controller, qt)
    expected = comparable_state(controller.screen)
    path = tmp_path / "explicit-example.mlcb"
    controller.save_experiment(str(path))
    finish(controller, qt)
    controller.show_data_page()
    if with_workspace:
        (tmp_path / "my measurements.csv").unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("Reopening a saved example must not build its preset or train")

    _, name, target, positive = app.DEMOS["linear-regression"]
    monkeypatch.setitem(app.DEMOS, "linear-regression", (forbidden, name, target, positive))
    monkeypatch.setitem(app.REPORT_BUILDERS, "linear-regression", forbidden)
    controller.open_experiment(str(path))
    finish(controller, qt)
    from mlcookbook.core.results import to_jsonable

    assert to_jsonable(comparable_state(controller.screen)) == to_jsonable(expected)
    assert controller.screen.source["kind"] == "demo" and controller.screen.is_example
    assert controller.imported is workspace
    assert "built-in example" in controller.screen.input_context.text().lower()
    if with_workspace:
        controller.use_workspace_data()
        assert comparable_state(controller.screen) == own
    else:
        controller.show_data_page()
        assert controller.screen.view is None


@pytest.mark.parametrize("key", ["llm", "wearable-study"])
def test_specialized_recipes_have_no_classical_example_action(controller, qt, monkeypatch, key):
    def forbidden(*args, **kwargs):
        pytest.fail("Opening a specialized setup must not run a workflow")

    monkeypatch.setitem(app.REPORT_BUILDERS, key, forbidden)
    open_recipe(controller, key)
    assert controller.job is None and not controller.screen._report
    buttons = [button for button in controller.screen.findChildren(QPushButton)
               if "example" in button.text().lower() and button.isVisibleTo(controller.screen)]
    assert not buttons
    screen = controller.screen
    controller.run_example()
    finish(controller, qt)
    assert controller.screen is screen and not screen._report and not screen.is_example
    assert controller.imported is None


def test_saved_example_in_a_fresh_controller_preserves_independent_study_session(controller, qt, tmp_path, monkeypatch):
    from mlcookbook.core.results import to_jsonable
    from tests.test_wearable import make_export

    open_recipe(controller, "linear-regression")
    controller.run_example()
    finish(controller, qt)
    expected = to_jsonable(comparable_state(controller.screen))
    saved = tmp_path / "saved-example.mlcb"
    controller.save_experiment(str(saved))
    finish(controller, qt)
    assert saved.is_file()
    study = make_export(tmp_path / "independent-study")
    fresh = app.MLCookbookApp()
    try:
        open_recipe(fresh, "wearable-study")
        fresh.load_wearable_study(str(study / "export.json"))
        finish(fresh, qt)
        fresh.screen.param_widgets["final_test"].setCurrentIndex(1)
        fresh.screen.param_widgets["random_state"].setValue(99)
        study_state = comparable_state(fresh.screen)
        study_dataset = fresh.screen.dataset

        def forbidden(*args, **kwargs):
            pytest.fail("Opening a saved example or returning to a study must not construct or train")

        _, name, target, positive = app.DEMOS["linear-regression"]
        monkeypatch.setitem(app.DEMOS, "linear-regression", (forbidden, name, target, positive))
        monkeypatch.setitem(app.REPORT_BUILDERS, "linear-regression", forbidden)
        monkeypatch.setitem(app.REPORT_BUILDERS, "wearable-study", forbidden)
        fresh.open_experiment(str(saved))
        finish(fresh, qt)
        assert to_jsonable(comparable_state(fresh.screen)) == expected
        assert fresh.screen.is_example and fresh.imported is None
        fresh.show_data_page()
        assert fresh.screen.view is None
        open_recipe(fresh, "wearable-study")
        assert fresh.screen.dataset is study_dataset and not fresh.screen._report
        assert comparable_state(fresh.screen) == study_state
        assert fresh.imported is None
    finally:
        fresh.wait_for_job()
        qt.processEvents()
        fresh.window.close()
        fresh.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("source_kind", ["empty", "demo"])
def test_llm_typed_prompt_with_zero_source_rows_saves_and_reopens_without_ordinary_data(
        controller, qt, tmp_path, monkeypatch, ollama_server, source_kind):
    import mlcookbook.ollama as ollama
    from mlcookbook.core.results import to_jsonable
    from mlcookbook.experiments import save_experiment

    open_recipe(controller, "llm")
    screen = controller.screen
    assert screen.dataset.df.empty and screen.view["rows"] == 0
    assert screen.source["kind"] == "empty" and controller.imported is None
    screen.param_widgets["backend"].setCurrentIndex(screen.param_widgets["backend"].findData("ollama"))
    screen.param_widgets["ollama_url"].setText(ollama_server.base_url)
    screen.connection_button.click()
    finish(controller, qt)
    screen.param_widgets["prompt"].setPlainText("A typed research prompt independent of tabular data")
    assert screen.run_button.isEnabled()
    screen.run_button.click()
    finish(controller, qt)
    assert screen._report["generation"]["outputs"][0]["text"] == "Hi 🌍"
    screen.param_widgets["prompt"].setPlainText("An unrun prompt edit")
    expected = to_jsonable(comparable_state(screen))
    path = tmp_path / (source_kind + "-typed-llm.mlcb")
    if source_kind == "empty":
        controller.save_experiment(str(path))
        finish(controller, qt)
    else:
        # Pre-stage-3 typed-prompt experiments used this provenance label even
        # though their is_example flag was false. Keep accepting that format.
        save_experiment(path, screen.dataset, recipe="llm", name=screen.dataset_name,
                        state=screen.session_state(), source={"kind": "demo", "recipe": "llm"})
    assert path.is_file()
    controller.show_data_page()
    assert controller.screen.view is None

    def offline(*args, **kwargs):
        raise ConnectionError("The inference server is offline")

    monkeypatch.setattr(ollama, "ollama_models", offline)
    monkeypatch.setattr(ollama, "generate_ollama", offline)
    controller.open_experiment(str(path))
    finish(controller, qt)
    assert controller.key == "llm" and controller.screen.dataset.df.empty
    assert to_jsonable(comparable_state(controller.screen)) == expected
    assert controller.screen.input_context.text() == "Input: typed prompt"
    assert not controller.screen.is_example and controller.imported is None
    controller.show_data_page()
    assert controller.screen.view is None


@pytest.mark.parametrize("workflow", ["rows", "extract"])
def test_empty_llm_source_blocks_row_and_structured_runs_until_data_is_imported(controller, workflow):
    open_recipe(controller, "llm")
    screen = controller.screen
    screen.param_widgets["backend"].setCurrentIndex(screen.param_widgets["backend"].findData("ollama"))
    screen.param_widgets["ollama_model"].setEditText("installed:test")
    if workflow == "rows":
        screen.param_widgets["input_mode"].setCurrentIndex(screen.param_widgets["input_mode"].findData("rows"))
    else:
        screen.param_widgets["workflow"].setCurrentIndex(screen.param_widgets["workflow"].findData("extract"))
        screen.param_widgets["template"].setPlainText("Extract the label from {{prompt}}")
        screen.param_widgets["field_spec"].setPlainText("label: text")
    assert screen.view["rows"] == 0 and not screen._report
    assert not screen.run_button.isEnabled()
    assert "dataset" in screen._problem().lower()
    assert controller.job is None and controller.imported is None


def test_repeated_context_switches_keep_both_unrun_states_and_recover_from_failed_example(
        controller, qt, linear_df, tmp_path, monkeypatch):
    own_screen = own_linear_result(controller, qt, linear_df, tmp_path)
    own_screen.param_widgets["alpha"].setValue(5)
    own_first = comparable_state(own_screen)
    workspace = controller.imported
    controller.run_example()
    finish(controller, qt)
    controller.screen.param_widgets["alpha"].setValue(7)
    example_unrun = comparable_state(controller.screen)
    assert "previous result" in controller.screen.result_state.text()
    controller.show_data_page()
    open_recipe(controller, "linear-regression")
    assert comparable_state(controller.screen) == example_unrun
    controller.use_workspace_data()
    assert comparable_state(controller.screen) == own_first
    controller.screen.param_widgets["alpha"].setValue(6)
    own_second = comparable_state(controller.screen)
    controller.navigate("data")
    open_recipe(controller, "pca")
    open_recipe(controller, "linear-regression")
    assert comparable_state(controller.screen) == own_second
    original = app.REPORT_BUILDERS["linear-regression"]

    def fail(*args, **kwargs):
        raise ValueError("A later example run failed")

    monkeypatch.setitem(app.REPORT_BUILDERS, "linear-regression", fail)
    controller.run_example()
    assert controller.job is None and comparable_state(controller.screen) == example_unrun
    example_screen = controller.screen
    example_screen.run_button.click()
    finish(controller, qt)
    assert controller.screen is example_screen
    assert comparable_state(controller.screen) == example_unrun and controller.imported is workspace
    controller.use_workspace_data()
    assert comparable_state(controller.screen) == own_second
    monkeypatch.setitem(app.REPORT_BUILDERS, "linear-regression", original)
    controller.run_example()
    assert controller.job is None and comparable_state(controller.screen) == example_unrun
    controller.screen.run_button.click()
    finish(controller, qt)
    assert controller.screen.is_example and controller.screen.param_widgets["alpha"].value() == 7
    assert controller.screen._report_request == example_unrun["request"]
    controller.use_workspace_data()
    assert comparable_state(controller.screen) == own_second and controller.imported is workspace


@pytest.mark.parametrize("invalid", ["missing-target", "incompatible-role"])
def test_cached_example_with_invalid_edits_can_be_viewed_repaired_and_run(
        controller, qt, linear_df, tmp_path, monkeypatch, invalid):
    own_screen = own_linear_result(controller, qt, linear_df, tmp_path)
    own_screen.param_widgets["alpha"].setValue(4)
    own = comparable_state(own_screen)
    workspace = controller.imported
    controller.run_example()
    finish(controller, qt)
    screen = controller.screen
    last_report = deepcopy(screen._report)
    if invalid == "missing-target":
        screen.target.setCurrentIndex(screen.target.findData(None))
        assert not screen.run_button.isEnabled()
    else:
        role_row = next(index for index, column in enumerate(screen.view["columns"])
                        if column["name"] == "age")
        role = screen.columns.cellWidget(role_row, 1)
        original_role = role.currentText()
        role.setCurrentText("datetime")
        assert screen.run_button.isEnabled()
        screen.run_button.click()
        finish(controller, qt)
        assert "not stored as dates" in screen.status.text()
        assert screen._report == last_report
    invalid_state = comparable_state(screen)
    assert "previous result" in screen.result_state.text()
    controller.use_workspace_data()
    assert comparable_state(controller.screen) == own and controller.imported is workspace
    assert controller.screen.example_button.text() == "View example"

    def forbidden(*args, **kwargs):
        pytest.fail("Viewing a cached invalid example must not load a preset or run its model")

    with monkeypatch.context() as blocked:
        _, name, target, positive = app.DEMOS["linear-regression"]
        blocked.setitem(app.DEMOS, "linear-regression", (forbidden, name, target, positive))
        blocked.setitem(app.REPORT_BUILDERS, "linear-regression", forbidden)
        controller.screen.example_requested.emit()
        finish(controller, qt)
        assert controller.job is None and controller.screen.is_example
        assert comparable_state(controller.screen) == invalid_state
        assert controller.screen._report == last_report and controller.imported is workspace
        assert "previous result" in controller.screen.result_state.text()
        if invalid == "missing-target":
            assert not controller.screen.run_button.isEnabled()

    if invalid == "missing-target":
        controller.screen.target.setCurrentIndex(controller.screen.target.findData("progression"))
    else:
        controller.screen.columns.cellWidget(role_row, 1).setCurrentText(original_role)
    assert controller.screen.run_button.isEnabled()
    request = deepcopy(controller.screen.request())
    controller.screen.run_button.click()
    finish(controller, qt)
    assert controller.screen.is_example and controller.screen._report
    assert controller.screen._report_request == request
    assert "Results match" in controller.screen.result_state.text()
    controller.use_workspace_data()
    assert comparable_state(controller.screen) == own and controller.imported is workspace
