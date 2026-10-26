"""Integration checks for the desktop import and background analysis flow."""
import sys
import time
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QLabel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from gui.recipes import RECIPES


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
        time.sleep(0.005)
    assert controller.job is None


def configure_text_workflow(controller, server):
    import pandas as pd

    from mlcookbook import Dataset
    from tests.test_llm_workflow import ANSWER, ndjson

    open_recipe(controller, "llm")
    data = Dataset(pd.DataFrame({"text": ["great", "bad", None], "target": [10., 20., 30.]}, index=[8, 8, 3]))
    controller._use_import((data, "club.csv", app._dataset_view(data, "club.csv")))
    screen = controller.screen
    screen.param_widgets["workflow"].setCurrentIndex(screen.param_widgets["workflow"].findData("extract"))
    screen.param_widgets["backend"].setCurrentIndex(screen.param_widgets["backend"].findData("ollama"))
    screen.param_widgets["ollama_url"].setText(server.base_url)
    screen.param_widgets["ollama_model"].setEditText("test:tiny")
    screen.param_widgets["template"].setPlainText("Read {{text}}")
    screen.param_widgets["field_spec"].setPlainText("sentiment: categorical: positive | negative\nscore: numeric\nurgent: boolean")
    screen.param_widgets["max_rows"].setValue(3)
    server.state["raw_stream"] = ndjson(ANSWER)
    return screen


def test_text_workflow_disk_retry_compare_export_and_derived_data(controller, qt, ollama_server, tmp_path):
    import json

    from mlcookbook.core.results import to_jsonable

    screen = configure_text_workflow(controller, ollama_server)
    assert screen.run_button.isEnabled()
    screen.run_button.click()
    finish(controller, qt)
    first = screen._report["workflow_run"]
    assert [row["status"] for row in first["rows"]] == ["valid", "valid", "skipped"]
    assert screen.workflow_review.isVisibleTo(screen) and not screen.workflow_compare.isEnabled()
    controller.workflow_action("retry", [1])
    finish(controller, qt)
    assert screen._report["workflow_run"]["parent_run_id"] == first["run_id"]
    assert len(screen._report["workflow_history"]) == 2 and screen.workflow_compare.isEnabled()
    controller.workflow_action("compare", None)
    finish(controller, qt)
    assert len(screen._report["plots"]) == 6
    assert screen._report["tables"][0]["title"].startswith("Saved run comparison")
    assert "Compared 2 saved runs" in screen.status.text()
    export = tmp_path / "club.json"
    controller.workflow_action("export", str(export))
    finish(controller, qt)
    assert json.loads(export.read_text(encoding="utf-8"))["run"] == screen._report["workflow_run"]
    report = to_jsonable(screen._report)
    path = tmp_path / "club.mlcb"
    controller.save_experiment(str(path))
    finish(controller, qt)
    ollama_server.state["status"] = 500
    controller.open_experiment(str(path))
    finish(controller, qt)
    screen = controller.screen
    assert screen._report == report
    assert screen.param_widgets["template"].toPlainText() == "Read {{text}}"
    assert len(screen._report["workflow_history"]) == 2
    screen.param_widgets["template"].setPlainText("Changed {{text}}")
    controller.workflow_action("derive", None)
    assert "Settings changed" in screen.status.text()
    screen.param_widgets["template"].setPlainText("Read {{text}}")
    controller.workflow_action("derive", None)
    finish(controller, qt)
    assert controller.imported[0].df.index.tolist() == [8, 8, 3]
    assert controller.imported[0].df.score.iloc[:2].tolist() == [2.5, 2.5]
    assert controller.imported[0].df.score.iloc[2:].isna().all()
    assert "score" not in screen.dataset.df
    controller.show_front_page()
    open_recipe(controller, "linear-regression")
    assert "score" in controller.screen.dataset.df
    assert controller.screen.source["kind"] == "llm_derived"
    controller.show_front_page()
    open_recipe(controller, "llm")
    assert controller.screen._report["workflow_run"] == report["workflow_run"]
    assert "score" not in controller.screen.dataset.df


def test_text_workflow_unrun_settings_restore_and_legacy_migration(controller, qt, ollama_server, tmp_path):
    from copy import deepcopy

    from gui.screens.llm import WORKFLOW_DEFAULTS

    screen = configure_text_workflow(controller, ollama_server)
    screen.run_button.click()
    finish(controller, qt)
    screen.param_widgets["template"].setPlainText("Unrun {{text}}")
    path = tmp_path / "unrun.mlcb"
    controller.save_experiment(str(path))
    finish(controller, qt)
    controller.open_experiment(str(path))
    finish(controller, qt)
    assert controller.screen.param_widgets["template"].toPlainText() == "Unrun {{text}}"
    assert "Settings changed" in controller.screen.result_state.text()
    state = deepcopy(controller.screen.session_state())
    for values in (state["params"], state["request"]["params"], state["report_request"]["params"]):
        for key in WORKFLOW_DEFAULTS:
            values.pop(key)
    migrated = controller.screen.prepare_session(state)
    assert migrated["params"]["workflow"] == "generate"
    controller.screen.validate_session(migrated)


@pytest.mark.parametrize("task,field", [("extract", "topic"), ("summarize", "summary"),
                                       ("rewrite", "rewritten"), ("classify", "sentiment")])
def test_text_workflow_presets_bind_selected_columns(controller, ollama_server, task, field):
    from mlcookbook.llm_workflow import parse_fields

    screen = configure_text_workflow(controller, ollama_server)
    screen.param_widgets["workflow"].setCurrentIndex(screen.param_widgets["workflow"].findData(task))
    screen.preset_button.click()
    assert "{{text}}" in screen.param_widgets["template"].toPlainText()
    assert parse_fields(screen.param_widgets["field_spec"].toPlainText())[0]["name"] == field
    assert screen.param_rows["prompt"][1].isHidden()
    screen.param_widgets["backend"].setCurrentIndex(0)
    assert not screen.run_button.isEnabled()
    assert "Ollama or llama.cpp" in screen._problem()


def open_recipe(controller, key):
    controller.open_recipe(next(r for r in RECIPES if r.key == key))
    finish(controller, QApplication.instance())


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
@pytest.mark.parametrize("key,fixture", [("linear-regression", "linear_df"), ("logistic-regression", "logistic_df"), ("decision-tree", "logistic_df"), ("random-forest", "logistic_df")])
def test_import_run_and_switch(controller, qt, tmp_path, request, key, fixture, suffix):
    frame = request.getfixturevalue(fixture)
    path = tmp_path / f"my_data.{suffix}"
    if suffix == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    open_recipe(controller, key)
    screen = controller.screen
    controller.load_file(str(path), ",", "utf-8-sig")
    assert not screen.back_button.isEnabled()
    finish(controller, qt)
    assert screen.dataset_name == path.name
    assert not screen.run_button.isEnabled()
    screen.target.setCurrentIndex(screen.target.findData("y"))
    assert screen.run_button.isEnabled()
    if key != "linear-regression":
        screen.positive.setCurrentText("no")
        assert screen.request()["positive_class"] == "no"
    screen.run_button.click()
    finish(controller, qt)
    assert "Analysis complete" in screen.status.text()
    assert any("Dataset: " + path.name in label.text() for label in screen.report_widget.findChildren(QLabel))
    assert screen.back_button.isEnabled()
    selections = screen.request()
    controller.show_front_page()
    open_recipe(controller, key)
    assert controller.screen.dataset_name == path.name
    assert controller.screen.request() == selections
    assert "Dataset: " + path.name in labels_text(controller.screen.report_widget)


def test_failed_import_preserves_data(controller, qt, tmp_path):
    open_recipe(controller, "linear-regression")
    original = controller.screen.dataset
    path = tmp_path / "bad.csv"
    path.write_text("x,x\n1,2")
    controller.load_file(str(path), ",", "utf-8-sig")
    finish(controller, qt)
    assert controller.screen.dataset is original
    assert "duplicate" in controller.screen.status.text()
    assert controller.screen.load_button.isEnabled()


def test_column_roles_selection_and_failure_recovery(controller, qt, linear_df):
    from mlcookbook import Dataset
    open_recipe(controller, "linear-regression")
    data = Dataset(linear_df)
    controller._use_import((data, "custom", app._dataset_view(data, "custom")))
    screen = controller.screen
    screen.target.setCurrentIndex(screen.target.findData("y"))
    screen.columns.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert "x1" not in screen.request()["features"]
    assert "y" not in screen.request()["features"]
    group_row = list(linear_df.columns).index("group")
    screen.columns.cellWidget(group_row, 1).setCurrentText("numeric")
    screen.run_button.click()
    finish(controller, qt)
    assert "non-numeric" in screen.status.text()
    assert screen.run_button.isEnabled()
    screen.columns.cellWidget(group_row, 1).setCurrentText("categorical")
    screen.run_button.click()
    finish(controller, qt)
    assert "Analysis complete" in screen.status.text()


def labels_text(widget):
    return "\n".join(label.text() for label in widget.findChildren(QLabel))


def test_llm_opens_setup_without_loading_a_model(controller, monkeypatch):
    import mlcookbook.llm as llm

    monkeypatch.setattr(llm, "_load_model", lambda *args: pytest.fail("Opening the card must not load a model"))
    controller.open_recipe(next(recipe for recipe in RECIPES if recipe.key == "llm"))
    screen = controller.screen
    assert controller.job is None and not screen.run_button.isEnabled()
    assert "Choose a local" in labels_text(screen.report_widget)
    assert screen.preprocessing is None and screen.cv_folds is None
    assert screen.columns.isHidden()
    screen.param_widgets["input_mode"].setCurrentIndex(1)
    assert not screen.columns.isHidden()
    screen.param_widgets["model_path"].setText("local-model")
    screen.columns.item(0, 0).setCheckState(Qt.CheckState.Checked)
    assert screen.run_button.isEnabled()


def test_llm_run_save_restore_unrun_prompt_and_failed_rerun(controller, qt, tiny_llm, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QPlainTextEdit

    import mlcookbook.llm as llm
    from mlcookbook.core.results import to_jsonable

    open_recipe(controller, "llm")
    screen = controller.screen
    screen.param_widgets["model_path"].setText(str(tiny_llm))
    screen.param_widgets["prompt"].setPlainText("hello world")
    screen.param_widgets["max_new_tokens"].setValue(4)
    screen.run_button.click()
    assert not screen.run_button.isEnabled()
    finish(controller, qt)
    report = to_jsonable(screen.session_state()["report"])
    assert report["generation"]["outputs"][0]["input_text"] == "hello world"
    assert report["provenance"]["versions"]["torch"] != "unavailable"
    assert screen.report_widget.findChildren(QPlainTextEdit)[0].isReadOnly()
    screen.param_widgets["prompt"].setPlainText("hello data")
    assert "previous result" in screen.result_state.text()
    path = tmp_path / "generation.mlcb"
    controller.save_experiment(str(path))
    finish(controller, qt)
    controller.show_front_page()
    with monkeypatch.context() as patch:
        patch.setattr(llm, "_load_model", lambda *args: pytest.fail("Reopening must not load weights"))
        controller.open_experiment(str(path))
        finish(controller, qt)
    assert controller.key == "llm"
    screen = controller.screen
    assert screen.request()["params"]["prompt"] == "hello data"
    assert screen.session_state()["report"] == report
    assert "previous result" in screen.result_state.text()
    screen.param_widgets["model_path"].setText(str(tmp_path / "missing-model"))
    screen.run_button.click()
    finish(controller, qt)
    assert "Unable to complete" in screen.status.text()
    assert screen.session_state()["report"] == report
    assert screen.run_button.isEnabled()


def test_llm_batch_column_controls_reach_generation(controller, qt, tiny_llm):
    import pandas as pd

    open_recipe(controller, "llm")
    screen = use_frame(controller, pd.DataFrame({"question": ["hello", "world", "data"], "number": [1, 2, 3]}))
    screen.param_widgets["input_mode"].setCurrentIndex(1)
    screen.param_widgets["model_path"].setText(str(tiny_llm))
    screen.param_widgets["prompt"].setPlainText("answer")
    screen.param_widgets["max_new_tokens"].setValue(2)
    screen.param_widgets["max_rows"].setValue(2)
    screen.columns.item(0, 0).setCheckState(Qt.CheckState.Checked)
    screen.columns.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    screen.run_button.click()
    finish(controller, qt)
    result = screen.session_state()["report"]["generation"]
    assert result["text_column"] == "question"
    assert [output["row_position"] for output in result["outputs"]] == [0, 1]
    assert "answer" in result["outputs"][0]["formatted_prompt"]
    screen.columns.item(1, 0).setCheckState(Qt.CheckState.Checked)
    assert not screen.run_button.isEnabled()


def test_llm_invalid_saved_prompt_is_rejected(controller, qt, tmp_path):
    from mlcookbook.experiments import save_experiment

    open_recipe(controller, "llm")
    screen = controller.screen
    state = screen.session_state()
    state["params"]["prompt"] = ["invalid"]
    path = tmp_path / "invalid-llm.mlcb"
    save_experiment(path, screen.dataset, recipe="llm", name="invalid", state=state)
    controller.open_experiment(str(path))
    finish(controller, qt)
    assert controller.screen is screen
    assert "incompatible" in screen.status.text()


def test_llm_cancel_keeps_previous_report(controller, qt, tiny_llm, monkeypatch):
    import threading

    import mlcookbook.llm as llm

    open_recipe(controller, "llm")
    screen = controller.screen
    screen.param_widgets["model_path"].setText(str(tiny_llm))
    screen.param_widgets["max_new_tokens"].setValue(2)
    screen.param_widgets["prompt"].setPlainText("hello")
    screen.run_button.click()
    finish(controller, qt)
    previous = screen.session_state()["report"]
    loader = llm._load_model
    loading, release = threading.Event(), threading.Event()

    def slow_load(*args):
        loading.set()
        assert release.wait(5)
        return loader(*args)

    monkeypatch.setattr(llm, "_load_model", slow_load)
    screen.run_button.click()
    try:
        assert loading.wait(5)
        assert not screen.cancel_button.isHidden()
        screen.cancel_button.click()
    finally:
        release.set()
    finish(controller, qt)
    assert "cancelled" in screen.status.text()
    assert screen.cancel_button.isHidden()
    assert screen.session_state()["report"] == previous
    assert screen.run_button.isEnabled()


def setup_ollama(controller, qt, server):
    open_recipe(controller, "llm")
    screen = controller.screen
    screen.param_widgets["backend"].setCurrentIndex(1)
    screen.param_widgets["ollama_url"].setText(server.base_url)
    assert screen.param_widgets["model_path"].isHidden()
    assert not screen.param_widgets["ollama_model"].isHidden()
    assert not server.requests  # Selecting an engine never contacts a server.
    screen.connection_button.click()
    finish(controller, qt)
    assert screen.param_widgets["ollama_model"].currentText() == "test:tiny"
    assert "test-0.40" in screen.connection_status.text()
    assert "4,096 tokens" in screen.connection_status.text()
    assert screen.run_button.isEnabled()
    return screen


def test_ollama_app_save_reopen_offline_and_failed_rerun(controller, qt, ollama_server, tmp_path, monkeypatch):
    import mlcookbook.ollama as ollama
    from mlcookbook.core.results import to_jsonable
    from mlcookbook.experiments import load_experiment

    screen = setup_ollama(controller, qt, ollama_server)
    screen.run_button.click()
    finish(controller, qt)
    previous = to_jsonable(screen.session_state()["report"])
    assert previous["generation"]["backend"] == "ollama"
    assert previous["generation"]["outputs"][0]["text"] == "Hi 🌍"
    assert "torch" not in previous["provenance"]["versions"]
    assert screen.preview_box.isHidden()
    screen.param_widgets["prompt"].setPlainText("An unrun edit")
    path = tmp_path / "ollama.mlcb"
    controller.save_experiment(str(path))
    finish(controller, qt)
    assert "torch" not in load_experiment(path).versions
    controller.show_front_page()

    def unavailable(*args, **kwargs):
        raise ConnectionError("Ollama is not running")

    monkeypatch.setattr(ollama, "ollama_models", unavailable)
    controller.open_experiment(str(path))
    finish(controller, qt)
    screen = controller.screen
    assert screen.session_state()["report"] == previous
    assert screen.request()["params"]["prompt"] == "An unrun edit"
    assert screen.param_widgets["ollama_model"].currentText() == "test:tiny"
    assert "previous result" in screen.result_state.text()
    assert "Results restored" in screen.status.text()
    screen.run_button.click()
    finish(controller, qt)
    assert "not running" in screen.status.text()
    assert screen.session_state()["report"] == previous


@pytest.mark.parametrize("after_preview", [False, True])
def test_ollama_app_cancel_discards_preview_keeps_report(controller, qt, ollama_server, after_preview):
    screen = setup_ollama(controller, qt, ollama_server)
    screen.run_button.click()
    finish(controller, qt)
    previous = screen.session_state()
    ollama_server.started.clear()
    ollama_server.state["hold_after_preview" if after_preview else "hold"] = True
    screen.param_widgets["prompt"].setPlainText("new prompt")
    screen.run_button.click()
    assert ollama_server.started.wait(3)
    if after_preview:
        deadline = time.monotonic() + 3
        while screen.preview_box.isHidden() and time.monotonic() < deadline:
            qt.processEvents()
            time.sleep(0.005)
        assert screen.preview_text.toPlainText() == "Hi 🌍"
        assert "not a completed result" in screen.preview_label.text()
    assert screen.session_state()["report"] == previous["report"]
    screen.cancel_button.click()
    finish(controller, qt)
    assert "cancelled" in screen.status.text()
    assert screen.preview_box.isHidden() and not screen.preview_text.toPlainText()
    assert screen.session_state()["report"] == previous["report"]
    assert screen.session_state()["report_request"] == previous["report_request"]
    assert screen.run_button.isEnabled()


def test_ollama_app_rows_and_incomplete_stream_recovery(controller, qt, ollama_server):
    import pandas as pd

    screen = setup_ollama(controller, qt, ollama_server)
    use_frame(controller, pd.DataFrame({"question": ["one", None, "two"]}))
    screen.param_widgets["input_mode"].setCurrentIndex(1)
    screen.param_widgets["max_rows"].setValue(2)
    screen.columns.item(0, 0).setCheckState(Qt.CheckState.Checked)
    screen.run_button.click()
    finish(controller, qt)
    previous = screen.session_state()["report"]
    assert [output["row_position"] for output in previous["generation"]["outputs"]] == [0, 2]
    ollama_server.state["raw_stream"] = b'{"message":{"content":"partial"}}\n'
    screen.run_button.click()
    finish(controller, qt)
    assert "without a completed response" in screen._status_message
    assert screen.session_state()["report"] == previous
    assert screen.preview_box.isHidden()


def test_llm_legacy_experiment_migrates_without_loading(controller, qt, tmp_path, monkeypatch):
    import mlcookbook.llm as llm
    from gui.screens.llm import NEW_DEFAULTS
    from mlcookbook.experiments import save_experiment

    open_recipe(controller, "llm")
    screen = controller.screen
    screen.param_widgets["model_path"].setText("old/model/folder")
    screen.show_report({"summary": "Previously generated text"})
    screen.param_widgets["prompt"].setPlainText("An unrun edit")
    state = screen.session_state()
    for values in (state["params"], state["request"]["params"], state["report_request"]["params"]):
        for key in NEW_DEFAULTS:
            values.pop(key)
    path = tmp_path / "legacy.mlcb"
    save_experiment(path, screen.dataset, recipe="llm", name="legacy", state=state)
    monkeypatch.setattr(llm, "_load_model", lambda *args: pytest.fail("Reopen must not load weights"))
    controller.open_experiment(str(path))
    finish(controller, qt)
    screen = controller.screen
    assert screen.request()["params"]["backend"] == "pytorch"
    assert screen.request()["params"]["model_path"] == "old/model/folder"
    assert screen.request()["params"]["prompt"] == "An unrun edit"
    assert screen.session_state()["report"] == state["report"]
    assert "previous result" in screen.result_state.text()


def setup_llamacpp(controller, qt, server):
    open_recipe(controller, "llm")
    screen = controller.screen
    screen.param_widgets["backend"].setCurrentIndex(2)
    screen.param_widgets["llamacpp_url"].setText(server.base_url)
    assert not server.requests
    for key in ("model_path", "context_tokens", "thinking", "prompt_format"):
        assert screen.param_widgets[key].isHidden()
    assert not screen.param_widgets["gguf_path"].isHidden()
    screen.connection_button.click()
    finish(controller, qt)
    return screen


def test_llamacpp_app_save_restore_unrun_edits_and_no_server(controller, qt, llamacpp_server, tmp_path, monkeypatch):
    import mlcookbook.llamacpp as cpp
    from mlcookbook.core.results import to_jsonable

    screen = setup_llamacpp(controller, qt, llamacpp_server)
    assert screen.param_widgets["llamacpp_model"].currentText() == "test-gguf"
    assert "b-test" in screen.connection_status.text()
    screen.run_button.click()
    finish(controller, qt)
    previous = to_jsonable(screen.session_state()["report"])
    assert previous["generation"]["outputs"][0]["text"] == "Hi 🌍"
    assert "torch" not in previous["provenance"]["versions"]
    assert "Selected file SHA-256" not in table_texts(screen.report_widget)
    screen.param_widgets["prompt"].setPlainText("An unrun llama.cpp edit")
    path = tmp_path / "llamacpp.mlcb"
    controller.save_experiment(str(path))
    finish(controller, qt)
    controller.show_front_page()

    def offline(*args, **kwargs):
        raise ConnectionError("llama-server is not running")

    monkeypatch.setattr(cpp, "llamacpp_models", offline)
    controller.open_experiment(str(path))
    finish(controller, qt)
    screen = controller.screen
    assert screen.request()["params"]["prompt"] == "An unrun llama.cpp edit"
    assert screen.session_state()["report"] == previous
    assert "Results restored" in screen.status.text()
    screen.run_button.click()
    finish(controller, qt)
    assert "not running" in screen.status.text()
    assert screen.session_state()["report"] == previous


@pytest.mark.parametrize("after_preview", [False, True])
def test_llamacpp_app_cancel_preserves_completed_result(controller, qt, llamacpp_server, after_preview):
    screen = setup_llamacpp(controller, qt, llamacpp_server)
    screen.run_button.click()
    finish(controller, qt)
    previous = screen.session_state()
    llamacpp_server.started.clear()
    llamacpp_server.state["hold_after_preview" if after_preview else "hold"] = True
    screen.param_widgets["prompt"].setPlainText("A new prompt")
    screen.run_button.click()
    assert llamacpp_server.started.wait(3)
    if after_preview:
        deadline = time.monotonic() + 3
        while screen.preview_box.isHidden() and time.monotonic() < deadline:
            qt.processEvents()
            time.sleep(0.005)
        assert "Hi 🌍" in screen.preview_text.toPlainText()
        assert "A thought" in screen.preview_text.toPlainText()
    assert screen.session_state()["report"] == previous["report"]
    screen.cancel_button.click()
    finish(controller, qt)
    assert "cancelled" in screen.status.text()
    assert screen.session_state()["report"] == previous["report"]
    assert screen.session_state()["report_request"] == previous["report_request"]
    assert screen.preview_box.isHidden()
    assert screen.run_button.isEnabled()


def test_llamacpp_app_loading_rows_and_broken_stream(controller, qt, llamacpp_server):
    import pandas as pd

    llamacpp_server.state["loading"] = True
    screen = setup_llamacpp(controller, qt, llamacpp_server)
    assert "loading" in screen.status.text() and not screen.run_button.isEnabled()
    llamacpp_server.state["loading"] = False
    screen.connection_button.click()
    finish(controller, qt)
    assert screen.run_button.isEnabled()
    use_frame(controller, pd.DataFrame({"text": ["one", None, "two"]}))
    screen.param_widgets["input_mode"].setCurrentIndex(1)
    screen.param_widgets["max_rows"].setValue(2)
    screen.columns.item(0, 0).setCheckState(Qt.CheckState.Checked)
    screen.run_button.click()
    finish(controller, qt)
    previous = screen.session_state()["report"]
    assert [output["row_position"] for output in previous["generation"]["outputs"]] == [0, 2]
    llamacpp_server.state["raw_stream"] = b'data: {"choices":[]}\n\n'
    screen.run_button.click()
    finish(controller, qt)
    assert "without a completed response" in screen._status_message
    assert screen.session_state()["report"] == previous


def test_llm_legacy_ollama_state_migrates_without_contacting_server(controller, qt, tmp_path, monkeypatch):
    import mlcookbook.ollama as ollama
    from gui.screens.llm import LLAMACPP_DEFAULTS
    from mlcookbook.experiments import save_experiment

    open_recipe(controller, "llm")
    screen = controller.screen
    screen.param_widgets["backend"].setCurrentIndex(1)
    screen.param_widgets["ollama_model"].setEditText("old-local-model")
    screen.show_report({"summary": "Saved Ollama result"})
    state = screen.session_state()
    for values in (state["params"], state["request"]["params"], state["report_request"]["params"]):
        for key in LLAMACPP_DEFAULTS:
            values.pop(key)
    path = tmp_path / "old-ollama.mlcb"
    save_experiment(path, screen.dataset, recipe="llm", name="old-ollama", state=state)
    monkeypatch.setattr(ollama, "ollama_models", lambda *args, **kwargs: pytest.fail("Reopening must not contact a server"))
    controller.open_experiment(str(path))
    finish(controller, qt)
    assert controller.screen.request()["params"]["backend"] == "ollama"
    assert controller.screen.request()["params"]["ollama_model"] == "old-local-model"
    assert controller.screen.session_state()["report"] == state["report"]


def table_texts(widget):
    from PySide6.QtWidgets import QTableWidget
    return [table.item(r, c).text() for table in widget.findChildren(QTableWidget)
            for r in range(table.rowCount()) for c in range(table.columnCount()) if table.item(r, c)]


def use_frame(controller, frame, name="custom"):
    from mlcookbook import Dataset
    data = Dataset(frame)
    controller._use_import((data, name, app._dataset_view(data, name)))
    return controller.screen


def test_multiclass_target_runs(controller, qt, logistic_df):
    open_recipe(controller, "logistic-regression")
    logistic_df.loc[:20, "y"] = "third"
    screen = use_frame(controller, logistic_df, "three classes")
    screen.target.setCurrentIndex(screen.target.findData("y"))
    assert screen.positive.currentText() == "All classes (multiclass)"
    assert screen.request()["positive_class"] is None
    assert screen.run_button.isEnabled()
    screen.run_button.click()
    finish(controller, qt)
    assert "Analysis complete" in screen.status.text()
    assert "THIRD" in labels_text(screen.report_widget) or "third" in table_texts(screen.report_widget)


def test_continuous_target_cannot_be_classified(controller, linear_df):
    open_recipe(controller, "logistic-regression")
    screen = use_frame(controller, linear_df)
    screen.target.setCurrentIndex(screen.target.findData("y"))
    assert not screen.run_button.isEnabled()
    assert "2 to 20 classes" in screen.status.text()


def test_binary_positive_class_defaults_like_the_core(controller, logistic_df):
    open_recipe(controller, "logistic-regression")
    screen = use_frame(controller, logistic_df)
    screen.target.setCurrentIndex(screen.target.findData("y"))
    assert screen.request()["positive_class"] == "yes"


@pytest.mark.parametrize("key", [key for key in app.SCREEN_FACTORIES if key not in ("llm", "wearable-study")])
def test_every_recipe_renders_its_demo(qt, key):
    screen = app.SCREEN_FACTORIES[key]()
    assert "Dataset: " in labels_text(screen.report_widget)
    assert "Built-in example" in screen.status.text()
    assert screen.run_button.isEnabled()
    assert screen.session_state()["report"]["summary"] in labels_text(screen.report_widget)
    screen.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("key", ["decision-tree", "random-forest", "svm", "neural-network"])
def test_tree_recipes_switch_to_regression_for_numeric_targets(controller, qt, linear_df, key):
    open_recipe(controller, key)
    screen = use_frame(controller, linear_df)
    screen.target.setCurrentIndex(screen.target.findData("y"))
    assert screen.current_task() == "regression"
    assert not screen.positive.isVisible()
    screen.run_button.click()
    finish(controller, qt)
    assert "Analysis complete" in screen.status.text()
    assert "regression" in labels_text(screen.report_widget)


def test_model_settings_and_preprocessing_reach_the_core(controller, qt, logistic_df):
    open_recipe(controller, "decision-tree")
    logistic_df.loc[:30, "x1"] = None
    screen = use_frame(controller, logistic_df)
    screen.target.setCurrentIndex(screen.target.findData("y"))
    screen.param_widgets["max_depth"].setValue(2)
    preprocessing = dict(screen.preprocessing, numeric_impute="mean", categorical_encoding="ordinal")
    screen.set_preprocessing(preprocessing, 3)
    assert "CV 3 folds" in screen.preprocessing_summary.text()
    screen.run_button.click()
    finish(controller, qt)
    text = labels_text(screen.report_widget)
    assert "max_depth=2" in text and "3-fold" in text.lower()
    cells = table_texts(screen.report_widget)
    assert any("training mean" in cell for cell in cells)
    assert any("ordinal codes" in cell for cell in cells)


def test_linear_penalty_setting(controller, qt, linear_df):
    open_recipe(controller, "linear-regression")
    screen = use_frame(controller, linear_df)
    screen.target.setCurrentIndex(screen.target.findData("y"))
    screen.param_widgets["penalty"].setCurrentIndex(screen.param_widgets["penalty"].findData("lasso"))
    screen.run_button.click()
    finish(controller, qt)
    assert "lasso, alpha=1" in labels_text(screen.report_widget)


@pytest.mark.parametrize("key", ["kmeans", "pca", "association-scan"])
def test_unsupervised_recipes_run_on_imported_data(controller, qt, logistic_df, key):
    open_recipe(controller, key)
    screen = use_frame(controller, logistic_df)
    assert not screen.target.isVisible()
    assert screen.run_button.isEnabled()
    assert screen.request()["target"] is None
    screen.run_button.click()
    finish(controller, qt)
    assert "Analysis complete" in screen.status.text()


def test_scan_needs_two_columns(controller, logistic_df):
    open_recipe(controller, "association-scan")
    screen = use_frame(controller, logistic_df)
    for row in range(1, screen.columns.rowCount()):
        screen.columns.item(row, 0).setCheckState(Qt.CheckState.Unchecked)
    assert not screen.run_button.isEnabled()
    assert "at least 2" in screen.status.text()


def test_import_notes_and_data_checks_are_shown(controller, qt, tmp_path):
    path = tmp_path / "codes.csv"
    rows = "\n".join(f"{i:05d},{i % 7},{'?' if i == 3 else i % 2},{1000 + i}" for i in range(30))
    path.write_text("zip,score,flag,row_id\n" + rows + "\n", encoding="utf-8")
    open_recipe(controller, "linear-regression")
    controller.load_file(str(path), ",", "utf-8-sig", "?, -999")
    finish(controller, qt)
    screen = controller.screen
    assert any("leading zeros" in note for note in screen.view["notes"])
    assert any("missing" in note for note in screen.view["notes"])
    text = labels_text(screen.report_widget)
    assert "Data checks" in text and "row identifier" in text
    row = [c["name"] for c in screen.view["columns"]].index("row_id")
    assert screen.columns.item(row, 0).checkState() == Qt.CheckState.Unchecked  # likely IDs start unchecked


def test_replacing_tree_report_cancels_deferred_scroll(controller, qt, monkeypatch):
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *args: errors.append(args))
    open_recipe(controller, "decision-tree")
    controller.screen.show_report({})
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt.processEvents()
    assert errors == []


def test_imported_navigation_does_not_run_demo(controller, linear_df, monkeypatch):
    from mlcookbook import Dataset
    data = Dataset(linear_df)
    controller.imported = (data, "custom", app._dataset_view(data, "custom"))
    def unexpected(*args, **kwargs):
        pytest.fail("Navigation should wait for the user to run their imported data")
    monkeypatch.setitem(app.REPORT_BUILDERS, "linear-regression", unexpected)
    open_recipe(controller, "linear-regression")
    assert controller.screen.dataset is data


@pytest.mark.parametrize("key", [key for key in app.SCREEN_FACTORIES if key not in ("llm", "wearable-study")])
def test_session_preserves_settings_and_previous_result(controller, qt, key, monkeypatch):
    open_recipe(controller, key)
    screen = controller.screen
    previous_report = screen.session_state()["report"]
    # Leave an unrun change so restoring must distinguish controls from the displayed run.
    feature = screen.request()["features"][0]
    row = next(i for i, col in enumerate(screen.view["columns"]) if col["name"] == feature)
    screen.columns.item(row, 0).setCheckState(Qt.CheckState.Unchecked)
    if screen.preprocessing is not None:
        screen.set_preprocessing(dict(screen.preprocessing, scaling="robust"), 3)
    for spec in screen._params:
        widget = screen.param_widgets[spec["key"]]
        if spec["kind"] == "choice":
            widget.setCurrentIndex(widget.count() - 1)
        else:
            widget.setValue(widget.minimum())
    screen.delimiter.setCurrentIndex(1)
    screen.encoding.setCurrentIndex(1)
    screen.markers.setText("?, -999")
    for section in screen.sections.values():
        section.setChecked(True)
    expected = screen.session_state()
    assert "previous result" in screen.result_state.text()
    controller.show_front_page()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def unexpected(*args, **kwargs):
        pytest.fail("Returning to a recipe must restore its report without retraining")
    monkeypatch.setitem(app.REPORT_BUILDERS, key, unexpected)
    open_recipe(controller, key)
    restored = controller.screen.session_state()
    for field in ("request", "params", "report_request", "delimiter", "encoding", "markers", "sections", "status"):
        assert restored[field] == expected[field]
    assert restored["report"] == previous_report
    assert "previous result" in controller.screen.result_state.text()


def test_stale_state_clears_when_settings_are_reverted(controller):
    open_recipe(controller, "linear-regression")
    screen = controller.screen
    screen.param_widgets["alpha"].setValue(2)
    assert "previous result" in screen.result_state.text()
    screen.param_widgets["alpha"].setValue(1)
    assert "match current settings" in screen.result_state.text()


def test_failed_run_keeps_report_provenance_and_error_on_return(controller, qt, monkeypatch):
    open_recipe(controller, "linear-regression")
    screen = controller.screen
    previous = screen.session_state()
    screen.param_widgets["alpha"].setValue(2)

    def fail(*args):
        raise ValueError("Review these settings")
    monkeypatch.setattr(app, "_run_report", fail)
    screen.run_button.click()
    finish(controller, qt)
    assert "Unable to complete: Review these settings" == screen.status.text()
    assert screen.session_state()["report_request"] == previous["report_request"]
    assert screen.session_state()["report"] == previous["report"]
    controller.show_front_page()
    open_recipe(controller, "linear-regression")
    assert "Review these settings" in controller.screen.status.text()
    assert "previous result" in controller.screen.result_state.text()


def test_replacement_dataset_invalidates_other_recipe_sessions(controller, linear_df):
    open_recipe(controller, "linear-regression")
    first = use_frame(controller, linear_df, "same-name.csv")
    first.target.setCurrentIndex(first.target.findData("y"))
    controller.show_front_page()
    open_recipe(controller, "decision-tree")
    second = use_frame(controller, linear_df.iloc[:50].copy(), "same-name.csv")
    data = second.dataset
    controller.show_front_page()
    open_recipe(controller, "linear-regression")
    assert controller.screen.dataset is data
    assert controller.screen.view["rows"] == 50
    assert controller.screen.target.currentData() is None
    assert not controller.screen.session_state()["report"]
    assert not controller.screen.is_example


def test_failed_import_preserves_sessions(controller, qt, tmp_path):
    open_recipe(controller, "linear-regression")
    previous = controller.screen.session_state()
    controller.show_front_page()
    open_recipe(controller, "decision-tree")
    path = tmp_path / "invalid.csv"
    path.write_text("x,x\n1,2", encoding="utf-8")
    controller.load_file(str(path), ",", "utf-8-sig")
    finish(controller, qt)
    controller.show_front_page()
    open_recipe(controller, "linear-regression")
    assert controller.screen.session_state()["report"] == previous["report"]


def test_warning_summary_and_data_checks_remain_accessible(controller, linear_df, monkeypatch):
    open_recipe(controller, "linear-regression")
    screen = use_frame(controller, linear_df.assign(record_id=range(len(linear_df))))
    screen.show_report({"dataset": "custom", "warnings": ["Review the assumptions", "Check missing values"]})
    assert not screen.warning_button.isHidden()
    assert "2 warnings" in screen.warning_button.text()
    shown = []
    monkeypatch.setattr(screen, "_show_notes", lambda title, notes: shown.append((title, notes)))
    screen.warning_button.click()
    screen.checks_button.click()
    assert shown[0][1] == ["Review the assumptions", "Check missing values"]
    assert any("row identifier" in note for note in shown[1][1])
    screen.show_report({})
    assert screen.warning_button.isHidden()


@pytest.mark.parametrize("size", [(1080, 720), (1280, 860)])
def test_actions_stay_visible_with_expanded_settings(controller, qt, size):
    from PySide6.QtCore import QPoint, QRect

    from gui import theme
    original = qt.styleSheet()
    qt.setStyleSheet(theme.stylesheet())
    try:
        open_recipe(controller, "neural-network")
        controller.window.resize(*size)
        controller.window.show()
        screen = controller.screen
        for section in screen.sections.values():
            section.setChecked(True)
        qt.processEvents()
        assert (controller.window.width(), controller.window.height()) == size
        for position in (0, screen.form_panel.verticalScrollBar().maximum()):
            screen.form_panel.verticalScrollBar().setValue(position)
            qt.processEvents()
            for widget in (screen.run_button, screen.status, screen.result_state):
                bounds = QRect(widget.mapTo(controller.window, QPoint(0, 0)), widget.size())
                assert widget.isVisible()
                assert controller.window.rect().contains(bounds)
        assert not screen.form_panel.isAncestorOf(screen.run_button)
    finally:
        qt.setStyleSheet(original)


def test_demo_opens_and_runs_in_background(controller, qt, monkeypatch):
    from threading import Event, current_thread, main_thread
    started, release = Event(), Event()
    threads = []

    def demo(*args):
        threads.append(current_thread())
        started.set()
        assert release.wait(5)
        return {"dataset": "example"}

    monkeypatch.setitem(app.REPORT_BUILDERS, "linear-regression", demo)
    try:
        controller.open_recipe(next(r for r in RECIPES if r.key == "linear-regression"))
        assert started.wait(5)
        assert controller.window._screen is controller.screen
        assert controller.job is not None
        assert not controller.screen.run_button.isEnabled()
        assert not controller.screen.back_button.isEnabled()
        assert not controller.screen.progress.isHidden()
        assert threads[0] is not main_thread()
    finally:
        release.set()
        finish(controller, qt)
    assert controller.screen.run_button.isEnabled()
    assert controller.screen.progress.isHidden()


def test_recipe_cards_support_keyboard(qt):
    from PySide6.QtTest import QTest

    from gui.recipe_card import RecipeCard
    card = RecipeCard(RECIPES[0])
    chosen = []
    card.clicked.connect(chosen.append)
    assert card.focusPolicy() == Qt.FocusPolicy.StrongFocus
    QTest.keyClick(card, Qt.Key.Key_Return)
    QTest.keyClick(card, Qt.Key.Key_Space)
    assert chosen == [RECIPES[0], RECIPES[0]]
    card.deleteLater()


def test_report_summary_uses_the_evaluated_result():
    from types import SimpleNamespace
    data = SimpleNamespace(df=[0] * 100)
    result = SimpleNamespace(features=["x"], target="price",
                             test_metrics=SimpleNamespace(n_obs=25, mae=12.345))
    regression = app._base(result, data, "sample")["summary"]
    assert "25 held-out rows: 12.35 in price units" in regression
    result.classes = ["no", "yes"]
    result.positive_class = "yes"
    result.test_metrics = SimpleNamespace(n_obs=25, accuracy=.84)
    classification = app._base(result, data, "sample")["summary"]
    assert "84.0% of 25 held-out rows" in classification


def test_large_report_tables_keep_all_rows_in_a_bounded_view(controller, qt):
    from PySide6.QtWidgets import QTableWidget
    open_recipe(controller, "linear-regression")
    screen = controller.screen
    screen.show_report({"dataset": "sample", "tables": [{"title": "Rows", "headers": ["VALUE"],
                                                        "rows": [[str(i)] for i in range(500)]}]})
    controller.window.show()
    qt.processEvents()
    table = screen.report_widget.findChild(QTableWidget)
    assert table.rowCount() == 500
    assert table.height() <= 330
    table.scrollToBottom()
    assert table.verticalScrollBar().value() == table.verticalScrollBar().maximum() > 0
    assert table.item(499, 0).text() == "499"


def test_multiclass_hides_positive_class_selector(controller):
    open_recipe(controller, "neural-network")
    assert controller.screen.positive.isHidden()
    assert controller.screen.request()["positive_class"] is None


def test_long_errors_have_accessible_details(controller, qt, monkeypatch):
    from PySide6.QtCore import QPoint, QRect
    open_recipe(controller, "linear-regression")
    controller.window.resize(1080, 720)
    controller.window.show()
    screen = controller.screen
    error = "Unable to complete: " + "Explain how to correct the input. " * 100
    shown = []
    monkeypatch.setattr(screen, "_show_notes", lambda title, notes: shown.extend(notes))
    screen.set_status(error)
    qt.processEvents()
    assert screen.status_details.isVisible()
    assert len(screen.status.text()) <= 180
    assert screen.status.text().endswith("…")
    assert screen.session_state()["status"] == error
    screen.status_details.click()
    assert shown == [error]
    button_bounds = QRect(screen.run_button.mapTo(controller.window, QPoint(0, 0)), screen.run_button.size())
    assert controller.window.rect().contains(button_bounds)


def test_navigation_restores_scroll_positions(controller, qt):
    open_recipe(controller, "neural-network")
    controller.window.resize(1080, 720)
    controller.window.show()
    screen = controller.screen
    for section in screen.sections.values():
        section.setChecked(True)
    qt.processEvents()
    screen.form_panel.verticalScrollBar().setValue(100)
    screen.report_widget.verticalScrollBar().setValue(300)
    positions = screen.session_state()["scroll"]
    assert all(positions)
    controller.show_front_page()
    open_recipe(controller, "neural-network")
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt.processEvents()
    assert controller.screen.session_state()["scroll"] == positions


def test_navigation_restores_regression_task_and_role_overrides(controller, qt, linear_df):
    open_recipe(controller, "decision-tree")
    screen = use_frame(controller, linear_df)
    screen.target.setCurrentIndex(screen.target.findData("y"))
    # Keep an intentional override that differs from inference, and a regression result.
    screen.columns.cellWidget(0, 1).setCurrentText("categorical")
    screen.run_button.click()
    finish(controller, qt)
    assert "Analysis complete" in screen.status.text()
    expected = screen.request()
    controller.show_front_page()
    open_recipe(controller, "decision-tree")
    assert controller.screen.request() == expected
    assert controller.screen.current_task() == "regression"
    assert controller.screen.positive.isHidden()
    assert "match current settings" in controller.screen.result_state.text()


@pytest.mark.parametrize("key", list(app.SCREEN_FACTORIES))
def test_experiment_reopens_in_new_app_without_training(controller, qt, tmp_path, monkeypatch, key):
    from mlcookbook.core.results import to_jsonable

    open_recipe(controller, key)
    screen = controller.screen
    screen.sections["import"].setChecked(True)
    screen.markers.setText("?, -999")
    if screen.preprocessing is not None:
        screen.set_preprocessing(dict(screen.preprocessing, scaling="robust"), 3)
    previous = to_jsonable(screen.session_state())
    path = tmp_path / f"{key}.mlcb"
    controller.save_experiment(str(path))
    assert not controller.window.open_experiment_button.isEnabled()
    assert not controller.window.save_experiment_button.isEnabled()
    finish(controller, qt)
    assert path.exists() and "Saved" in screen.status.text()

    def unexpected(*args, **kwargs):
        pytest.fail("Opening a saved experiment must not train")

    monkeypatch.setattr(app, "_run_report", unexpected)
    reopened = app.MLCookbookApp()
    try:
        reopened.open_experiment(str(path))
        finish(reopened, qt)
        assert reopened.key == key
        restored = to_jsonable(reopened.screen.session_state())
        for field in ("request", "report", "report_request", "params", "markers", "sections", "is_example"):
            assert restored[field] == previous[field]
        assert reopened.screen.dataset.df.equals(screen.dataset.df)
        assert reopened.window.save_experiment_button.isEnabled()
        if screen.preprocessing is not None:
            assert "previous result" in reopened.screen.result_state.text()
    finally:
        reopened.wait_for_job()
        qt.processEvents()
        reopened.window.close()
        reopened.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_saved_import_is_independent_of_original_file_and_reruns(controller, qt, tmp_path, linear_df):
    from mlcookbook.experiments import load_experiment

    path = tmp_path / "study.csv"
    linear_df.to_csv(path, sep=";", index=False)
    open_recipe(controller, "model-comparison")
    controller.load_file(str(path), ";", "utf-8-sig", "?")
    finish(controller, qt)
    screen = controller.screen
    screen.target.setCurrentIndex(screen.target.findData("y"))
    screen.param_widgets["n_estimators"].setValue(10)
    screen.param_widgets["random_state"].setValue(17)
    screen.set_preprocessing(dict(screen.preprocessing, scaling="robust"), 3)
    screen.run_button.click()
    finish(controller, qt)
    result = screen.session_state()["report"]["comparison"]
    saved = tmp_path / "study.mlcb"
    controller.save_experiment(str(saved))
    finish(controller, qt)
    assert load_experiment(saved).source["delimiter"] == ";"
    assert load_experiment(saved).source["missing_markers"] == ["?"]
    path.unlink()
    controller.show_front_page()
    controller.open_experiment(str(saved))
    finish(controller, qt)
    controller.screen.run_button.click()
    finish(controller, qt)
    assert controller.screen.session_state()["report"]["comparison"] == result
    controller.show_front_page()
    open_recipe(controller, "linear-regression")
    assert controller.screen.dataset_name == "study.csv"
    assert controller.screen.source["delimiter"] == ";"
    assert not controller.screen.session_state()["report"]


@pytest.mark.parametrize("kind", ["corrupt", "recipe", "settings", "report"])
def test_failed_experiment_open_preserves_current_work(controller, qt, tmp_path, kind):
    from mlcookbook.experiments import save_experiment

    open_recipe(controller, "linear-regression")
    screen = controller.screen
    before = deepcopy(screen.session_state())
    path = tmp_path / "broken.mlcb"
    if kind == "corrupt":
        path.write_bytes(b"damaged archive")
    else:
        state = deepcopy(before)
        if kind == "settings":
            state["params"]["alpha"] = -100
        if kind == "report":
            state["report"]["metric_groups"] = [None]
        save_experiment(path, screen.dataset, recipe="unknown" if kind == "recipe" else "linear-regression",
                        name="broken", state=state)
    controller.open_experiment(str(path))
    finish(controller, qt)
    assert controller.screen is screen
    assert screen.request() == before["request"]
    assert screen.session_state()["report"] == before["report"]
    assert "Unable to complete" in screen.status.text()
    assert controller.window.open_experiment_button.isEnabled()


def test_empty_experiment_and_failed_open_from_home(controller, qt, tmp_path, linear_df):
    controller.open_experiment(str(tmp_path / "missing.mlcb"))
    finish(controller, qt)
    assert controller.screen is None
    assert "Unable to complete" in controller.window.statusBar().currentMessage()
    open_recipe(controller, "model-comparison")
    use_frame(controller, linear_df)
    assert controller.screen.request()["target"] is None
    path = tmp_path / "unrun.mlcb"
    controller.save_experiment(str(path))
    finish(controller, qt)
    controller.show_front_page()
    controller.open_experiment(str(path))
    finish(controller, qt)
    assert controller.screen.request()["target"] is None
    assert not controller.screen.session_state()["report"]
    assert not controller.screen.run_button.isEnabled()


def test_comparison_requires_cv_and_shows_shared_split(controller, qt):
    open_recipe(controller, "model-comparison")
    screen = controller.screen
    report = screen.session_state()["report"]
    assert "Shared split" in report["settings"]
    assert "selected before test evaluation" in report["summary"]
    assert len(report["comparison"]["candidates"]) == 2
    screen.set_preprocessing(screen.preprocessing, 0)
    assert not screen.run_button.isEnabled()
    assert "needs cross-validation" in screen.status.text()


def test_experiment_restores_scroll_and_does_not_accumulate_import_notes(controller, qt, tmp_path):
    from mlcookbook.experiments import load_experiment

    open_recipe(controller, "model-comparison")
    controller.window.resize(1280, 860)
    controller.window.show()
    controller.screen.sections["settings"].setChecked(True)
    qt.processEvents()
    controller.screen.form_panel.verticalScrollBar().setValue(100)
    controller.screen.report_widget.verticalScrollBar().setValue(300)
    positions = controller.screen.session_state()["scroll"]
    assert all(positions)
    path = tmp_path / "comparison.mlcb"
    controller.save_experiment(str(path))
    finish(controller, qt)
    notes = load_experiment(path).source["import_notes"]
    controller.open_experiment(str(path))
    finish(controller, qt)
    qt.processEvents()
    assert controller.screen.session_state()["scroll"] == positions
    controller.save_experiment(str(path))
    finish(controller, qt)
    assert load_experiment(path).source["import_notes"] == notes


def test_failed_save_preserves_current_report_and_reenables_actions(controller, qt, tmp_path):
    open_recipe(controller, "linear-regression")
    screen = controller.screen
    before = deepcopy(screen.session_state())
    controller.save_experiment(str(tmp_path / "missing-folder" / "analysis.mlcb"))
    finish(controller, qt)
    assert controller.screen is screen
    assert screen.request() == before["request"]
    assert screen.session_state()["report"] == before["report"]
    assert "Unable to complete" in screen.status.text()
    assert controller.window.save_experiment_button.isEnabled()


def test_movement_study_import_run_export_and_offline_reopen(controller, qt, tmp_path):
    import json

    from mlcookbook.core.results import to_jsonable
    from tests.test_wearable import make_export

    open_recipe(controller, "wearable-study")
    screen = controller.screen
    assert not screen.run_button.isEnabled() and not screen.is_example
    assert not screen._report
    study = make_export(tmp_path / "movement")
    controller.load_wearable_study(str(study / "export.json"))
    finish(controller, qt)
    assert screen.run_button.isEnabled()
    assert screen.source["kind"] == "wearable_study"
    assert controller.imported is None  # Reserved holdouts never become other recipes' exploratory input.
    screen.run_button.click()
    finish(controller, qt)
    result = screen._report["wearable_evaluation"]
    assert "test" not in result["predictions"]
    assert "validation" in result["predictions"]
    assert screen.export_button.isEnabled()
    expected = to_jsonable(screen.session_state())
    exported = tmp_path / "movement-evaluation.json"
    controller.export_wearable_evaluation(str(exported))
    finish(controller, qt)
    assert json.loads(exported.read_text()) == result
    screen.param_widgets["final_test"].setCurrentIndex(1)  # Keep an unrun test choice separate from the report.
    expected = to_jsonable(screen.session_state())
    saved = tmp_path / "movement.mlcb"
    controller.save_experiment(str(saved))
    finish(controller, qt)
    for file in study.iterdir():
        file.unlink()
    controller.open_experiment(str(saved))
    finish(controller, qt)
    restored = controller.screen
    assert restored._report == expected["report"]
    assert restored.request() == expected["request"]
    assert "previous result" in restored.result_state.text()
    assert controller.imported is None
    restored.run_button.click()
    finish(controller, qt)
    assert "test" in restored._report["wearable_evaluation"]["predictions"]
    controller.show_front_page()
    open_recipe(controller, "wearable-study")
    assert "test" in controller.screen._report["wearable_evaluation"]["predictions"]


def test_failed_movement_import_preserves_previous_study_and_result(controller, qt, tmp_path):
    from tests.test_wearable import make_export

    open_recipe(controller, "wearable-study")
    study = make_export(tmp_path / "movement")
    controller.load_wearable_study(str(study))
    finish(controller, qt)
    controller.screen.run_button.click()
    finish(controller, qt)
    original = controller.screen.session_state()
    (study / "train.csv").write_text("damaged", encoding="utf-8")
    controller.load_wearable_study(str(study))
    finish(controller, qt)
    assert "checksum" in controller.screen.status.text()
    assert controller.screen._report == original["report"]
    assert controller.screen.run_button.isEnabled()


def test_ordinary_import_does_not_automatically_fill_movement_recipe(controller, linear_df):
    from mlcookbook import Dataset

    data = Dataset(linear_df)
    controller.imported = (data, "custom", app._dataset_view(data, "custom"))
    open_recipe(controller, "wearable-study")
    assert not controller.screen.run_button.isEnabled()
    assert controller.screen.dataset is not data



def test_startup_study_is_verified_without_training_or_test_evaluation(controller, qt, tmp_path):
    from tests.test_wearable import make_export

    study = make_export(tmp_path / "study with spaces")
    options, qt_arguments = app._startup_options(["--study", str(study / "export.json"), "-platform", "windows"])
    assert qt_arguments == ["-platform", "windows"]
    app._open_startup(controller, options)
    finish(controller, qt)
    assert controller.key == "wearable-study"
    assert controller.screen.run_button.isEnabled()
    assert not controller.screen._report
    assert controller.screen.request()["params"]["final_test"] is False
    assert controller.imported is None
    saved = tmp_path / "study snapshot.mlcb"
    controller.save_experiment(str(saved))
    finish(controller, qt)
    controller.show_front_page()
    options, _ = app._startup_options(["--experiment", str(saved)])
    app._open_startup(controller, options)
    finish(controller, qt)
    assert controller.key == "wearable-study"
    assert controller.screen.source["kind"] == "wearable_study"
    assert not controller.screen._report


def test_startup_failure_and_empty_setup_remain_usable(controller, qt, tmp_path):
    options, _ = app._startup_options(["--movement-study"])
    app._open_startup(controller, options)
    assert controller.key == "wearable-study"
    assert not controller.screen.run_button.isEnabled()
    options, _ = app._startup_options(["--study", str(tmp_path / "missing/export.json")])
    app._open_startup(controller, options)
    finish(controller, qt)
    assert controller.screen.status.text()
    assert controller.screen.load_button.isEnabled()
    assert not controller.screen.run_button.isEnabled()
    with pytest.raises(SystemExit):
        app._startup_options(["--movement-study", "--study", "export.json"])
