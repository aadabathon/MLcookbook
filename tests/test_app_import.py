"""Integration checks for the desktop import and background analysis flow."""
import sys
import time
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


@pytest.mark.parametrize("key", list(app.SCREEN_FACTORIES))
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
    assert "DATA CHECKS" in text and "row identifier" in text
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


@pytest.mark.parametrize("key", list(app.SCREEN_FACTORIES))
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
