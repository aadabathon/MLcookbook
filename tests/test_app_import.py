"""Integration checks for the desktop import and background analysis flow."""
import time
from pathlib import Path
import sys

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
    controller.show_front_page()
    open_recipe(controller, key)
    assert controller.screen.dataset_name == path.name
    assert controller.screen.target.currentData() is None


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
    screen.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("key", ["decision-tree", "random-forest"])
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
