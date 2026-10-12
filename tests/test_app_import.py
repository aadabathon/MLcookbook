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


def test_multiclass_target_disables_run(controller, logistic_df):
    from mlcookbook import Dataset
    open_recipe(controller, "logistic-regression")
    logistic_df.loc[0, "y"] = "third"
    data = Dataset(logistic_df)
    controller._use_import((data, "three classes", app._dataset_view(data, "three classes")))
    controller.screen.target.setCurrentIndex(controller.screen.target.findData("y"))
    assert not controller.screen.run_button.isEnabled()


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
