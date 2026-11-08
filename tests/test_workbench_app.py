"""Dataset-first SQLite handoffs, completed-run Python and source isolation."""

import json
import sqlite3
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402 -- the coordinator is the intended UI/core boundary.
from gui.recipes import RECIPES  # noqa: E402


@pytest.fixture(scope="module")
def qt():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(qt, tmp_path, monkeypatch):
    monkeypatch.setattr(app.QStandardPaths, "writableLocation", lambda *_: str(tmp_path / "appdata"))
    controller = app.MLCookbookApp()
    controller.window.resize(1080, 760)
    controller.window.show()
    qt.processEvents()
    yield controller
    finish(controller, qt)
    for dialog in (controller.sqlite_dialog, controller.python_dialog):
        if dialog is not None:
            dialog.close()
    controller.close_programming()
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


def sql_data(tmp_path):
    path = tmp_path / "measurements.sqlite"
    x = np.linspace(0.1, 8, 80)
    z = np.sin(x)
    frame = pd.DataFrame({"row_id": range(80), "x": x, "z": z,
                          "y": 4 * x + 2 * z, "specimen_code": [f"{i:05d}" for i in range(80)]})
    # SQLite ingestion deliberately uses nullable storage for SQL NULL support.
    # Assert that exact contract through offline save/reopen, not just values.
    frame = frame.astype({"row_id": "Int64", "x": "Float64", "z": "Float64",
                          "y": "Float64", "specimen_code": "string"})
    with sqlite3.connect(path) as connection:
        frame.to_sql("observations", connection, index=False)
        connection.execute('CREATE VIEW measurements AS SELECT * FROM observations')
    return path, frame


def import_sql(controller, qt, path):
    controller.open_sqlite()
    dialog = controller.sqlite_dialog
    dialog.path.setText(str(path))
    dialog.catalog_button.click()
    finish(controller, qt)
    assert dialog.objects.count() == 2
    query = 'SELECT * FROM "measurements" ORDER BY "row_id"'
    dialog.query.setPlainText(query)
    dialog.preview_button.click()
    finish(controller, qt)
    assert dialog.import_button.isEnabled()
    dialog.import_button.click()
    finish(controller, qt)
    assert controller.imported is not None
    return controller.imported, query


def open_regression_from_data(controller, qt):
    finish(controller, qt)
    page = controller.screen
    page.workflow_intent.setCurrentIndex(page.workflow_intent.findData("regression"))
    finish(controller, qt)
    page.workflow_target.setCurrentIndex(page.workflow_target.findData("y"))
    finish(controller, qt)
    assert page.workflow_open.isEnabled(), page.workflow_reason.text()
    page.workflow_open.click()
    finish(controller, qt)
    assert controller.key == "linear-regression"
    assert controller.screen.request()["features"] == ["x", "z"]
    assert controller.screen.request()["target"] == "y"
    return controller.screen


def open_recipe(controller, qt, key):
    controller.open_recipe(next(recipe for recipe in RECIPES if recipe.key == key))
    finish(controller, qt)
    return controller.screen


def test_sqlite_preview_handoff_run_save_and_offline_reopen(controller, qt, tmp_path, monkeypatch):
    path, expected = sql_data(tmp_path)
    assert controller.imported is None and controller.key == "data"
    ordinary, query = import_sql(controller, qt, path)
    assert controller.screen.view["rows"] == 80
    pd.testing.assert_frame_equal(ordinary[0].df, expected)
    assert ordinary[2]["source"]["kind"] == "sqlite"
    assert ordinary[2]["source"]["query"] == query
    assert ordinary[0].df.specimen_code.iloc[0] == "00000"
    recipe = open_regression_from_data(controller, qt)
    assert recipe.dataset is ordinary[0] and not recipe.is_example
    assert not recipe._report
    recipe.run_button.click()
    finish(controller, qt)
    assert recipe._report and recipe._report_request["target"] == "y"
    report = json.loads(json.dumps(recipe._report))
    request = deepcopy(recipe._report_request)
    saved = tmp_path / "sql.mlcb"
    controller.save_experiment(str(saved))
    finish(controller, qt)
    assert saved.exists()
    path.unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("Offline reopening must not query SQLite or fit a model.")

    import mlcookbook.sqlite_io as sqlite_io

    monkeypatch.setattr(sqlite_io, "load_sqlite_dataset", forbidden)
    monkeypatch.setattr(app, "_run_report", forbidden)
    controller.open_experiment(str(saved))
    finish(controller, qt)
    assert controller.screen._report == report
    assert controller.screen._report_request == request
    assert controller.screen.source["kind"] == "sqlite"
    assert controller.screen.source["query"] == query
    pd.testing.assert_frame_equal(controller.screen.dataset.df, expected)
    controller.navigate("data")
    finish(controller, qt)
    assert "SQLite" in controller.screen.source.text()
    assert controller.screen.view["rows"] == 80


def test_completed_run_python_uses_recorded_settings_after_edits_and_executes_without_database(controller, qt, tmp_path):
    path, _ = sql_data(tmp_path)
    import_sql(controller, qt, path)
    recipe = open_regression_from_data(controller, qt)
    penalty = recipe.param_widgets["penalty"]
    penalty.setCurrentIndex(penalty.findData("ridge"))
    recipe.param_widgets["alpha"].setValue(2.5)
    recipe.run_button.click()
    finish(controller, qt)
    completed = deepcopy(recipe._report_request)
    assert completed["params"]["alpha"] == 2.5
    recipe.param_widgets["alpha"].setValue(9)
    assert recipe.request()["params"]["alpha"] == 9
    controller.show_python()
    finish(controller, qt)
    assert controller.python_dialog is not None
    assert controller.python_context["request"] == completed
    controller.export_python(str(tmp_path))
    finish(controller, qt)
    exports = list(tmp_path.glob("linear-regression-*"))
    assert len(exports) == 1
    metadata = json.loads((exports[0] / "run.json").read_text(encoding="utf-8"))
    assert metadata["request"] == completed
    path.unlink()
    result = subprocess.run([sys.executable, str(exports[0] / "run.py")], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30, check=True)
    reproduced = json.loads(result.stdout)
    assert reproduced["alpha"] == 2.5
    assert reproduced["features"] == ["x", "z"]
    from mlcookbook import linear_regression
    from mlcookbook.preprocessing.config import PreprocessingConfig

    recorded = linear_regression(controller.python_context["dataset"], target="y", features=["x", "z"],
                                 penalty="ridge", alpha=2.5, cv_folds=completed["cv_folds"],
                                 preprocessing=PreprocessingConfig(**completed["preprocessing"]))
    assert reproduced == recorded.to_dict()


def test_data_workflow_state_survives_navigation_but_new_import_clears_it(controller, qt, tmp_path):
    path, frame = sql_data(tmp_path)
    import_sql(controller, qt, path)
    page = controller.screen
    page.workflow_intent.setCurrentIndex(page.workflow_intent.findData("regression"))
    finish(controller, qt)
    page.workflow_target.setCurrentIndex(page.workflow_target.findData("y"))
    finish(controller, qt)
    z_row = next(row for row in range(page.workflow_columns.rowCount()) if page.workflow_columns.item(row, 1).text() == "z")
    page.workflow_columns.item(z_row, 0).setCheckState(Qt.CheckState.Unchecked)
    finish(controller, qt)
    page.workflow_columns_toggle.setChecked(True)
    expected = page.workflow_state()
    controller.navigate("recipes")
    controller.navigate("data")
    finish(controller, qt)
    assert controller.screen.workflow_state() == expected
    assert controller.screen.workflow_context()["features"] == ["x"]
    assert controller.screen.workflow_open.isEnabled()
    replacement = tmp_path / "replacement.parquet"
    frame.to_parquet(replacement, index=False)
    controller.load_file(str(replacement), ",", "utf-8", "")
    finish(controller, qt)
    assert controller.data_workflow_state == {}
    assert controller.screen.workflow_context()["target"] is None
    assert controller.screen.workflow_context()["features"] == ["x", "z", "y"]


def test_data_handoff_uses_ordinary_snapshot_after_cached_example_and_study(controller, qt, tmp_path):
    path, _ = sql_data(tmp_path)
    ordinary, _ = import_sql(controller, qt, path)
    recipe = open_regression_from_data(controller, qt)
    controller.run_example()
    finish(controller, qt)
    assert controller.screen.is_example and controller.imported is ordinary
    example_dataset = controller.screen.dataset
    example_report = deepcopy(controller.screen._report)
    controller.navigate("data")
    finish(controller, qt)
    recipe = open_regression_from_data(controller, qt)
    assert recipe.dataset is ordinary[0] and not recipe.is_example
    assert controller.example_sessions["linear-regression"]["dataset"] is example_dataset
    assert controller.example_sessions["linear-regression"]["state"]["report"] == example_report
    from tests.test_wearable import make_export

    study_path = make_export(tmp_path / "study")
    open_recipe(controller, qt, "wearable-study")
    controller.load_wearable_study(str(study_path / "export.json"))
    finish(controller, qt)
    study_dataset = controller.screen.dataset
    assert controller.imported is ordinary
    assert not controller.screen.request()["params"]["final_test"]
    controller.navigate("data")
    finish(controller, qt)
    assert controller.screen.view["rows"] == 80
    assert controller.imported[0] is not study_dataset
    recipe = open_regression_from_data(controller, qt)
    assert recipe.dataset is ordinary[0]
    assert "subject_id" not in recipe.dataset.df.columns


def test_sqlite_failure_and_declared_dependence_preserve_ordinary_work(controller, qt, tmp_path):
    path, _ = sql_data(tmp_path)
    ordinary, _ = import_sql(controller, qt, path)
    controller.open_sqlite()
    dialog = controller.sqlite_dialog
    dialog.path.setText(str(path))
    dialog.query.setPlainText("SELECT missing_column FROM observations")
    dialog.preview_button.click()
    finish(controller, qt)
    assert controller.imported is ordinary and not dialog.import_button.isEnabled()
    assert "missing_column" in dialog.status.text()
    assert controller.activity.view()[-1]["outcome"] == "Failed"
    dialog.close()
    qt.processEvents()
    page = controller.screen
    page.workflow_intent.setCurrentIndex(page.workflow_intent.findData("regression"))
    finish(controller, qt)
    page.workflow_target.setCurrentIndex(page.workflow_target.findData("y"))
    finish(controller, qt)
    page.workflow_group.setCurrentIndex(page.workflow_group.findData("row_id"))
    finish(controller, qt)
    assert not page.workflow_open.isEnabled()
    assert "grouped or temporal evaluator" in page.workflow_reason.text()
    controller.open_dataset_workflow("linear-regression", {"task": "regression", **page.workflow_context()})
    assert controller.key == "data" and controller.imported is ordinary
