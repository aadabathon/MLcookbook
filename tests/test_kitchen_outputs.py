"""Completed Kitchen outputs are inspectable, reusable and exported atomically."""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


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
    assert controller.job is None
    qt.processEvents()


def prepared(controller, qt):
    from mlcookbook.core.dataset import Dataset
    from mlcookbook.core.fingerprint import dataset_fingerprint
    from mlcookbook.kitchen import KitchenDocument, KitchenEdge, KitchenGraph, KitchenNode, default_settings

    frame = pd.DataFrame({"x": pd.Series([1., 2., None, 4.], dtype="Float64"),
                          "label": pd.Series(["a", "b", "c", "d"], dtype="string")})
    frame.index = [7, 7, 3, 2]
    data = Dataset(frame)
    controller._use_import((data, "own.parquet", app._dataset_view(data, "own.parquet")))
    controller.navigate("kitchen")
    finish(controller, qt)
    graph = KitchenGraph([
        KitchenNode("a", "dataset", {"snapshot": "own"}),
        KitchenNode("x", "column", {"column": "x"}),
        KitchenNode("m", "map", default_settings("map") | {"operation": "multiply", "value": 2.}),
        KitchenNode("b", "build", {"name": "double"}), KitchenNode("s", "summary", {}),
    ], [KitchenEdge("a", "data", "x", "data"), KitchenEdge("x", "vector", "m", "vector"),
        KitchenEdge("m", "vector", "b", "vector"), KitchenEdge("m", "vector", "s", "vector")])
    controller.kitchen = KitchenDocument(graph, {"own": data}, {"own": {"name": "own.parquet", "source": {}}}, selected="b")
    controller.kitchen_hashes = {"own": dataset_fingerprint(data)}
    controller.kitchen_views = {"own": app._dataset_view(data, "own.parquet")}
    controller.show_kitchen(remember=False)
    controller.run_kitchen()
    finish(controller, qt)
    assert controller.kitchen.run.attempts["b"]["state"] == "completed"
    return data


def test_exports_keep_complete_values_and_analysis_provenance(controller, qt, tmp_path):
    data = prepared(controller, qt)
    parquet, csv, result = tmp_path / "vector.parquet", tmp_path / "values.csv", tmp_path / "summary.json"
    controller.export_kitchen_output("x", "vector", str(parquet))
    finish(controller, qt)
    pd.testing.assert_frame_equal(pd.read_parquet(parquet), data.df[["x"]])
    controller.export_kitchen_output("b", "data", str(csv))
    finish(controller, qt)
    values = pd.read_csv(csv)
    assert values.columns.tolist() == ["double"]
    assert values["double"].dropna().tolist() == [2., 4., 8.]
    controller.export_kitchen_output("s", "result", str(result))
    finish(controller, qt)
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["current_settings_match"] is True
    assert payload["completed"]["payload"]["statistics"]["mean"] == pytest.approx(14 / 3)
    assert payload["completed"]["inputs"]["vector"]["port"] == "vector"


def test_stale_export_uses_old_values_without_execution_and_active_data_requires_current_output(controller, qt, tmp_path, monkeypatch):
    import mlcookbook.kitchen as kitchen

    prepared(controller, qt)
    doc, original = controller.kitchen, controller.imported
    old = doc.run.records["b"]
    doc.graph = replace(doc.graph, nodes=[replace(n, settings=n.settings | {"value": 3.}) if n.id == "m" else n
                                         for n in doc.graph.nodes])
    controller.show_kitchen(remember=False)
    monkeypatch.setattr(kitchen, "map_vector", lambda *a, **kw: pytest.fail("Inspection/export must not execute"))
    controller.accept_kitchen_dataset("b", "data")
    assert controller.job is None and controller.imported is original
    assert "older settings" in controller.screen.status.text()
    output = tmp_path / "old.parquet"
    controller.export_kitchen_output("b", "data", str(output))
    finish(controller, qt)
    assert pd.read_parquet(output)["double"].dropna().tolist() == [2., 4., 8.]
    assert "current settings differ" in controller.screen.status.text()
    assert doc.run.records["b"] == old


def test_accepting_current_dataset_preserves_kitchen_source_and_records_data_provenance(controller, qt):
    original = prepared(controller, qt)
    doc = controller.kitchen
    controller.accept_kitchen_dataset("b", "data")
    finish(controller, qt)
    data, _, view = controller.imported
    assert controller.kitchen is doc and doc.snapshots["own"] is original
    assert data.df["double"].dropna().tolist() == [2., 4., 8.]
    assert data.df.index.tolist() == [7, 7, 3, 2]
    assert view["source"]["kind"] == "kitchen_output"
    controller.navigate("data")
    assert "Kitchen output: b · data" in controller.screen.source.text()
    assert "LLM" not in controller.screen.source.text()
    controller.navigate("kitchen")
    assert controller.kitchen is doc


def test_export_failure_preserves_previous_file_and_completed_output(controller, qt, tmp_path, monkeypatch):
    prepared(controller, qt)
    output = tmp_path / "keep.parquet"
    output.write_bytes(b"previous contents")
    record = controller.kitchen.run.records["b"]

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    controller.export_kitchen_output("b", "data", str(output))
    finish(controller, qt)
    assert output.read_bytes() == b"previous contents"
    assert "simulated disk failure" in controller.screen.status.text()
    assert controller.kitchen.run.records["b"] == record
    assert not list(tmp_path.glob(".keep.parquet.*.tmp"))


def test_linear_saved_data_formatter_accepts_unavailable_diagnostics_without_fitting(monkeypatch):
    import mlcookbook.regression.linear as linear
    from mlcookbook.core.dataset import Dataset

    data = Dataset(pd.DataFrame({"x": range(20), "y": [3 * i + i % 2 for i in range(20)]}))
    result = linear.linear_regression(data, "y", ["x"], cv_folds=2).to_dict()
    result["input_rows"] = len(data.df)
    result["residual_diagnostics"]["normality_p_value"] = None
    monkeypatch.setattr(linear, "linear_regression", lambda *a, **kw: pytest.fail("Formatting must not fit"))
    report = app._kitchen_linear_report(result, "offline.parquet")
    assert report["plots"][0]["title"] == "Actual vs predicted · held-out rows"
    assert report["tables"][-1]["title"].startswith("Preprocessing")
    assert ("Normality p", "—") in report["metric_groups"][-1]["items"]
    json.dumps(report, allow_nan=False)


def test_complete_linear_report_fits_a_narrow_inspector_after_resizing(qt):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QLabel, QWidget

    from gui import theme
    from gui.plot import PlotWidget
    from gui.report_view import ReportView
    from mlcookbook.core.dataset import Dataset
    from mlcookbook.regression.linear import linear_regression

    data = Dataset(pd.DataFrame({"x": range(20), "y": [3 * i + i % 2 for i in range(20)]}))
    result = linear_regression(data, "y", ["x"], cv_folds=2).to_dict()
    result["input_rows"] = len(data.df)
    report = app._kitchen_linear_report(result, "own.parquet")
    previous = qt.styleSheet()
    qt.setStyleSheet(theme.stylesheet())
    view = ReportView(report)
    try:
        view.show()
        for width in (620, 330, 420, 330):
            view.resize(width, 640)
            qt.processEvents()
            viewport = view.viewport()
            assert view.widget().width() <= viewport.width()
            for child in view.widget().findChildren(QWidget):
                if child.parentWidget() is view.widget() and child.isVisibleTo(view):
                    left = child.mapTo(viewport, QPoint(0, 0)).x()
                    assert 0 <= left and left + child.width() <= viewport.width()
            assert view.findChildren(PlotWidget)
            assert view.findChild(QLabel, "ResultSummary").text() == report["summary"]
            metric_values = [label.text() for label in view.findChildren(QLabel, "MetricValue")]
            assert len(metric_values) == sum(len(group["items"]) for group in report["metric_groups"])
    finally:
        view.close()
        view.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qt.setStyleSheet(previous)
