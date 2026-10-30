"""Research views render supplied evidence and emit explicit coordinator actions."""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import theme  # noqa: E402 -- the GUI package lives at the repository root.
from gui.llm_lab import LLMLabPanel  # noqa: E402


@pytest.fixture(scope="module")
def qt():
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(theme.stylesheet())
    return application


@pytest.fixture
def panel(qt):
    widget = LLMLabPanel()
    widget.resize(850, 720)
    widget.show()
    qt.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def inspection():
    return {"path": "C:/external/model.gguf", "metadata": {"general.architecture": "llama", "long": "x" * 5000},
            "summary": {"architecture": "llama", "tensor_count": 350},
            "tensors": [{"name": f"blk.{index}.attn.weight", "shape": [64, 64], "dtype": "Q4_K",
                         "elements": 4096, "bytes": 2304, "offset": index * 2304,
                         "file_offset": 1024 + index * 2304, "group": f"Layer {index}"}
                        for index in range(350)], "warnings": ["Selected-file identity only."]}


def capture(count=3):
    nodes = [{"id": str(index), "name": f"blk.0.op.{index}", "op": "NONE" if index == 0 else "MUL_MAT",
              "shape": [16, 8, 1, 1], "dtype": "F32", "strides": [4, 64, 512, 512], "bytes": 512,
              "inputs": [str(index - 1)] if index else []} for index in range(count)]
    return {"format": "mlcookbook-llama-graph", "version": 1,
            "provenance": {"attribution": "Separate helper-loaded execution; not the connected server.",
                           "runtime_commit": "recorded-revision"},
            "graphs": [{"id": "prefill", "phase": "prefill", "nodes": nodes},
                       {"id": "decode", "phase": "decode", "nodes": deepcopy(nodes)}],
            "warnings": ["Logical operations are not measured kernel timings."]}


def benchmark():
    return {"settings": {"cache_policy": "uncontrolled", "seconds_scope": "generation request"},
            "trials": [{"index": 1, "warmup": True, "status": "completed", "seconds": 9., "decode_seconds": 8.},
                       {"index": 2, "warmup": False, "status": "completed", "seconds": 2., "decode_seconds": 1.,
                        "decode_tokens_per_second": 20., "end_to_end_tokens_per_second": 10., "text": "answer"},
                       {"index": 3, "warmup": False, "status": "failed", "error": "timed out"}],
            "summary": {"planned_trials": 3, "completed_measured_trials": 1, "failed_measured_trials": 1,
                        "action_seconds": 12., "aggregates": {
                            "seconds": {"count": 1, "median": 2., "mean": 2., "sd": None, "min": 2., "max": 2.},
                            "decode_tokens_per_second": {"count": 1, "median": 20., "mean": 20.,
                                                         "sd": None, "min": 20., "max": 20.}}},
            "warnings": ["Chunk arrivals do not establish individual-token timings."]}


def test_open_edit_and_restore_emit_no_operations(panel, qt):
    actions = []
    panel.action_requested.connect(lambda *values: actions.append(values))
    panel.model_path.setText("C:/external/model.gguf")
    panel.tabs.setCurrentIndex(1)
    panel.capture_tokens.setValue(3)
    panel.tabs.setCurrentIndex(2)
    panel.server_address.setText("http://127.0.0.1:18080")
    panel.server_model.setText("local-model")
    panel.benchmark_prompt.setPlainText("Measured prompt")
    saved = panel.lab_document()
    panel.restore_document(saved)
    qt.processEvents()
    assert actions == []
    panel.benchmark_button.click()
    assert actions == [("benchmark", {"base_url": "http://127.0.0.1:18080", "model": "local-model",
                                       "prompt": "Measured prompt", "repetitions": 3, "warmups": 1,
                                       "max_new_tokens": 64, "max_seconds": 90., "cache_policy": "uncontrolled"})]
    panel.set_busy(True)
    panel.benchmark_button.click()
    panel._capture_request()
    panel._probe()
    assert len(actions) == 1
    assert not panel.server_address.isEnabled() and not panel.open_button.isEnabled()
    panel.set_busy(False)
    assert panel.server_address.isEnabled() and panel.open_button.isEnabled()


def test_tensor_filter_retains_every_row_and_complete_details(panel, qt):
    supplied = inspection()
    panel.show_inspection(supplied)
    supplied["tensors"].clear()
    assert panel.tensor_model.rowCount() == 350
    assert panel.tensor_model.columns[5] == ("file_offset", "File offset")
    panel.tensor_search.setText("blk.349.")
    assert panel.tensor_proxy.rowCount() == 1
    panel.tensor_table.setCurrentIndex(panel.tensor_proxy.index(0, 0))
    detail = json.loads(panel.tensor_detail.toPlainText())
    assert detail["name"] == "blk.349.attn.weight" and detail["file_offset"] == 805120
    panel.metadata_table.setCurrentIndex(panel.metadata_table.model().index(1, 0))
    assert json.loads(panel.tensor_detail.toPlainText())["value"] == "x" * 5000
    assert panel.tensor_model.rowCount() == 350


def test_capture_file_and_performance_server_are_independent_inputs(panel):
    actions = []
    panel.action_requested.connect(lambda *values: actions.append(values))
    panel.model_path.setText("C:/models/selected.gguf")
    panel.set_connection("http://127.0.0.1:8081", "loaded-alias")
    panel._capture_request()
    panel._benchmark_request()
    assert actions[0][0] == "capture" and actions[0][1]["model_path"] == "C:/models/selected.gguf"
    assert "base_url" not in actions[0][1]
    assert actions[1][0] == "benchmark" and actions[1][1]["model"] == "loaded-alias"
    assert actions[1][1]["base_url"] == "http://127.0.0.1:8081" and "model_path" not in actions[1][1]


def test_model_comparison_emits_a_snapshot_and_preserves_the_open_inventory(panel, monkeypatch, qt):
    from PySide6.QtWidgets import QFileDialog

    actions = []
    panel.action_requested.connect(lambda *values: actions.append(values))
    assert not panel.compare_model_button.isEnabled()
    panel.show_inspection(inspection())
    current = deepcopy(panel._inspection)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("C:/external/second.gguf", ""))
    panel.compare_model_button.click()
    assert actions == [("compare_model", {"path": "C:/external/second.gguf", "baseline": current})]
    assert panel._inspection == current
    result = {"summary": {"changed": 1}, "rows": [{"name": "weight", "change": "dtype",
                                                   "left_dtype": "Q4_K", "right_dtype": "Q8_0"}],
              "metadata_changes": [{"key": "template", "left": "before", "right": "after"}], "warnings": []}
    dialog = panel.show_model_comparison(result)
    try:
        qt.processEvents()
        assert dialog.model.rowCount() == 1 and dialog.metadata_model.rowCount() == 1
        dialog.metadata_table.setCurrentIndex(dialog.metadata_proxy.index(0, 0))
        assert json.loads(dialog.detail.toPlainText())["right"] == "after"
    finally:
        dialog.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_actual_graph_phase_groups_trace_and_keyboard_exploration(panel, qt):
    panel.show_capture(capture())
    panel.tabs.setCurrentIndex(1)
    assert panel.graph_phase.count() == 2
    assert "Separate helper-loaded" in panel.capture_summary.text()
    group = panel.graph_view.items_by_key["Layer 0"]
    group.setSelected(True)
    panel.graph_view.setFocus()
    QTest.keyClick(panel.graph_view, Qt.Key.Key_Return)
    assert panel.graph_group.currentData() == "Layer 0"
    assert set(panel.graph_view.items_by_key) == {"0", "1", "2"}
    panel._select_node("1")
    detail = json.loads(panel.graph_detail.toPlainText())
    assert detail["recorded_inputs"] == [{"id": "0", "name": "blk.0.op.0"}]
    assert detail["recorded_consumers"] == [{"id": "2", "name": "blk.0.op.2"}]
    assert detail["strides"] == [4, 64, 512, 512]
    assert detail["backend"] == "Unavailable in this capture"
    panel.trace_upstream.click()
    assert {node["id"] for node in panel._filtered_nodes} == {"0", "1"}
    panel.trace_clear.click()
    panel._select_node("1")
    panel.trace_downstream.click()
    assert {node["id"] for node in panel._filtered_nodes} == {"1", "2"}
    panel.graph_phase.setCurrentIndex(1)
    assert panel._trace_ids is None and len(panel._graph_nodes) == 3


def test_large_graph_pages_preserve_recorded_inputs(panel):
    panel.show_capture(capture(260))
    panel.graph_group.setCurrentIndex(panel.graph_group.findData("Layer 0"))
    assert len(panel.graph_view.items_by_key) == 120 and panel.node_model.rowCount() == 260
    panel.next_button.click()
    assert "120" in panel.graph_view.items_by_key and "0" not in panel.graph_view.items_by_key
    assert "1 inputs outside" in panel.graph_coverage.text()
    panel._select_node("120")
    assert json.loads(panel.graph_detail.toPlainText())["recorded_inputs"][0]["id"] == "119"
    panel.next_button.click()
    assert len(panel.graph_view.items_by_key) == 20 and not panel.next_button.isEnabled()
    assert panel.node_model.rowCount() == 260


def test_graph_alias_edges_are_traversable_and_distinct_from_operation_inputs(panel):
    supplied = capture()
    supplied["graphs"][0]["nodes"][2].update(inputs=[], view_source="1", op="VIEW")
    panel.show_capture(supplied)
    panel._select_node("2")
    detail = json.loads(panel.graph_detail.toPlainText())
    assert detail["recorded_inputs"] == []
    assert detail["recorded_alias_source"] == {"id": "1", "name": "blk.0.op.1"}
    panel.trace_upstream.click()
    assert {node["id"] for node in panel._filtered_nodes} == {"0", "1", "2"}
    panel.trace_clear.click()
    panel._select_node("1")
    assert json.loads(panel.graph_detail.toPlainText())["recorded_consumers"] == [{"id": "2", "name": "blk.0.op.2"}]


def test_phase_comparison_is_explicit_and_keeps_every_supplied_difference(panel, qt):
    actions = []
    panel.action_requested.connect(lambda *values: actions.append(values))
    assert not panel.compare_button.isEnabled()
    supplied = capture()
    panel.show_capture(supplied)
    panel.compare_button.click()
    assert actions == [("compare_graph", {"capture": supplied, "left": 0, "right": 1})]
    supplied["graphs"].clear()
    assert len(actions[0][1]["capture"]["graphs"]) == 2
    result = {"summary": "One tensor changed shape", "warnings": ["Duplicate names were left unmatched."],
              "rows": [{"name": str(index), "change": "shape", "left_shape": [4, 8], "right_shape": [4, 1]}
                       for index in range(240)]}
    dialog = panel.show_graph_comparison(result)
    try:
        qt.processEvents()
        assert dialog.model.rowCount() == 240
        dialog.table.setCurrentIndex(dialog.proxy.index(239, 0))
        assert json.loads(dialog.detail.toPlainText())["name"] == "239"
    finally:
        dialog.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_performance_aggregates_and_plots_exclude_warmups_and_failures(panel):
    panel.show_benchmark(benchmark())
    assert panel.trial_model.rowCount() == 3
    assert panel.summary_model.rowCount() == 2
    assert panel.summary_model.rows[0]["metric"] == "seconds"
    assert panel.summary_model.rows[1]["median"] == 20.
    assert [row["index"] for row in panel.trial_plot.rows] == [2]
    assert [row["index"] for row in panel.throughput_plot.rows] == [2]
    assert "failed measured trials: 1" in panel.benchmark_summary.text()
    panel.trial_table.setCurrentIndex(panel.trial_table.model().index(2, 0))
    assert json.loads(panel.trial_detail.toPlainText())["error"] == "timed out"
    assert "Chunk arrivals" in panel._warnings()[0]


def test_document_roundtrip_preserves_artifacts_selection_and_trace(panel, qt):
    panel.show_inspection(inspection())
    panel.show_capture(capture(260))
    panel.show_benchmark(benchmark())
    panel.tensor_search.setText("blk.349.")
    panel.tensor_table.setCurrentIndex(panel.tensor_proxy.index(0, 0))
    panel.trial_table.setCurrentIndex(panel.trial_table.model().index(1, 0))
    panel.graph_phase.setCurrentIndex(1)
    panel._select_node("120")
    panel.trace_upstream.click()
    panel._graph_page = 1
    panel._render_graph()
    panel.tabs.setCurrentIndex(1)
    panel.graph_view.resetTransform()
    panel.graph_view.scale(.5, .5)
    saved = panel.lab_document()
    clone = LLMLabPanel()
    actions, changed = [], []
    clone.action_requested.connect(lambda *values: actions.append(values))
    clone.configuration_changed.connect(lambda: changed.append(True))
    try:
        clone.resize(850, 720)
        clone.show()
        qt.processEvents()
        clone.restore_document(saved)
        restored = clone.lab_document()
        assert restored["inspection"] == saved["inspection"] and restored["capture"] == saved["capture"]
        assert restored["benchmark"] == saved["benchmark"]
        assert restored["state"]["graph_trace"] == saved["state"]["graph_trace"]
        assert restored["state"]["graph_selected"] == "120" and clone._graph_page == 1
        assert restored["state"]["graph_view"]["scale"] == .5
        assert json.loads(clone.tensor_detail.toPlainText())["name"] == "blk.349.attn.weight"
        assert json.loads(clone.trial_detail.toPlainText())["text"] == "answer"
        assert actions == changed == []
    finally:
        clone.close()
        clone.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_model_only_document_restores_without_creating_a_graph_selection(panel):
    panel.show_inspection(inspection())
    before = panel.lab_document()
    assert before["state"]["graph_group"] is None
    panel.restore_document(before)
    assert panel.lab_document() == before
    assert panel._capture is None and panel.graph_group.count() == panel.graph_phase.count() == 0


def test_saved_comparisons_restore_without_opening_a_dialog(panel, qt):
    panel.show_capture(capture())
    panel.tabs.setCurrentIndex(1)
    result = {"summary": "One changed tensor", "rows": [{"name": "weight", "change": "shape"}], "warnings": []}
    first = panel.show_graph_comparison(result)
    second = panel.show_model_comparison({**result, "metadata_changes": []})
    first.close()
    second.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    saved = panel.lab_document()
    from PySide6.QtWidgets import QDialog

    assert panel.findChildren(QDialog) == []
    panel.restore_document(saved)
    assert panel.lab_document()["graph_comparison"] == result
    assert panel.findChildren(QDialog) == []
    assert panel.view_graph_comparison_button.isVisibleTo(panel)
    dialog = panel._view_graph_comparison()
    try:
        assert dialog.model.rowCount() == 1
    finally:
        dialog.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_table_selection_focuses_a_readable_node_and_supports_zoom_buttons(panel, qt):
    panel.show_capture(capture(260))
    panel.tabs.setCurrentIndex(1)
    panel.node_table.setCurrentIndex(panel.node_proxy.index(130, 0))
    assert panel._graph_page == 1 and panel._selected_node == "130"
    assert panel.graph_view.transform().m11() >= .65
    before = panel.graph_view.transform().m11()
    panel.zoom_in_button.click()
    assert panel.graph_view.transform().m11() > before
    panel.zoom_out_button.click()
    assert panel.graph_view.transform().m11() == pytest.approx(before)


@pytest.mark.parametrize("change", [
    lambda doc: doc["state"].update(repetitions=1000),
    lambda doc: doc["state"].update(graph_view={"scale": float("nan"), "center": [0, 0]}),
    lambda doc: doc.update(capture={"graphs": [{"nodes": [{"name": "missing id"}]}]}),
    lambda doc: doc.update(inspection={"metadata": []}),
    lambda doc: doc.update(benchmark={"summary": {"aggregates": []}}),
    lambda doc: doc.update(capabilities={"warnings": "invalid warning list"}),
])
def test_invalid_restore_preserves_the_current_completed_session(panel, change):
    panel.show_inspection(inspection())
    panel.show_capture(capture())
    saved = panel.lab_document()
    invalid = deepcopy(saved)
    change(invalid)
    with pytest.raises(ValueError):
        panel.restore_document(invalid)
    assert panel.lab_document() == saved


@pytest.mark.skipif(os.environ.get("QT_QPA_PLATFORM") == "offscreen",
                    reason="Exact native font/layout bounds are checked on Windows at 100/125/150%; offscreen fonts differ.")
def test_native_narrow_views_contain_tables_plots_and_graph(panel, qt):
    panel.show_inspection(inspection())
    panel.show_capture(capture())
    panel.show_benchmark(benchmark())
    panel.resize(850, 500)
    for index in range(panel.tabs.count()):
        panel.tabs.setCurrentIndex(index)
        qt.processEvents()
        assert panel.width() == 850
        assert panel.height() <= 500
        page = panel.tabs.widget(index)
        for widget in (panel.tensor_table, panel.metadata_table, panel.graph_view, panel.graph_detail,
                       panel.trial_plot, panel.throughput_plot, panel.trial_table, panel.summary_table):
            if widget.isVisibleTo(page):
                left = widget.mapTo(page, widget.rect().topLeft()).x()
                right = widget.mapTo(page, widget.rect().topRight()).x()
                assert left >= 0 and right < page.width(), (index, type(widget).__name__, left, right, page.width())
        assert not panel.grab().isNull()


def test_server_controls_only_take_space_in_performance(panel):
    for index in (0, 1):
        panel.tabs.setCurrentIndex(index)
        assert panel.server_controls.isHidden() and panel.probe_button.isHidden()
    panel.tabs.setCurrentIndex(2)
    assert not panel.server_controls.isHidden() and not panel.probe_button.isHidden()
    panel.show_status("Running trial")
    assert not panel.status.isHidden()
    panel.show_status("")
    assert panel.status.isHidden()


def test_unrun_research_edits_mark_records_without_replacing_them(panel):
    panel.show_inspection(inspection())
    before = panel.lab_document()["inspection"]
    panel.model_path.setText("another.gguf")
    assert "File selection changed" in panel.inspection_summary.text()
    assert panel.lab_document()["inspection"] == before
    result = benchmark()
    result["settings"] = {"prompt": panel.benchmark_prompt.toPlainText(), "repetitions": panel.repetitions.value(),
                          "warmups": panel.warmups.value(), "max_new_tokens": panel.benchmark_tokens.value(),
                          "max_seconds": panel.benchmark_seconds.value(), "cache_policy": panel.cache_policy.currentData()}
    result.update(panel.session_state()["connection"])
    panel.show_benchmark(result)
    assert "Controls changed" not in panel.benchmark_summary.text()
    panel.repetitions.setValue(panel.repetitions.value() + 1)
    assert "Controls changed" in panel.benchmark_summary.text()
    assert panel.lab_document()["benchmark"] == result


def test_opening_an_empty_lab_clears_previous_artifact_labels(panel):
    empty = panel.lab_document()
    panel.show_inspection(inspection())
    panel.show_capture(capture())
    panel.show_benchmark(benchmark())
    panel.restore_document(empty)
    assert panel.lab_document() == empty
    assert panel.inspection_summary.text() == "No model inspection"
    assert panel.benchmark_summary.text() == "No completed performance trials"
    assert panel.tensor_detail.toPlainText() == "" and panel.trial_detail.toPlainText() == ""
