"""Research jobs preserve ordinary data, recipe experiments and completed artifacts."""

from __future__ import annotations

import json
import struct
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402
from gui.recipes import RECIPES  # noqa: E402


@pytest.fixture(scope="module")
def qt():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(qt):
    controller = app.MLCookbookApp()
    controller.window.resize(1080, 760)
    controller.window.show()
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


def open_lab(controller, qt):
    controller.open_recipe(next(recipe for recipe in RECIPES if recipe.key == "llm"))
    finish(controller, qt)
    controller.screen.workspace_mode.setCurrentIndex(1)
    qt.processEvents()
    return controller.screen.research


def tiny_gguf(path):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    raw = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
    raw += string("general.architecture") + struct.pack("<I", 8) + string("llama")
    raw += string("blk.0.attn_norm.weight") + struct.pack("<IQIQ", 1, 4, 0, 0)
    raw += b"\0" * (-len(raw) % 32) + struct.pack("<ffff", 1, 2, 3, 4)
    path.write_bytes(raw)
    return path


def graph_capture():
    return {"format": "mlcookbook-llama-graph", "version": 1, "provenance": {"capture_kind": "fixture"},
            "graphs": [{"id": phase, "phase": phase, "input_token_ids": [1], "nodes": [
                {"id": "n0", "name": "input", "op": "NONE", "shape": [count], "strides": [4],
                 "dtype": "f32", "inputs": []},
                {"id": "n1", "name": "norm-0", "op": "RMS_NORM", "shape": [count], "strides": [4],
                 "dtype": "f32", "inputs": ["n0"]}]} for phase, count in (("prefill", 4), ("decode", 1))],
            "warnings": []}


def test_lab_navigation_is_local_and_independent_of_recipe_state(controller, qt, monkeypatch):
    import mlcookbook.llm_lab as core

    monkeypatch.setattr(core, "probe_llamacpp", lambda *a, **kw: pytest.fail("Navigation must not contact a server"))
    panel = open_lab(controller, qt)
    original = controller.screen.session_state()
    panel.capture_prompt.setText("Accelerator tensor shapes")
    panel.repetitions.setValue(7)
    assert controller.imported is None
    assert not controller.window.save_experiment_button.isEnabled()
    # Research configuration does not add keys to the strict .mlcb v1 state.
    assert controller.screen.session_state() == original
    controller.navigate("data")
    assert controller.job is None
    panel = open_lab(controller, qt)
    assert panel.capture_prompt.text() == "Accelerator tensor shapes"
    assert panel.repetitions.value() == 7
    assert controller.screen.workspace_mode.currentData() == "research"


def test_descriptor_job_runs_off_gui_thread_and_lab_reopens_offline(controller, qt, monkeypatch, tmp_path):
    import mlcookbook.llm_lab as core

    panel = open_lab(controller, qt)
    path = tiny_gguf(tmp_path / "tensor.gguf")
    original = core.inspect_gguf
    threads = []
    def inspect(*args, **kwargs):
        threads.append(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setattr(core, "inspect_gguf", inspect)
    controller.llm_lab_action("inspect", {"path": str(path)})
    assert controller.job is not None and not panel.inspect_button.isEnabled()
    finish(controller, qt)
    assert threads and threads[0] != threading.get_ident()
    assert panel.lab_document()["inspection"]["tensors"][0]["shape"] == [4]
    assert controller.imported is None and not controller.screen._report
    saved = tmp_path / "accelerator.mlcl"
    controller.llm_lab_action("save", {"path": str(saved)})
    finish(controller, qt)
    expected = panel.lab_document()
    path.unlink()
    monkeypatch.setattr(core, "inspect_gguf", lambda *a, **kw: pytest.fail("Reopen must not inspect source files"))
    panel.capture_prompt.setText("Unrun changed prompt")
    controller.llm_lab_action("open", {"path": str(saved)})
    finish(controller, qt)
    assert panel.lab_document() == expected
    exported = tmp_path / "accelerator.json"
    controller.llm_lab_action("export", {"path": str(exported)})
    finish(controller, qt)
    assert json.loads(exported.read_text(encoding="utf-8")) == expected
    before = panel.lab_document()
    bad = tmp_path / "bad.mlcl"
    bad.write_bytes(b"bad archive")
    controller.llm_lab_action("open", {"path": str(bad)})
    finish(controller, qt)
    assert panel.lab_document() == before
    assert "Unable to complete" in controller.screen._status_message


def test_lab_benchmark_and_probe_keep_generation_and_active_data(controller, qt, llamacpp_server):
    panel = open_lab(controller, qt)
    recipe_state = controller.screen.session_state()
    panel.set_connection(llamacpp_server.base_url, "")
    controller.llm_lab_action("probe", {"base_url": llamacpp_server.base_url, "model": ""})
    finish(controller, qt)
    assert panel.session_state()["connection"]["model"] == "test-gguf"
    assert not any(path == "/v1/chat/completions" for path, _ in llamacpp_server.requests)
    controller.llm_lab_action("benchmark", {"base_url": llamacpp_server.base_url, "model": "test-gguf",
                                          "prompt": "Matrix multiply", "repetitions": 2, "warmups": 1,
                                          "max_new_tokens": 8, "cache_policy": "uncontrolled"})
    finish(controller, qt)
    result = panel.lab_document()["benchmark"]
    assert result["summary"]["completed_measured_trials"] == 2
    assert len(result["trials"]) == 3 and result["trials"][0]["warmup"]
    assert controller.screen._report == recipe_state["report"] and controller.imported is None
    assert controller.screen.request() == recipe_state["request"]
    assert controller.screen.cancel_button.isHidden() and panel.benchmark_button.isEnabled()


def test_failed_and_cancelled_research_jobs_keep_completed_artifacts(controller, qt, monkeypatch):
    import mlcookbook.llm_lab as core

    panel = open_lab(controller, qt)
    expected = panel.lab_document()
    started = threading.Event()
    def inspect(path, *, stop_requested=None, on_progress=None):
        started.set()
        while not stop_requested():
            time.sleep(.005)
        raise InterruptedError("Research cancelled")
    monkeypatch.setattr(core, "inspect_gguf", inspect)
    controller.llm_lab_action("inspect", {"path": "selected.gguf"})
    assert started.wait(2)
    controller.navigate("data")
    assert controller.key == "llm"  # Navigation stays locked while the worker owns the operation.
    controller.cancel_analysis()
    finish(controller, qt)
    assert panel.lab_document() == expected
    assert "cancelled" in controller.screen._status_message.lower()
    assert panel.inspect_button.isEnabled()
    assert not controller.screen.cancel_button.isVisible()


def test_probe_result_is_rendered_without_calling_gui_in_worker(controller, qt, monkeypatch):
    import mlcookbook.llm_lab as core

    panel = open_lab(controller, qt)
    main_thread = threading.get_ident()
    threads = []
    def probe(*args, **kwargs):
        threads.append(threading.get_ident())
        return SimpleNamespace(to_dict=lambda: {"base_url": "http://127.0.0.1:8080", "observed_utc": "fixture",
            "connection": {"models": []}, "capabilities": {}, "warnings": []})
    monkeypatch.setattr(core, "probe_llamacpp", probe)
    controller.llm_lab_action("probe", {"base_url": "http://127.0.0.1:8080"})
    finish(controller, qt)
    assert threads[0] != main_thread
    assert panel.lab_document()["capabilities"]["observed_utc"] == "fixture"


def test_capture_import_validation_and_phase_comparison(controller, qt, monkeypatch, tmp_path):
    import mlcookbook.llm_graph as graph

    panel = open_lab(controller, qt)
    threads = []
    def capture(*args, **kwargs):
        threads.append(threading.get_ident())
        return graph_capture()
    monkeypatch.setattr(graph, "capture_graph", capture)
    controller.llm_lab_action("capture", {"model_path": "selected.gguf", "helper_path": "", "runtime_dir": "",
                                         "prompt": "matrix", "max_tokens": 1, "max_seconds": 90})
    finish(controller, qt)
    assert threads and threads[0] != threading.get_ident()
    expected = panel.lab_document()["capture"]
    assert expected == graph_capture()
    assert panel.graph_phase.count() == 2
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(expected), encoding="utf-8")
    controller.llm_lab_action("import_graph", {"path": str(path)})
    finish(controller, qt)
    assert panel.lab_document()["capture"] == expected
    controller.llm_lab_action("compare_graph", {"capture": expected, "left": 0, "right": 1})
    finish(controller, qt)
    assert "Unable to complete" not in controller.screen._status_message
    bad = graph_capture()
    bad["graphs"][0]["nodes"][1]["inputs"] = ["not-captured"]
    path.write_text(json.dumps(bad), encoding="utf-8")
    controller.llm_lab_action("import_graph", {"path": str(path)})
    finish(controller, qt)
    assert panel.lab_document()["capture"] == expected
    assert "Unable to complete" in controller.screen._status_message
