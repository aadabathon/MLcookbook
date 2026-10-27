"""Recipe organization retains controls, result provenance and saved state."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def qt():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _dispose(*widgets):
    from PySide6.QtCore import QCoreApplication, QEvent

    for widget in widgets:
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _view():
    return {
        "name": "own measurements.parquet", "rows": 12,
        "columns": [{"name": name, "type": "numeric", "missing": 0, "classes": [], "issues": []}
                    for name in ("input", "response")],
        "preview": [["1", "2"]], "notes": [], "issues": [],
    }


def _screen():
    from gui.screens.linear_regression import LinearRegressionScreen

    screen = LinearRegressionScreen({
        "description": "Analyze the selected measurements.",
        "evaluation": "25% test holdout; split seed 0.",
    }, {})
    screen.resize(1080, 720)
    screen.set_dataset(_view(), target="response")
    return screen


def _report(rows=2):
    return {
        "dataset": "own measurements.parquet", "target": "response", "features": "input",
        "settings": "penalty=none; alpha=1.0",
        "summary": "Mean absolute error on held-out rows: 0.000000123456.",
        "plots": [{"title": "Actual versus predicted", "x_label": "Actual", "y_label": "Predicted",
                   "series": [{"kind": "points", "points": [(1., 1.1), (2., 1.9)]}]}],
        "metric_groups": [
            {"title": "Test set", "items": [("MAE", "0.000000123456")]},
            {"title": "Training set", "items": [("R²", "-0.000012345")]},
            {"title": "Training cross-validation", "items": [("RMSE", "1.23456789e-09 ± 0.00001")]},
        ],
        "tables": [{"title": "Full diagnostics", "headers": ["Identifier", "Estimate"],
                    "rows": [[f"{row:05}", "1.23456789e-09"] for row in range(rows)]}],
        "warnings": ["Review the signed estimate before interpretation."],
    }


@pytest.mark.parametrize("accept", [False, True], ids=["cancel", "accept"])
def test_preprocessing_remains_accessible_when_model_settings_are_closed(qt, accept):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDialog

    from gui.preprocessing_dialog import PreprocessingDialog

    screen = _screen()
    original = _report()
    try:
        screen.show_report(original)
        screen.sections["settings"].setChecked(False)
        screen.show()
        qt.processEvents()
        for widget in (screen.preprocessing_button, screen.preprocessing_summary, screen.evaluation_summary):
            assert widget.isVisibleTo(screen)
        assert not screen.param_widgets["alpha"].isVisibleTo(screen)
        before = deepcopy(screen.request())
        seen = []

        def edit_dialog():
            dialog = qt.activeModalWidget()
            seen.append(isinstance(dialog, PreprocessingDialog))
            if not isinstance(dialog, QDialog):
                return
            if isinstance(dialog, PreprocessingDialog):
                dialog.combos["scaling"].setCurrentIndex(dialog.combos["scaling"].findData("robust"))
                dialog.cv.setCurrentIndex(dialog.cv.findData(3))
            dialog.accept() if accept else dialog.reject()

        QTimer.singleShot(0, edit_dialog)
        screen.preprocessing_button.click()
        assert seen == [True]
        if accept:
            assert screen.request()["preprocessing"]["scaling"] == "robust"
            assert screen.request()["cv_folds"] == 3
            assert "3" in screen.evaluation_summary.text()
            assert "Settings changed" in screen.result_state.text()
        else:
            assert screen.request() == before
            assert "match current settings" in screen.result_state.text()
        assert screen._report == original
        assert screen.session_state()["report_request"] == before
    finally:
        _dispose(screen)


def test_only_stale_results_need_a_banner_while_footer_keeps_input_errors(qt):
    screen = _screen()
    try:
        screen.show_report(_report())
        screen.show()
        qt.processEvents()
        assert not screen.result_state.isVisibleTo(screen)
        assert screen.warning_button.isVisibleTo(screen)
        screen.param_widgets["alpha"].setValue(2.)
        assert "Settings changed" in screen.result_state.text()
        assert screen.result_state.isVisibleTo(screen)
        assert screen.status.text() == ""
        assert screen.run_button.isEnabled()
        screen.columns.cellWidget(1, 1).setCurrentText("text")
        assert screen.status.text() == "Regression needs a numeric target."
        assert not screen.run_button.isEnabled()
        assert "previous result" in screen.result_state.text()
        screen.columns.cellWidget(1, 1).setCurrentText("numeric")
        screen.param_widgets["alpha"].setValue(1.)
        assert "match current settings" in screen.result_state.text()
        assert not screen.result_state.isVisibleTo(screen)
        assert screen.warning_button.isVisibleTo(screen)
        assert screen.status.text() == ""
        assert screen.run_button.isEnabled()
    finally:
        _dispose(screen)


def test_recipe_setup_has_one_input_identity_and_one_empty_result_state(qt):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel

    screen = _screen()
    try:
        screen.show()
        qt.processEvents()
        assert not screen.findChildren(QLabel, "ScreenSubtitle")
        assert not screen.findChildren(QLabel, "DatasetBadge")
        identities = screen.findChildren(QLabel, "InputContext")
        assert identities == [screen.input_context]
        assert screen.input_context.text() == "Input: own measurements.parquet · 12 rows · 2 columns"
        assert screen.input_context.textFormat() == Qt.TextFormat.PlainText
        assert screen.input_context.isVisibleTo(screen)
        assert screen.load_button.isVisibleTo(screen)
        assert not screen.form_panel.isAncestorOf(screen.load_button)
        assert screen.result_state.isVisibleTo(screen)
        assert screen.result_state.text() == "No completed run"
        assert not screen.report_widget.findChildren(QLabel)
        assert "Types describe roles; values are not converted." in screen.columns.toolTip()
        assert screen.columns.accessibleDescription() == screen.columns.toolTip()
        assert screen.columns_label.toolTip() == screen.columns.toolTip()
    finally:
        _dispose(screen)


def test_input_row_import_uses_csv_options_and_locks_with_the_form(qt, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    screen = _screen()
    requests = []
    screen.load_requested.connect(lambda *args: requests.append(args))
    path = "C:/own data/measurements.csv"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (path, ""))
    try:
        screen.show_report(_report())
        screen.sections["import"].setChecked(True)
        screen.delimiter.setCurrentIndex(screen.delimiter.findData(";"))
        screen.encoding.setCurrentIndex(screen.encoding.findData("cp1252"))
        screen.markers.setText("?, -999")
        before = screen.session_state()
        screen.load_button.click()
        assert requests == [(path, ";", "cp1252", "?, -999")]
        screen.set_status("Reading dataset…")
        screen.set_busy(True)
        assert not screen.load_button.isEnabled()
        assert not screen.form_panel.isEnabled()
        assert not screen.run_button.isEnabled()
        screen.load_button.click()
        assert len(requests) == 1
        assert screen.status.text() == "Reading dataset…"
        screen.set_busy(False)
        assert screen.load_button.isEnabled() and screen.form_panel.isEnabled()
        assert screen.run_button.isEnabled()
        after = screen.session_state()
        assert {key: value for key, value in after.items() if key != "status"} == {
            key: value for key, value in before.items() if key != "status"}
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
        screen.load_button.click()
        assert len(requests) == 1
    finally:
        _dispose(screen)


def test_legacy_session_preserves_unrun_controls_status_and_section_keys(qt):
    original, restored = _screen(), _screen()
    try:
        original.show_report(_report())
        original.param_widgets["alpha"].setValue(2.75)
        original.markers.setText("?, -999")
        original.delimiter.setCurrentIndex(original.delimiter.findData(";"))
        original.sections["import"].setChecked(True)
        original.sections["settings"].setChecked(False)
        original.set_status("Settings changed. Run Analysis to update the results.")
        legacy = json.loads(json.dumps(original.session_state()))
        assert set(legacy) == {
            "request", "params", "report", "report_request", "delimiter", "encoding", "markers",
            "status", "sections", "scroll", "is_example",
        }
        assert legacy["sections"] == {"import": True, "settings": False}
        restored.validate_session(legacy)
        restored.restore_session(legacy)
        qt.processEvents()
        actual = json.loads(json.dumps(restored.session_state()))
        assert actual == legacy
        assert "previous result" in restored.result_state.text()
        assert restored.status.text() == legacy["status"]
    finally:
        _dispose(original, restored)


def test_run_details_are_transient_and_keep_full_report_precision_and_table_rows(qt):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QPushButton, QTableWidget

    screen = _screen()
    report = _report(rows=500)
    report["run_details"] = ["Source rows: 00042, 00043", "Runtime identity: <local & explicit>"]
    try:
        screen.show_report(report)
        screen.show()
        qt.processEvents()
        toggle = screen.report_widget.findChild(QPushButton, "RunDetailsToggle")
        assert toggle is not None and not toggle.isChecked()
        labels = screen.report_widget.findChildren(QLabel)
        metadata = next(label for label in labels if "Dataset:" in label.text() and "Features:" in label.text())
        assert not metadata.isVisibleTo(screen.report_widget)
        before = screen.session_state()
        toggle.click()
        assert metadata.isVisibleTo(screen.report_widget)
        assert metadata.textFormat() == Qt.TextFormat.PlainText
        assert all(detail in metadata.text() for detail in report["run_details"])
        after = screen.session_state()
        assert after["sections"] == before["sections"]
        assert set(after) == set(before)
        assert after["report"] == report
        assert [label.text() for label in labels if label.text() in
                ("Test set", "Training set", "Training cross-validation")] == [
                    "Test set", "Training set", "Training cross-validation"]
        texts = [label.text() for label in labels]
        for group in report["metric_groups"]:
            assert all(value in texts for _, value in group["items"])
        table = screen.report_widget.findChild(QTableWidget)
        assert table.rowCount() == 500
        assert table.item(499, 0).text() == "00499"
        assert table.item(499, 1).text() == "1.23456789e-09"
        assert not screen.warning_button.isHidden()
    finally:
        _dispose(screen)


def test_local_llm_backend_visibility_and_saved_output_survive_shared_form_changes(qt):
    import app

    screen = app.SCREEN_FACTORIES["llm"](demo=False, setup=True)
    try:
        screen.show()
        screen.param_widgets["model_path"].setText("C:/models/selected-locally")
        qt.processEvents()
        assert screen.preprocessing is None and screen.cv_folds is None
        assert screen.run_button.isEnabled()
        assert not screen.columns.isVisibleTo(screen)
        assert screen.param_widgets["prompt"].isVisibleTo(screen)
        assert screen.param_widgets["model_path"].isVisibleTo(screen)
        report = {
            "dataset": "Typed prompt", "features": "Typed prompt", "settings": "PyTorch / Transformers",
            "text_sections": [{"title": "Output", "text": "An output kept from the previous run."}],
            "warnings": ["Generated text has not been evaluated for accuracy."],
        }
        screen.show_report(report)
        previous_request = deepcopy(screen.request())
        screen.param_widgets["backend"].setCurrentIndex(screen.param_widgets["backend"].findData("llamacpp"))
        screen.param_widgets["llamacpp_model"].setEditText("local-model")
        assert screen.param_widgets["llamacpp_model"].isVisibleTo(screen)
        assert not screen.param_widgets["device"].isVisibleTo(screen)
        assert not screen.param_widgets["context_tokens"].isVisibleTo(screen)
        assert not screen.param_widgets["model_path"].isVisibleTo(screen)
        assert screen.run_button.isEnabled()
        assert "Settings changed" in screen.result_state.text()
        assert screen.session_state()["report_request"] == previous_request
        assert screen._report == report
        assert screen.status.text() == ""
    finally:
        _dispose(screen)


@pytest.mark.parametrize("backend,input_mode", [
    ("pytorch", "prompt"), ("ollama", "rows"), ("llamacpp", "prompt"),
])
def test_llm_controls_fit_horizontally_at_the_minimum_window_size(qt, backend, input_mode):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QFont

    import app
    from gui import theme
    from gui.main_window import MainWindow

    original_style, original_font = qt.styleSheet(), qt.font()
    qt.setFont(QFont(theme.FONT_FAMILY, theme.BODY_POINT_SIZE))
    qt.setStyleSheet(theme.stylesheet())
    window = MainWindow()
    screen = app.SCREEN_FACTORIES["llm"](demo=False, setup=True)
    try:
        screen.param_widgets["backend"].setCurrentIndex(screen.param_widgets["backend"].findData(backend))
        screen.param_widgets["input_mode"].setCurrentIndex(screen.param_widgets["input_mode"].findData(input_mode))
        window.show_screen(screen)
        window.resize(theme.WINDOW_MIN_WIDTH, theme.WINDOW_MIN_HEIGHT)
        window.show()
        qt.processEvents()
        window.grab()
        qt.processEvents()
        assert window.width() == theme.WINDOW_MIN_WIDTH
        viewport = screen.form_panel.viewport()
        visible = []
        for key, row in screen.param_rows.items():
            for widget in row:
                if not widget.isVisibleTo(screen):
                    continue
                visible.append(key)
                left = widget.mapTo(viewport, QPoint(0, 0)).x()
                assert 0 <= left and left + widget.width() <= viewport.width(), (
                    key, left, widget.width(), viewport.width())
        assert "backend" in visible and "prompt" in visible
        assert ("max_rows" in visible) is (input_mode == "rows")
        for widget in (screen.preview_button, screen.checks_button):
            if widget.isVisibleTo(screen):
                left = widget.mapTo(viewport, QPoint(0, 0)).x()
                assert 0 <= left and left + widget.width() <= viewport.width(), (
                    widget.text(), left, widget.width(), viewport.width())
    finally:
        _dispose(window)
        qt.setStyleSheet(original_style)
        qt.setFont(original_font)


def test_movement_settings_keep_final_test_sealed_and_grouping_explicit(qt):
    import app

    screen = app.SCREEN_FACTORIES["wearable-study"](demo=False, setup=True)
    try:
        screen.show()
        screen.sections["settings"].setChecked(True)
        qt.processEvents()
        assert screen.request()["params"]["final_test"] is False
        assert screen.preprocessing is None and screen.cv_folds is None
        assert not screen.run_button.isEnabled()
        assert "sealed" in screen.param_widgets["final_test"].currentText().lower()
        assert "group" in screen.sections["settings"].text().lower()
        assert screen.param_widgets["cv_folds"].isVisibleTo(screen)
        assert screen.param_widgets["final_test"].isVisibleTo(screen)
        original = screen.session_state()
        screen.param_widgets["random_state"].setValue(73)
        assert screen.request()["params"]["final_test"] is False
        assert screen.session_state()["sections"] == original["sections"]
        assert screen.session_state()["report"] == original["report"]
    finally:
        _dispose(screen)


@pytest.mark.parametrize("key", ["linear-regression", "logistic-regression", "decision-tree", "random-forest",
                                "svm", "neural-network"])
def test_supervised_setup_exposes_the_actual_fixed_evaluation_defaults(qt, key):
    import inspect

    import app

    screen = app.SCREEN_FACTORIES[key](demo=False, setup=True)
    try:
        defaults = inspect.signature(app._screens()[key][0]).parameters
        summary = screen.evaluation_summary.text()
        assert f"{defaults['test_size'].default:.0%}" in summary
        assert f"seed {defaults['random_state'].default}" in summary
        assert f"{defaults['cv_folds'].default} folds" in summary
        assert "training" in summary.lower()
        assert screen._report == {}
    finally:
        _dispose(screen)


def test_comparison_split_and_seed_are_accessible_and_summary_tracks_unrun_edits(qt):
    import app

    screen = app.SCREEN_FACTORIES["model-comparison"](demo=False, setup=True)
    try:
        screen.set_dataset(_view(), target="response")
        screen.sections["settings"].setChecked(False)
        screen.show()
        qt.processEvents()
        assert screen.param_widgets["test_size"].isVisibleTo(screen)
        assert screen.param_widgets["random_state"].isVisibleTo(screen)
        assert not screen.param_widgets["alpha"].isVisibleTo(screen)
        screen.show_report(_report())
        previous = deepcopy(screen.session_state()["report_request"])
        screen.param_widgets["test_size"].setValue(.4)
        screen.param_widgets["random_state"].setValue(17)
        screen.set_preprocessing(screen.preprocessing, 3)
        summary = screen.evaluation_summary.text()
        assert "40%" in summary and "seed 17" in summary and "3 folds" in summary
        assert "select" in summary.lower() and "before" in summary.lower()
        assert "Settings changed" in screen.result_state.text()
        assert screen.session_state()["report_request"] == previous
        assert screen.request()["params"]["test_size"] == .4
        assert screen.request()["params"]["random_state"] == 17
    finally:
        _dispose(screen)


@pytest.mark.parametrize("key", ["kmeans", "pca"])
def test_unsupervised_setup_does_not_claim_a_train_test_split(qt, key):
    import app

    screen = app.SCREEN_FACTORIES[key](demo=False, setup=True)
    try:
        summary = screen.evaluation_summary.text().lower()
        assert "all selected rows" in summary
        assert "no test holdout" in summary
        assert screen.request()["cv_folds"] is None
        assert "training cross-validation:" not in summary
        assert screen._report == {}
    finally:
        _dispose(screen)


def test_unsupervised_preprocessing_dialog_explains_full_data_fitting_without_changing_values(qt):
    from PySide6.QtWidgets import QLabel

    from gui.preprocessing_dialog import PREPROCESSING_OPTIONS, PreprocessingDialog, default_preprocessing

    dialog = PreprocessingDialog(default_preprocessing(), None)
    try:
        help_text = dialog.findChild(QLabel, "HelpText").text()
        assert "all selected rows" in help_text and "no test holdout" in help_text
        assert dialog.cv is None
        assert dialog.values() == (default_preprocessing(), None)
        for key, _, options in PREPROCESSING_OPTIONS:
            combo = dialog.combos[key]
            assert all("training" not in combo.itemText(index) for index in range(combo.count()))
            for index, (_, expected) in enumerate(options):
                actual = combo.itemData(index)
                assert actual == expected and type(actual) is type(expected)
        dialog.combos["min_category_frequency"].setCurrentIndex(3)
        dialog.combos["missing_indicators"].setCurrentIndex(1)
        values, cv = dialog.values()
        assert values["min_category_frequency"] == 10 and type(values["min_category_frequency"]) is int
        assert values["missing_indicators"] is True
        assert cv is None
    finally:
        _dispose(dialog)


def test_empty_llm_setup_is_actionable_without_presenting_example_data_or_generated_output(qt, monkeypatch, tmp_path):
    from functools import wraps

    from PySide6.QtWidgets import QLabel, QPlainTextEdit

    import app
    import mlcookbook

    @wraps(mlcookbook.generate_text)
    def unexpected(*args, **kwargs):
        pytest.fail("Opening or configuring Local LLMs must not load preset rows or run inference")

    monkeypatch.setattr(mlcookbook, "generate_text", unexpected)
    monkeypatch.setattr(app, "_run_report", unexpected)
    _, name, target, positive = app.DEMOS["llm"]
    monkeypatch.setitem(app.DEMOS, "llm", (unexpected, name, target, positive))
    screen = app.SCREEN_FACTORIES["llm"](demo=False, setup=True)
    try:
        screen.show()
        qt.processEvents()
        assert screen.dataset.df.empty and screen.view["rows"] == 0
        assert list(screen.dataset.df.columns) == ["prompt"]
        assert screen.source == {"kind": "empty"}
        assert screen.session_state()["report"] == {}
        assert screen.session_state()["report_request"] is None
        assert "model" in screen.status.text().lower()
        assert "choose" in screen.status.text().lower()
        assert not screen.run_button.isEnabled()
        assert screen.result_state.text() != screen.status.text()
        assert not screen.warning_button.isVisibleTo(screen)
        assert not screen.report_widget.findChildren(QPlainTextEdit)
        assert not screen.report_widget.findChildren(QLabel)
        assert screen.result_state.isVisibleTo(screen)
        assert screen.result_state.text() == "No completed run"
        requests = []
        screen.run_requested.connect(requests.append)
        screen.param_widgets["model_path"].setText(str(tmp_path / "chosen-later-at-run-time"))
        assert screen.run_button.isEnabled()
        assert screen.request()["params"]["input_mode"] == "prompt"
        screen.run_button.click()
        assert requests == [screen.request()]
        assert screen.session_state()["report"] == {}
        assert screen.dataset.df.empty
    finally:
        _dispose(screen)


def test_retired_report_and_disclosure_wrappers_are_released_by_gui_deferred_delete(qt):
    import gc
    import weakref

    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QPushButton

    was_enabled = gc.isenabled()
    screen = _screen()
    try:
        # Automatic Python collection must not be needed to retire Qt widgets:
        # the next allocation may occur in an analysis worker instead of Qt's thread.
        gc.disable()
        screen.show_report(_report())
        for _ in range(3):
            screen.sections["settings"].setChecked(True)
            screen.sections["settings"].setChecked(False)
            old_report = screen.report_widget
            toggle = old_report.findChild(QPushButton, "RunDetailsToggle")
            toggle.click()
            assert toggle.isChecked() and "▾" in toggle.text()
            toggle.click()
            assert not toggle.isChecked() and "▸" in toggle.text()
            report_ref, toggle_ref = weakref.ref(old_report), weakref.ref(toggle)
            del old_report, toggle
            screen.show_report(_report())
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            assert report_ref() is None
            assert toggle_ref() is None
        assert screen.sections["settings"].isChecked() is False
        assert screen._report == _report()
    finally:
        _dispose(screen)
        if was_enabled:
            gc.enable()
