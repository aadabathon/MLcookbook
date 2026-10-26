"""Interaction and report-data contracts retained by the shared visual refresh."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ready_recipes():
    from gui.recipes import RECIPES

    return [recipe for recipe in RECIPES if recipe.variant == "ready"]


@pytest.fixture(scope="module")
def qt():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def styled_qt(qt):
    from PySide6.QtGui import QFont

    from gui import theme

    original_style, original_font = qt.styleSheet(), qt.font()
    qt.setFont(QFont("Segoe UI", theme.BODY_POINT_SIZE))
    qt.setStyleSheet(theme.stylesheet())
    yield qt
    qt.setStyleSheet(original_style)
    qt.setFont(original_font)
    _dispose()


def _dispose(*widgets):
    from PySide6.QtCore import QCoreApplication, QEvent

    for widget in widgets:
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _visible_labels(widget):
    from PySide6.QtWidgets import QLabel

    return [label.text() for label in widget.findChildren(QLabel) if label.isVisibleTo(widget)]


@pytest.mark.parametrize("recipe", _ready_recipes(), ids=lambda recipe: recipe.key)
def test_ready_cards_keep_accessible_mouse_and_keyboard_actions(styled_qt, recipe):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from gui.recipe_card import RecipeCard

    card = RecipeCard(recipe)
    chosen = []
    card.clicked.connect(chosen.append)
    try:
        card.resize(380, 240)
        card.show()
        styled_qt.processEvents()
        labels = _visible_labels(card)
        assert recipe.title in labels and recipe.description in labels
        assert recipe.status not in labels
        assert recipe.icon not in labels
        assert card.accessibleName() == recipe.title
        assert card.accessibleDescription() == recipe.description
        assert card.focusPolicy() == Qt.FocusPolicy.StrongFocus
        QTest.mouseClick(card, Qt.MouseButton.LeftButton, pos=card.rect().center())
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            QTest.keyClick(card, key)
        assert chosen == [recipe] * 4
    finally:
        _dispose(card)


@pytest.mark.parametrize("variant,status", [("planned", "Planned"), ("soon", "Coming soon")])
def test_unavailable_cards_keep_status_without_becoming_actions(styled_qt, variant, status):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from gui.recipe_card import RecipeCard
    from gui.recipes import Recipe

    recipe = Recipe("Future method", "A method that cannot run yet.", status=status, variant=variant)
    card = RecipeCard(recipe)
    chosen = []
    card.clicked.connect(chosen.append)
    try:
        card.show()
        styled_qt.processEvents()
        assert status in _visible_labels(card)
        assert card.focusPolicy() == Qt.FocusPolicy.NoFocus
        QTest.mouseClick(card, Qt.MouseButton.LeftButton, pos=card.rect().center())
        QTest.keyClick(card, Qt.Key.Key_Return)
        QTest.keyClick(card, Qt.Key.Key_Space)
        assert not chosen
    finally:
        _dispose(card)


def _screen():
    from gui.screens.report_screen import ReportScreen

    screen = ReportScreen(
        "Precision study", {"description": "Inspect a supplied report."}, {}, task="regression",
        params=[{"key": "alpha", "label": "Penalty strength", "kind": "float", "min": .001,
                 "max": 10., "value": 1., "decimals": 3}],
    )
    screen.resize(1280, 860)
    screen.set_dataset({
        "name": "measurements.parquet", "rows": 4,
        "columns": [{"name": name, "type": "numeric", "missing": 0, "classes": [], "issues": []}
                    for name in ("input", "output")],
        "preview": [["1", "2"]], "notes": [], "issues": [],
    }, target="output")
    return screen


def _assert_report_strings(screen, report):
    from PySide6.QtWidgets import QLabel, QTableWidget

    labels = [label.text() for label in screen.report_widget.findChildren(QLabel)]
    assert report["summary"] in labels
    assert all(value in labels for _, value in report["metric_groups"][0]["items"])
    table = screen.report_widget.findChild(QTableWidget)
    assert [[table.item(row, col).text() for col in range(table.columnCount())]
            for row in range(table.rowCount())] == report["tables"][0]["rows"]


def test_known_metric_captions_are_display_only_and_unknown_case_is_preserved(styled_qt):
    from PySide6.QtWidgets import QLabel

    report = {
        "dataset": "measurements.parquet", "target": "output", "features": "input",
        "metric_groups": [{"title": "Test metrics", "items": [
            ("ACCURACY", "0.875000"), ("R²", "-0.125"), ("MAE", "1.23456789e-09"),
            ("mixedCase", "001"), ("UPPER", "002")]}],
        "warnings": [],
    }
    screen = _screen()
    try:
        screen.show_report(report)
        screen.show()
        styled_qt.processEvents()
        captions = [label.text() for label in screen.report_widget.findChildren(QLabel)
                    if label.objectName() == "MetricLabel"]
        assert captions == ["Accuracy", "R²", "MAE", "mixedCase", "UPPER"]
        values = [label.text() for label in screen.report_widget.findChildren(QLabel)
                  if label.objectName() == "MetricValue"]
        assert values == [value for _, value in report["metric_groups"][0]["items"]]
        assert json.loads(json.dumps(screen.session_state()["report"])) == json.loads(json.dumps(report))
    finally:
        _dispose(screen)


def test_report_precision_warnings_and_last_run_survive_json_session_restore(styled_qt):
    report = {
        "dataset": "measurements.parquet", "target": "output", "features": "input",
        "summary": "Mean error: 0.000000123456 target units.",
        "metric_groups": [{"title": "Held-out observations", "items": [
            ("MAE", "0.000000123456"), ("R²", "-0.000012345")]}],
        "tables": [{"title": "Exact values", "headers": ["Class", "Score", "Identifier"],
                    "rows": [["mixedCase", "1.23456789e-09", "00042"],
                             ["UPPER", "-123456789.012345", "01-A"]]}],
        "warnings": ["Review the signed estimate before interpretation."],
    }
    original, restored = _screen(), _screen()
    try:
        original.show()
        original.show_report(report)
        styled_qt.processEvents()
        _assert_report_strings(original, report)
        assert not original.warning_button.isHidden()
        original.param_widgets["alpha"].setValue(2.)
        assert "Settings changed" in original.result_state.text()
        state = json.loads(json.dumps(original.session_state()))
        restored.validate_session(state)
        restored.restore_session(state)
        restored.show()
        styled_qt.processEvents()
        actual = json.loads(json.dumps(restored.session_state()))
        for key in ("request", "params", "report", "report_request", "sections"):
            assert actual[key] == state[key]
        assert "Settings changed" in restored.result_state.text()
        assert not restored.warning_button.isHidden()
        _assert_report_strings(restored, report)
        shown = []
        restored._show_notes = lambda title, notes: shown.append(notes)
        restored.warning_button.click()
        assert shown == [report["warnings"]]
    finally:
        _dispose(original, restored)


@pytest.mark.parametrize("values", [(120000., 940000.), (-940000., -120000.), (1.2e-8, 9.4e-8)])
def test_plot_y_caption_does_not_overlap_complete_tick_values_at_narrow_width(styled_qt, values):
    from PySide6.QtGui import QFont, QFontMetrics

    from gui import theme
    from gui.plot import PlotWidget, _fmt, _nice_ticks

    spec = {"title": "Measured response", "y_label": "actual price_usd", "x_label": "prediction",
            "series": [{"kind": "points", "label": "test rows", "points": [(1., values[0]), (2., values[1])]}]}
    widget = PlotWidget(spec)
    original = json.dumps(spec)
    try:
        widget.resize(584, widget.height())
        widget.show()
        styled_qt.processEvents()
        metrics = QFontMetrics(QFont(theme.FONT_FAMILY, theme.PLOT_POINT_SIZE))
        plot = widget._plot_rect(metrics)
        tick_width = max(metrics.horizontalAdvance(_fmt(value)) for value in _nice_ticks(*widget._y_range))
        tick_left = plot.left() - 8 - tick_width
        caption_right = 14 + max(20, metrics.height()) / 2
        assert tick_left >= caption_right + 8
        assert plot.width() >= 400
        assert not widget.grab().isNull()
        assert json.dumps(spec) == original
    finally:
        _dispose(widget)
