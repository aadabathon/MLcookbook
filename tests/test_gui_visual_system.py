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


def _contrast(foreground, background):
    from PySide6.QtGui import QColor

    def luminance(color):
        channels = QColor(color).getRgbF()[:3]
        linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in channels]
        return sum(value * weight for value, weight in zip(linear, (.2126, .7152, .0722)))

    values = sorted((luminance(foreground), luminance(background)))
    return (values[1] + .05) / (values[0] + .05)


@pytest.mark.parametrize("recipe", _ready_recipes(), ids=lambda recipe: recipe.key)
def test_ready_cards_keep_accessible_mouse_and_keyboard_actions(styled_qt, recipe):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLabel

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
        assert recipe.icon in labels
        symbol = card.findChild(QLabel, "RecipeSymbol")
        assert symbol is not None and symbol.text() == recipe.icon
        assert symbol.textFormat() == Qt.TextFormat.PlainText
        assert symbol.focusPolicy() == Qt.FocusPolicy.NoFocus
        assert symbol.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        assert card.accessibleName() == recipe.title
        assert card.accessibleDescription() == recipe.description
        assert card.focusPolicy() == Qt.FocusPolicy.StrongFocus
        QTest.mouseClick(card, Qt.MouseButton.LeftButton, pos=card.rect().center())
        QTest.mouseClick(card.windowHandle(), Qt.MouseButton.LeftButton,
                         pos=symbol.mapTo(card, symbol.rect().center()))
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            QTest.keyClick(card, key)
        assert chosen == [recipe] * 5
    finally:
        _dispose(card)


@pytest.mark.parametrize("variant,status", [("planned", "Planned"), ("soon", "Coming soon")])
def test_unavailable_cards_keep_status_without_becoming_actions(styled_qt, variant, status):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLabel

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
        symbol = card.findChild(QLabel, "RecipeSymbol")
        assert symbol is not None and symbol.text() == recipe.icon
        assert symbol.property("available") is False
        assert symbol.focusPolicy() == Qt.FocusPolicy.NoFocus
        assert card.focusPolicy() == Qt.FocusPolicy.NoFocus
        QTest.mouseClick(card, Qt.MouseButton.LeftButton, pos=card.rect().center())
        QTest.mouseClick(card.windowHandle(), Qt.MouseButton.LeftButton,
                         pos=symbol.mapTo(card, symbol.rect().center()))
        QTest.keyClick(card, Qt.Key.Key_Return)
        QTest.keyClick(card, Qt.Key.Key_Space)
        assert not chosen
    finally:
        _dispose(card)


def test_primary_and_selected_navigation_states_keep_readable_text(styled_qt):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QPalette
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QPushButton

    from gui import theme
    from gui.main_window import MainWindow

    window = MainWindow()
    primary = QPushButton("Run analysis")
    primary.setObjectName("PrimaryButton")
    window.show_screen(primary)
    window.resize(1080, 720)

    def colors(widget):
        image = widget.grab().toImage()
        ratio = image.devicePixelRatio()
        background = image.pixelColor(int(widget.width() / 2 * ratio), int((widget.height() - 5) * ratio))
        group = QPalette.ColorGroup.Active if widget.isEnabled() else QPalette.ColorGroup.Disabled
        foreground = widget.palette().color(group, QPalette.ColorRole.ButtonText)
        return foreground, background

    try:
        window.show()
        QTest.mouseMove(window.windowHandle(), QPoint(20, 20))
        styled_qt.processEvents()
        foreground, background = colors(primary)
        assert background.name() == theme.UI_PRIMARY
        assert _contrast(foreground, background) >= 4.5
        QTest.mouseMove(window.windowHandle(), primary.mapTo(window, primary.rect().center()))
        styled_qt.processEvents()
        foreground, background = colors(primary)
        assert background.name() == theme.UI_PRIMARY_HOVER
        assert _contrast(foreground, background) >= 4.5
        primary.setEnabled(False)
        styled_qt.processEvents()
        assert _contrast(*colors(primary)) >= 4.5
        window.set_navigation("data")
        selected = window.navigation_buttons["data"]
        styled_qt.processEvents()
        foreground, background = colors(selected)
        assert selected.isChecked() and background.name() == theme.UI_SELECTION
        assert _contrast(foreground, background) >= 4.5
        window.set_navigation("data", busy=True)
        styled_qt.processEvents()
        assert selected.isChecked() and not selected.isEnabled()
        foreground, background = colors(selected)
        assert background.name() == theme.UI_SELECTION
        assert _contrast(foreground, background) >= 4.5
    finally:
        _dispose(window)


def test_recipe_symbol_and_focus_colors_remain_distinguishable_from_their_surfaces():
    from gui import theme

    for background in (theme.CARD, theme.CARD_HOVER, theme.UI_SELECTION):
        assert _contrast(theme.UI_ACCENT, background) >= 4.5
    assert _contrast(theme.BORDER_ACCENT, theme.CARD) >= 3
    assert _contrast(theme.UI_BORDER, theme.CARD_HOVER) >= 3


def test_secondary_text_remains_readable_on_restored_dark_surfaces():
    from gui import theme

    for foreground in (theme.TEXT, theme.TEXT_MUTED, theme.TEXT_FAINT):
        for background in (theme.BG, theme.SURFACE, theme.CARD, theme.CARD_HOVER):
            assert _contrast(foreground, background) >= 4.5


def test_display_typography_does_not_spread_to_controls_or_report_values(styled_qt):
    from PySide6.QtWidgets import QLabel

    from gui import theme
    from gui.main_window import MainWindow

    window, screen = MainWindow(), _screen()
    window.show_screen(screen)
    screen.show_report({
        "dataset": "measurements.parquet", "target": "output", "features": "input",
        "metric_groups": [{"title": "Test metrics", "items": [("MAE", "0.000012345")] }],
        "warnings": [],
    })
    try:
        window.show()
        styled_qt.processEvents()
        for name in ("Wordmark", "ScreenTitle"):
            label = window.findChild(QLabel, name)
            assert label is not None and label.font().families()[0] == "Palatino Linotype"
        value = screen.report_widget.findChild(QLabel, "MetricValue")
        assert value is not None and value.text() == "0.000012345"
        for widget in (screen.run_button, screen.param_widgets["alpha"], value):
            assert widget.font().families()[0] == theme.FONT_FAMILY
    finally:
        _dispose(window)


def test_ui_accent_changes_preserve_analytical_series_and_category_colors():
    from gui import theme

    assert theme.palette("accent") == "#b29aef"
    assert theme.palette("alt") == "#2dd4bf"
    assert [theme.palette(f"c{index}") for index in range(10)] == [
        "#8b5cf6", "#2dd4bf", "#f59e0b", "#f472b6", "#60a5fa",
        "#a3e635", "#fb7185", "#e879f9", "#facc15", "#94a3b8",
    ]


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
