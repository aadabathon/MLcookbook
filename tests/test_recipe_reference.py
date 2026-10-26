"""Every ready recipe has readable reference content without core work in the dialog."""

import gc
import inspect
import sys
import weakref
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLabel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402
from gui.recipe_reference import (  # noqa: E402
    CALLABLE_RECIPES,
    REFERENCE_CONTENT,
    SECTION_TITLES,
    RecipeReferenceDialog,
    recipe_reference,
)
from gui.recipes import RECIPES  # noqa: E402
from gui.screens.report_screen import MAX_CLASSES  # noqa: E402


@pytest.fixture(scope="module")
def qt():
    return QApplication.instance() or QApplication([])


def dispose(dialog, qt):
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt.processEvents()


def test_every_ready_recipe_has_a_callable_matched_reference():
    screens = app._screens()
    assert set(REFERENCE_CONTENT) == {recipe.key for recipe in RECIPES if recipe.variant == "ready"}
    for key, (function, _) in screens.items():
        engine = app._engine_info(function)
        assert CALLABLE_RECIPES[engine["callable"]] == key
        reference = recipe_reference(engine)
        assert tuple(section for section, _ in reference) == SECTION_TITLES
        assert all(text.strip() for _, text in reference)
        assert recipe_reference(dict(engine, summary="changed", description="changed")) == reference


@pytest.mark.parametrize("key", list(REFERENCE_CONTENT))
def test_reference_is_scrollable_and_api_is_optional(qt, key):
    callable_path = next(name for name, recipe in CALLABLE_RECIPES.items() if recipe == key)
    engine = {"callable": callable_path, "signature": "(dataset, *, setting='value')", "summary": "Original"}
    before = deepcopy(engine)
    dialog = RecipeReferenceDialog("Recipe", engine)
    dialog.show()
    qt.processEvents()
    assert tuple(dialog.reference_labels) == SECTION_TITLES
    assert dialog.scroll.widgetResizable()
    assert dialog.api_toggle.isCheckable() and not dialog.api_toggle.isChecked()
    assert not dialog.api_text.isVisible()
    assert dialog.api_text.isReadOnly()
    assert dialog.api_text.toPlainText() == callable_path + engine["signature"]
    dialog.api_toggle.click()
    qt.processEvents()
    assert dialog.api_text.isVisible()
    assert "▾" in dialog.api_toggle.text()
    dialog.api_toggle.click()
    assert not dialog.api_text.isVisible()
    assert engine == before
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Close).click()
    assert not dialog.isVisible()
    dispose(dialog, qt)


def test_reference_treats_unknown_metadata_as_literal_text(qt):
    title = "<b>Literal recipe & name</b>"
    engine = {"callable": "<script>unknown</script>", "signature": "(x='<img src=remote>')",
              "summary": "<a href='https://invalid'>A & B</a>"}
    dialog = RecipeReferenceDialog(title, engine)
    assert dialog.reference_labels["Recipe reference"].text() == engine["summary"]
    assert any(label.text() == title for label in dialog.findChildren(QLabel))
    assert all(label.textFormat() == Qt.TextFormat.PlainText for label in dialog.findChildren(QLabel))
    assert dialog.api_text.toPlainText() == engine["callable"] + engine["signature"]
    assert recipe_reference({}) == (("Recipe reference", "No recipe reference is available."),)
    dispose(dialog, qt)


def test_pca_reference_matches_screen_controls_and_core_default(qt):
    function, screen_class = app._screens()["pca"]
    engine = app._engine_info(function)
    screen = screen_class(engine, {})
    reference = dict(recipe_reference(engine))
    assert "n_components" not in screen.param_widgets
    assert inspect.signature(function).parameters["n_components"].default is None
    assert "all available components" in reference["Settings"]
    assert "no component-count control" in reference["Settings"]
    assert "90%" in reference["Interpreting results"]
    dispose(screen, qt)


def test_classification_references_match_the_ui_class_limit(qt):
    classification_keys = set()
    for key, (function, screen_class) in app._screens().items():
        engine = app._engine_info(function)
        screen = screen_class(engine, {})
        if "classification" in (screen._tasks or (screen._fixed_task,)):
            classification_keys.add(key)
            assert f"2–{MAX_CLASSES} classes" in dict(recipe_reference(engine))["Data and assumptions"]
        dispose(screen, qt)
    assert classification_keys == {"logistic-regression", "decision-tree", "random-forest", "svm", "neural-network"}


def test_forest_reference_distinguishes_the_actual_task_defaults(qt):
    from mlcookbook.regression.forest import random_forest_regression

    function, screen_class = app._screens()["random-forest"]
    screen = screen_class(app._engine_info(function), {})
    assert "max_features" not in screen.param_widgets
    assert inspect.signature(function).parameters["max_features"].default == "sqrt"
    assert inspect.signature(random_forest_regression).parameters["max_features"].default == 1.0
    assumptions = dict(recipe_reference(app._engine_info(function)))["Data and assumptions"]
    assert "classifier considers random feature subsets" in assumptions
    assert "regressor's default considers all encoded features" in assumptions
    dispose(screen, qt)


def test_reference_dialog_and_controls_are_released_after_deferred_delete(qt):
    dialog = RecipeReferenceDialog("Reference", {"callable": "mlcookbook.llm.generate_text"})
    references = [weakref.ref(widget) for widget in (dialog, dialog.api_toggle, dialog.api_text)]
    dialog.show()
    qt.processEvents()
    dialog.api_toggle.click()
    assert dialog.api_text.isVisible()
    dialog.api_toggle.click()
    assert not dialog.api_text.isVisible()
    dialog.close()
    dialog.deleteLater()
    del dialog
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt.processEvents()
    gc.collect()
    assert all(reference() is None for reference in references)


def test_reference_module_import_does_not_load_the_core():
    import subprocess

    code = "import sys; import gui.recipe_reference; assert not any(n == 'mlcookbook' or n.startswith('mlcookbook.') for n in sys.modules)"
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
