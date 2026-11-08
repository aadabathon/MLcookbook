"""The SQLite dialog emits explicit requests and keeps previews independent."""

import sys
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


@pytest.fixture
def dialog(qt):
    from PySide6.QtCore import QCoreApplication, QEvent

    from gui.sqlite_dialog import SQLiteDialog

    widget = SQLiteDialog()
    widget.show()
    qt.processEvents()
    yield widget
    widget.set_busy(False)
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _view():
    return {"rows": 20, "columns": [{"name": "id"}, {"name": "value"}],
            "preview": [[f"{row:05d}", str(row)] for row in range(20)], "source": {"has_more": True}}


def test_catalog_query_preview_and_import_are_explicit(dialog):
    catalog, previews, imports = [], [], []
    dialog.catalog_requested.connect(catalog.append)
    dialog.preview_requested.connect(lambda *args: previews.append(args))
    dialog.import_requested.connect(lambda *args: imports.append(args))
    assert not dialog.import_button.isEnabled()
    dialog.path.setText("test.sqlite")
    dialog.catalog_button.click()
    assert catalog == ["test.sqlite"]
    dialog.set_catalog({"objects": [{"name": 'a"b', "kind": "view"}]})
    assert dialog.query.toPlainText() == 'SELECT * FROM "a""b"'
    assert not previews and not imports
    dialog.preview_button.click()
    assert previews == [("test.sqlite", 'SELECT * FROM "a""b"', 100_000)]
    dialog.set_preview(_view())
    assert dialog.table.item(0, 0).text() == "00000"
    assert "more rows exist" in dialog.preview_label.text()
    dialog.import_button.click()
    assert imports == previews


def test_query_path_and_limit_edits_require_matching_new_preview(dialog):
    dialog.path.setText("test.sqlite")
    dialog.query.setPlainText("SELECT 1 AS value")
    dialog.set_preview(_view())
    assert dialog.import_button.isEnabled()
    dialog.max_rows.setValue(5)
    assert not dialog.import_button.isEnabled()
    dialog.set_preview(_view())
    assert dialog.import_button.isEnabled()
    dialog.query.setPlainText("SELECT 2 AS value")
    assert not dialog.import_button.isEnabled()
    dialog.set_preview(_view())
    dialog.path.setText("different.sqlite")
    assert not dialog.import_button.isEnabled()
    dialog.set_preview(_view(), path="test.sqlite")
    assert not dialog.import_button.isEnabled()
    assert dialog.table.rowCount() == 20


def test_empty_preview_cannot_be_imported(dialog):
    dialog.path.setText("test.sqlite")
    dialog.query.setPlainText("SELECT 1 AS value WHERE 0")
    dialog.set_preview({"rows": 0, "columns": [{"name": "value"}], "preview": []})
    assert not dialog.import_button.isEnabled()


def test_busy_close_requests_cancellation_and_waits_for_retirement(dialog, qt):
    requests = []
    dialog.cancel_requested.connect(lambda: requests.append(True))
    dialog.path.setText("test.sqlite")
    dialog.query.setPlainText("SELECT 1 AS value")
    dialog.set_preview(_view())
    dialog.set_busy(True)
    assert not dialog.path.isEnabled() and not dialog.query.isEnabled()
    assert not dialog.import_button.isEnabled() and not dialog.preview_button.isEnabled()
    dialog.close_button.click()
    qt.processEvents()
    assert requests == [True] and dialog.isVisible()
    dialog.set_busy(False)
    qt.processEvents()
    assert not dialog.isVisible()
