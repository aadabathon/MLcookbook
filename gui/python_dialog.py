"""Read-only completed-run Python with explicit portable export."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout


class PythonDialog(QDialog):
    export_requested = Signal()

    def __init__(self, source: str, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Python for completed run")
        self.resize(820, 560)
        layout = QVBoxLayout(self)
        label = QLabel("Recorded settings and a portable data snapshot reproduce the completed run.")
        label.setWordWrap(True)
        layout.addWidget(label)
        self.code = QPlainTextEdit(source)
        self.code.setReadOnly(True)
        layout.addWidget(self.code)
        row = QHBoxLayout()
        copy = QPushButton("Copy Python")
        copy.clicked.connect(self._copy)
        row.addWidget(copy)
        export = QPushButton("Export Python and data…")
        export.clicked.connect(self.export_requested)
        row.addWidget(export)
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        layout.addLayout(row)

    def _copy(self):
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.code.toPlainText())
