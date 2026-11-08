"""Plain coordinator activity rendered without analytical imports."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import theme


class ActivityPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._records = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.SPACE_MD, theme.SPACE_SM, theme.SPACE_MD, theme.SPACE_SM)
        heading = QHBoxLayout()
        heading.addWidget(QLabel("Activity"))
        self.log_location = QLabel()
        self.log_location.setObjectName("HelpText")
        self.log_location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        heading.addWidget(self.log_location, stretch=1)
        layout.addLayout(heading)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.operations = QTreeWidget()
        self.operations.setHeaderLabels(["Operation", "Outcome", "Time"])
        self.operations.setRootIsDecorated(False)
        self.operations.setMinimumWidth(0)
        self.operations.currentItemChanged.connect(self._selected)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumWidth(0)
        splitter.addWidget(self.operations)
        splitter.addWidget(self.details)
        splitter.setSizes([420, 420])
        layout.addWidget(splitter)
        self.setMaximumHeight(230)
        self.setMinimumHeight(150)

    def set_records(self, records, log_path=""):
        selected = self.operations.currentItem()
        identifier = selected.data(0, Qt.ItemDataRole.UserRole) if selected else None
        self._records = {record["id"]: record for record in records}
        self.operations.clear()
        for record in reversed(records):
            duration = record.get("elapsed_seconds")
            item = QTreeWidgetItem([record["operation"], record["outcome"],
                                   f"{duration:.2f} s" if duration is not None else ""])
            item.setData(0, Qt.ItemDataRole.UserRole, record["id"])
            self.operations.addTopLevelItem(item)
            if record["id"] == identifier:
                self.operations.setCurrentItem(item)
        if self.operations.currentItem() is None and self.operations.topLevelItemCount():
            self.operations.setCurrentItem(self.operations.topLevelItem(0))
        self.operations.header().setStretchLastSection(False)
        self.operations.setColumnWidth(0, 240)
        self.log_location.setText("Local diagnostic log" if log_path else "")
        self.log_location.setToolTip(log_path)

    def _selected(self, item, previous=None):
        if item is None:
            self.details.clear()
            return
        record = self._records[item.data(0, Qt.ItemDataRole.UserRole)]
        lines = [record["operation"], record["outcome"], "Started (UTC): " + record["started_utc"]]
        if record.get("input"):
            lines.append("Input: " + record["input"])
        if record.get("input_identity"):
            lines.append("Snapshot: " + record["input_identity"])
        if record.get("finished_utc"):
            lines.append("Finished (UTC): " + record["finished_utc"])
        lines.extend(record.get("warnings", []))
        lines.extend(value for value in (record.get("error"), record.get("details")) if value)
        self.details.setPlainText("\n\n".join(lines))
