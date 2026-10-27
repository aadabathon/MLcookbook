"""The application window shell: persistent navigation + swappable screen area.

Navigation is delete-and-redraw: ``show_screen`` destroys the current screen
widget and installs the new one. The window knows nothing about which screens
exist or when to swap them — that is app.py's job.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui import theme


class MainWindow(QMainWindow):
    experiment_open_requested = Signal(str)
    experiment_save_requested = Signal(str)
    navigation_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MLcookbook")
        self.setMinimumSize(theme.WINDOW_MIN_WIDTH, theme.WINDOW_MIN_HEIGHT)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_app_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.sidebar = self._build_sidebar()
        body.addWidget(self.sidebar)
        content = QWidget()
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        body.addWidget(content, stretch=1)
        outer.addLayout(body, stretch=1)
        self.setCentralWidget(central)

        self._screen: QWidget | None = None

    def set_experiment_actions(self, *, can_save: bool, busy: bool):
        self.open_experiment_button.setEnabled(not busy)
        self.save_experiment_button.setEnabled(can_save and not busy)
        for button in self.navigation_buttons.values():
            button.setEnabled(not busy)

    def set_navigation(self, active: str, *, busy: bool = False):
        """Select a shell destination without initiating navigation."""
        active = active if active in self.navigation_buttons else "recipes"
        self.navigation_buttons[active].setChecked(True)
        for button in self.navigation_buttons.values():
            button.setEnabled(not busy)

    def set_dataset_context(self, name: str, detail: str = ""):
        """Identify the ordinary workspace dataset, independently of recipe input."""
        full_name = name or "No dataset loaded"
        label = self.dataset_name_label
        label.ensurePolished()
        width = theme.SIDEBAR_WIDTH - theme.SPACE_MD * 2 - 2
        label.setText(label.fontMetrics().elidedText(full_name, Qt.TextElideMode.ElideMiddle, width))
        label.setToolTip(full_name)
        label.setAccessibleName(full_name)
        self.dataset_detail_label.setText(detail)
        self.dataset_detail_label.setToolTip(detail)
        self.dataset_detail_label.setVisible(bool(detail))

    def set_status(self, message: str):
        self.statusBar().showMessage(message)
        self.statusBar().setToolTip(message)
        self.statusBar().setVisible(bool(message))

    def _open_experiment(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open experiment", "", "MLcookbook experiments (*.mlcb)")
        if path:
            self.experiment_open_requested.emit(path)

    def _save_experiment(self):
        dialog = QFileDialog(self, "Save experiment (includes a copy of the dataset)",
                             "experiment.mlcb", "MLcookbook experiments (*.mlcb)")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("mlcb")
        if dialog.exec():
            self.experiment_save_requested.emit(dialog.selectedFiles()[0])

    def show_screen(self, screen: QWidget) -> None:
        """Replace the current screen (delete-and-redraw)."""
        if self._screen is not None:
            self._layout.removeWidget(self._screen)
            self._screen.deleteLater()
        self._screen = screen
        self._layout.addWidget(screen, stretch=1)

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(theme.SIDEBAR_WIDTH)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(theme.SPACE_MD, theme.SPACE_LG, theme.SPACE_MD, theme.SPACE_MD)
        layout.setSpacing(theme.SPACE_XS)
        self.navigation_buttons = {}
        self._navigation_group = QButtonGroup(self)
        for key, title in (("data", "Data"), ("recipes", "Recipes"), ("kitchen", "Kitchen"),
                           ("llm", "Local LLMs"), ("wearable-study", "Movement Study")):
            button = QPushButton(title)
            button.setObjectName("SidebarButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, destination=key: self.navigation_requested.emit(destination))
            self._navigation_group.addButton(button)
            self.navigation_buttons[key] = button
            layout.addWidget(button)
        layout.addStretch()
        caption = QLabel("Active dataset")
        caption.setObjectName("SidebarCaption")
        layout.addWidget(caption)
        self.dataset_name_label = QLabel()
        self.dataset_name_label.setObjectName("SidebarDataset")
        self.dataset_name_label.setTextFormat(Qt.TextFormat.PlainText)
        self.dataset_name_label.setMinimumWidth(0)
        self.dataset_name_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.dataset_name_label)
        self.dataset_detail_label = QLabel()
        self.dataset_detail_label.setObjectName("SidebarCaption")
        self.dataset_detail_label.setTextFormat(Qt.TextFormat.PlainText)
        self.dataset_detail_label.setWordWrap(True)
        self.dataset_detail_label.setMinimumWidth(0)
        layout.addWidget(self.dataset_detail_label)
        self.set_dataset_context("")
        self.set_navigation("recipes")
        return sidebar

    def _build_app_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("AppHeader")
        header.setFixedHeight(64)

        row = QHBoxLayout(header)
        row.setContentsMargins(theme.SPACE_XL, 0, theme.SPACE_XL, 0)

        wordmark = QLabel("ML Cookbook")
        wordmark.setObjectName("Wordmark")
        row.addWidget(wordmark)
        row.addStretch()
        self.open_experiment_button = QPushButton("Open experiment…")
        self.open_experiment_button.clicked.connect(self._open_experiment)
        self.save_experiment_button = QPushButton("Save experiment…")
        self.save_experiment_button.setToolTip("Save this recipe's settings, last report, and a copy of the dataset.")
        self.save_experiment_button.clicked.connect(self._save_experiment)
        self.save_experiment_button.setEnabled(False)
        row.addWidget(self.open_experiment_button)
        row.addWidget(self.save_experiment_button)
        return header
