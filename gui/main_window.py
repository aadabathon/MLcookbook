"""The application window shell: persistent header + swappable screen area.

Navigation is delete-and-redraw: ``show_screen`` destroys the current screen
widget and installs the new one. The window knows nothing about which screens
exist or when to swap them — that is app.py's job.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from gui import theme


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MLcookbook")
        self.setMinimumSize(theme.WINDOW_MIN_WIDTH, theme.WINDOW_MIN_HEIGHT)

        central = QWidget()
        self._layout = QVBoxLayout(central)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.addWidget(self._build_app_header())
        self.setCentralWidget(central)

        self._screen: QWidget | None = None

    def show_screen(self, screen: QWidget) -> None:
        """Replace the current screen (delete-and-redraw)."""
        if self._screen is not None:
            self._layout.removeWidget(self._screen)
            self._screen.deleteLater()
        self._screen = screen
        self._layout.addWidget(screen, stretch=1)

    def _build_app_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("AppHeader")
        header.setFixedHeight(64)

        row = QHBoxLayout(header)
        row.setContentsMargins(theme.SPACE_XL, 0, theme.SPACE_XL, 0)

        # Two-tone wordmark: violet "ML", pale "COOKBOOK".
        wordmark = QHBoxLayout()
        wordmark.setSpacing(0)
        ml = QLabel("ML")
        ml.setObjectName("WordmarkAccent")
        rest = QLabel("COOKBOOK")
        rest.setObjectName("Wordmark")
        wordmark.addWidget(ml)
        wordmark.addWidget(rest)

        row.addLayout(wordmark)
        row.addStretch()
        return header
