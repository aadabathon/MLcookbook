"""Shared visual system for the native desktop workbench.

Ink-black surfaces and serif display headings retain the cookbook's character.
Controls and plots use Segoe UI; violet interaction and semantic colors stay distinct.
Widget object names remain stable for saved-screen and presentation contracts.
"""
from __future__ import annotations

BG = "#0a0a10"
SURFACE = "#101018"
CARD = "#13121c"
CARD_HOVER = "#191624"
BORDER = "#302a40"
ACCENT = "#b29aef"
ACCENT_SOFT = "#cbb9f5"
ACCENT_DIM = "#40364f"
UI_ACCENT = "#a78bfa"
UI_PRIMARY = "#9b75e8"
UI_PRIMARY_HOVER = "#b99af8"
UI_SELECTION = "#241c35"
UI_BORDER = "#8c6fbc"
BORDER_ACCENT = UI_ACCENT
TEXT = "#eee9f5"
TEXT_MUTED = "#aaa3b7"
TEXT_FAINT = "#9991a7"
WARNING = "#edc58a"
WARNING_BG = "#2c2822"
WARNING_BORDER = "#786444"
SCROLL_HANDLE = "#5b626f"
PLOT_ALT = "#2dd4bf"
CATEGORICAL = ["#8b5cf6", "#2dd4bf", "#f59e0b", "#f472b6", "#60a5fa",
               "#a3e635", "#fb7185", "#e879f9", "#facc15", "#94a3b8"]

FONT_FAMILY = "Segoe UI"
SANS = '\"Segoe UI\", \"Helvetica Neue\", Arial, sans-serif'
SERIF = '\"Palatino Linotype\", \"Book Antiqua\", Georgia, serif'
MONO = '\"Cascadia Code\", Consolas, \"Courier New\", monospace'
BODY_POINT_SIZE = 11
PLOT_POINT_SIZE = 9
PLOT_TITLE_POINT_SIZE = 10

SPACE_XS = 6
SPACE_SM = 10
SPACE_MD = 16
SPACE_LG = 24
SPACE_XL = 32
CARD_RADIUS = 5
CARD_MIN_HEIGHT = 174
CARD_PADDING = 20
GRID_COLUMNS = 3
WINDOW_MIN_WIDTH = 1080
WINDOW_MIN_HEIGHT = 720
SIDEBAR_WIDTH = 180


def palette(name: str) -> str:
    """Resolve stable series/category identifiers used by plain plot specs."""
    if name.startswith("c") and name[1:].isdigit():
        return CATEGORICAL[int(name[1:]) % len(CATEGORICAL)]
    return {"accent": ACCENT, "alt": PLOT_ALT, "muted": TEXT_FAINT}.get(name, ACCENT)


def stylesheet() -> str:
    """Shared Qt styling; values and layout state remain owned by the widgets."""
    return f"""
    QMainWindow, QWidget {{
        background-color: {BG}; color: {TEXT}; font-family: {SANS};
        font-size: 14px; font-weight: 400;
    }}
    QFrame#AppHeader {{ border-bottom: 1px solid {BORDER}; }}
    QFrame#AppHeader QLabel {{ background: transparent; }}
    QFrame#Sidebar {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
    QFrame#Sidebar QLabel {{ background: transparent; }}
    QPushButton#SidebarButton {{
        background: transparent; color: {TEXT_MUTED}; border: 1px solid transparent;
        border-radius: 4px; padding: 9px 10px; min-height: 20px; text-align: left;
    }}
    QPushButton#SidebarButton:hover {{ background-color: {CARD_HOVER}; color: {TEXT}; }}
    QPushButton#SidebarButton:checked {{ background-color: {UI_SELECTION}; color: {UI_ACCENT}; border-color: {UI_BORDER}; }}
    QPushButton#SidebarButton:focus {{ border-color: {BORDER_ACCENT}; }}
    QPushButton#SidebarButton:disabled {{ color: {TEXT_FAINT}; }}
    QLabel#SidebarCaption {{ font-size: 13px; color: {TEXT_MUTED}; }}
    QLabel#SidebarDataset {{ font-size: 14px; color: {TEXT}; }}
    QLabel#Wordmark {{ font-family: {SERIF}; font-size: 22px; font-weight: 400; color: {TEXT}; }}
    QLabel#PageTitle, QLabel#ScreenTitle {{ font-family: {SERIF}; font-size: 28px; font-weight: 400; color: {TEXT}; }}

    QFrame#RecipeCard {{
        background-color: {CARD}; border: 1px solid {BORDER}; border-radius: {CARD_RADIUS}px;
    }}
    QFrame#RecipeCard:hover {{ background-color: {CARD_HOVER}; border-color: {TEXT_FAINT}; }}
    QFrame#RecipeCard[available="true"]:hover {{ border-color: {UI_BORDER}; }}
    QFrame#RecipeCard:focus {{ border: 2px solid {BORDER_ACCENT}; }}
    QFrame#RecipeCard QLabel {{ background: transparent; }}
    QLabel#RecipeSymbol {{ font-family: {SERIF}; font-size: 25px; font-weight: 400; color: {UI_ACCENT}; }}
    QLabel#RecipeSymbol[available="false"] {{ color: {TEXT_FAINT}; }}
    QLabel#CardTitle {{ font-family: {SERIF}; font-size: 20px; font-weight: 400; color: {TEXT}; }}
    QLabel#CardDescription {{ font-size: 13px; color: {TEXT_MUTED}; }}
    QLabel#StatusPill {{ font-size: 13px; color: {TEXT_MUTED}; padding-top: 4px; }}

    QPushButton#BackButton {{
        background: transparent; border: 1px solid transparent; color: {TEXT_MUTED};
        padding: 4px 0; text-align: left;
    }}
    QPushButton#BackButton:hover {{ color: {TEXT}; }}
    QPushButton#BackButton:focus {{ border-color: {BORDER_ACCENT}; }}
    QLabel#EngineInfo {{ font-family: {MONO}; font-size: 13px; color: {TEXT_MUTED}; padding: 6px 0; }}
    QLabel#FieldLabel {{ font-size: 14px; font-weight: 500; color: {TEXT_MUTED}; }}
    QLabel#HelpText, QLabel#InputContext {{ font-size: 13px; color: {TEXT_MUTED}; }}
    QTabWidget#KitchenInspector::pane {{ border: none; }}
    QTabWidget#KitchenInspector QTabBar::tab {{
        background: {BG}; color: {TEXT_MUTED}; border: 1px solid transparent;
        border-radius: 4px; padding: 7px 12px;
    }}
    QTabWidget#KitchenInspector QTabBar::tab:selected {{
        background: {UI_SELECTION}; color: {UI_ACCENT}; border-color: {UI_BORDER};
    }}
    QTabWidget#KitchenInspector QTabBar::tab:focus {{ border-color: {BORDER_ACCENT}; }}
    QFrame#ActionBar {{ border-top: 1px solid {BORDER}; }}
    QLabel#ResultState {{ font-size: 13px; color: {TEXT_MUTED}; }}
    QLabel#ResultSummary {{ background-color: {SURFACE}; padding: 12px; border-radius: 0px; border-left: 2px solid {UI_ACCENT}; }}

    QPushButton {{
        background-color: {CARD}; border: 1px solid {BORDER}; border-radius: 4px;
        padding: 7px 10px; min-height: 18px;
    }}
    QPushButton:hover {{ background-color: {CARD_HOVER}; border-color: {TEXT_FAINT}; }}
    QPushButton:focus {{ border-color: {BORDER_ACCENT}; }}
    QPushButton:disabled {{ color: {TEXT_FAINT}; background-color: {SURFACE}; }}
    QPushButton#SectionToggle, QPushButton#RunDetailsToggle {{ text-align: left; color: {TEXT}; }}
    QPushButton#WarningSummary {{
        background-color: {WARNING_BG}; color: {WARNING}; border-color: {WARNING_BORDER}; text-align: left;
    }}
    QPushButton#PrimaryButton {{
        background-color: {UI_PRIMARY}; color: {BG}; border: 1px solid {UI_PRIMARY};
        border-radius: 4px; padding: 9px 16px; font-weight: 500;
    }}
    QPushButton#PrimaryButton:hover {{ background-color: {UI_PRIMARY_HOVER}; border-color: {UI_PRIMARY_HOVER}; }}
    QPushButton#PrimaryButton:focus {{ border-color: {TEXT}; }}
    QPushButton#PrimaryButton:disabled {{ background-color: {SURFACE}; border-color: {BORDER}; color: {TEXT_FAINT}; }}

    QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {{
        background-color: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
        border-radius: 4px; padding: 5px 7px; min-height: 21px;
    }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus {{ border-color: {BORDER_ACCENT}; }}
    QComboBox QAbstractItemView {{
        background-color: {CARD}; color: {TEXT}; selection-background-color: {ACCENT_DIM}; selection-color: {TEXT};
    }}
    QLineEdit, QPlainTextEdit {{
        background-color: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 4px;
        padding: 8px 10px; selection-background-color: {ACCENT_DIM}; selection-color: {TEXT};
    }}
    QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {BORDER_ACCENT}; }}
    QLineEdit::placeholder {{ color: {TEXT_FAINT}; }}
    QProgressBar {{ border: none; background-color: {CARD}; }}
    QProgressBar::chunk {{ background-color: {ACCENT}; }}
    QToolTip {{ background-color: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px; font-size: 13px; }}

    QFrame#MetricTile {{ background-color: {SURFACE}; border: none; border-radius: 4px; }}
    QFrame#MetricTile QLabel {{ background: transparent; }}
    QLabel#MetricValue {{ font-size: 20px; font-weight: 400; color: {TEXT}; }}
    QLabel#MetricLabel {{ font-size: 13px; color: {TEXT_MUTED}; }}
    QLabel#WarningNote {{ color: {WARNING}; border-left: 2px solid {WARNING_BORDER}; padding: 5px 10px; }}
    QFrame#GraphPlaceholder {{ background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; }}
    QLabel#GraphHint {{ color: {TEXT_MUTED}; background: transparent; }}

    QTableWidget {{
        background-color: {CARD}; alternate-background-color: {SURFACE}; color: {TEXT};
        border: 1px solid {BORDER}; border-radius: 5px; gridline-color: {BORDER};
        selection-background-color: {ACCENT_DIM}; selection-color: {TEXT};
    }}
    QTableWidget::item {{ padding: 3px 8px; border-bottom: 1px solid {BORDER}; }}
    QHeaderView::section {{
        background-color: {SURFACE}; color: {TEXT_MUTED}; border: none;
        border-bottom: 1px solid {BORDER}; padding: 7px 8px; font-size: 13px; font-weight: 500;
    }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{ background: {SURFACE}; width: 10px; margin: 0; border-radius: 5px; }}
    QScrollBar:horizontal {{ background: {SURFACE}; height: 10px; margin: 0; border-radius: 5px; }}
    QScrollBar::handle:vertical {{ background: {SCROLL_HANDLE}; min-height: 40px; border-radius: 5px; }}
    QScrollBar::handle:horizontal {{ background: {SCROLL_HANDLE}; min-width: 40px; border-radius: 5px; }}
    QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{ background: {TEXT_FAINT}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    """
