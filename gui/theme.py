"""Centralized visual constants and the application stylesheet.

Every color, font, spacing value, and card dimension lives here so the look
can be tuned in one place. Widgets opt into styles via objectName (e.g.
``RecipeCard``) and dynamic properties (e.g. ``variant`` on status pills).
"""

from __future__ import annotations

# --- Palette ------------------------------------------------------------------
BG = "#0a0a10"  # window background, near-black with a violet cast
SURFACE = "#101018"  # page panel
CARD = "#12121c"  # card face
CARD_HOVER = "#171624"
BORDER = "#262238"  # resting borders
BORDER_ACCENT = "#4c3d78"  # hover / emphasized borders
ACCENT = "#8b5cf6"  # primary violet
ACCENT_SOFT = "#a78bfa"  # lighter violet for text accents
ACCENT_DIM = "#6d55b0"
TEXT = "#e8e4f2"
TEXT_MUTED = "#8f8a9e"
TEXT_FAINT = "#6b6678"

# --- Typography ---------------------------------------------------------------
SERIF = '"Palatino Linotype", "Book Antiqua", Georgia, serif'
SANS = '"Segoe UI", "Helvetica Neue", Arial, sans-serif'
MONO = '"Cascadia Code", Consolas, "Courier New", monospace'
BODY_POINT_SIZE = 10

# --- Spacing / dimensions -----------------------------------------------------
SPACE_XS = 6
SPACE_SM = 12
SPACE_MD = 20
SPACE_LG = 32
SPACE_XL = 48

CARD_RADIUS = 12
CARD_MIN_HEIGHT = 230
CARD_PADDING = 22
ICON_SIZE = 60
GRID_COLUMNS = 3

WINDOW_MIN_WIDTH = 1080
WINDOW_MIN_HEIGHT = 720


def stylesheet() -> str:
    """The full application QSS, assembled from the constants above."""
    return f"""
    QMainWindow, QWidget {{
        background-color: {BG};
        color: {TEXT};
        font-family: {SANS};
    }}

    /* --- Header bar ------------------------------------------------------- */
    QFrame#AppHeader {{
        background-color: {BG};
        border-bottom: 1px solid {BORDER};
    }}
    QFrame#AppHeader QLabel {{
        background: transparent;
    }}
    QLabel#Wordmark {{
        font-family: {SERIF};
        font-size: 24px;
        letter-spacing: 7px;
        color: {TEXT};
    }}
    QLabel#WordmarkAccent {{
        font-family: {SERIF};
        font-size: 24px;
        letter-spacing: 7px;
        color: {ACCENT_SOFT};
    }}

    /* --- Page header ------------------------------------------------------ */
    QLabel#PageTitle {{
        font-family: {SERIF};
        font-size: 30px;
        color: {TEXT};
    }}
    QLabel#PageSubtitle {{
        font-size: 13px;
        color: {TEXT_MUTED};
    }}
    QFrame#PageTitleTick {{
        background-color: {ACCENT};
        border-radius: 1px;
    }}

    /* --- Recipe cards ----------------------------------------------------- */
    QFrame#RecipeCard {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: {CARD_RADIUS}px;
    }}
    QFrame#RecipeCard:hover {{
        background-color: {CARD_HOVER};
        border: 1px solid {BORDER_ACCENT};
    }}
    QFrame#RecipeCard QLabel {{
        background: transparent;
    }}
    QLabel#CardIcon {{
        font-family: {SERIF};
        font-size: 22px;
        color: {ACCENT_SOFT};
        border: 1px solid {BORDER_ACCENT};
        border-radius: {ICON_SIZE // 2}px;
        background-color: rgba(139, 92, 246, 14);
    }}
    QLabel#CardTitle {{
        font-family: {SERIF};
        font-size: 18px;
        color: {TEXT};
    }}
    QLabel#CardDescription {{
        font-size: 12px;
        color: {TEXT_MUTED};
    }}

    /* --- Status pills ----------------------------------------------------- */
    QLabel#StatusPill {{
        font-size: 11px;
        letter-spacing: 1px;
        color: {ACCENT_SOFT};
        border: 1px solid {BORDER_ACCENT};
        border-radius: 11px;
        padding: 3px 14px;
    }}
    QLabel#StatusPill[variant="ready"] {{
        color: {BG};
        background-color: {ACCENT};
        border: 1px solid {ACCENT};
    }}
    QLabel#StatusPill[variant="soon"] {{
        color: {TEXT_FAINT};
        border: 1px solid {BORDER};
    }}

    /* --- Recipe screens --------------------------------------------------- */
    QPushButton#BackButton {{
        background: transparent;
        border: none;
        color: {TEXT_MUTED};
        font-size: 12px;
        letter-spacing: 1px;
        padding: 4px 0;
        text-align: left;
    }}
    QPushButton#BackButton:hover {{
        color: {ACCENT_SOFT};
    }}
    QLabel#ScreenTitle {{
        font-family: {SERIF};
        font-size: 28px;
        color: {TEXT};
    }}
    QLabel#ScreenSubtitle {{
        font-size: 13px;
        color: {TEXT_MUTED};
    }}
    QLabel#EngineInfo {{
        font-family: {MONO};
        font-size: 11px;
        color: {TEXT_FAINT};
        padding: 6px 0;
        border-top: 1px solid {BORDER};
        border-bottom: 1px solid {BORDER};
    }}
    QLabel#FieldLabel {{
        font-size: 10px;
        letter-spacing: 2px;
        color: {TEXT_FAINT};
    }}
    QLineEdit {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 8px 12px;
        color: {TEXT};
        font-size: 12px;
        selection-background-color: {ACCENT_DIM};
    }}
    QLineEdit:focus {{
        border: 1px solid {BORDER_ACCENT};
    }}
    QLineEdit::placeholder {{
        color: {TEXT_FAINT};
    }}
    QPushButton#PrimaryButton {{
        background-color: {ACCENT};
        border: 1px solid {ACCENT};
        border-radius: 8px;
        padding: 9px 18px;
        color: {BG};
        font-size: 12px;
        letter-spacing: 1px;
    }}
    QPushButton#PrimaryButton:hover {{
        background-color: {ACCENT_SOFT};
    }}
    QPushButton#PrimaryButton:disabled {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        color: {TEXT_FAINT};
    }}
    QFrame#GraphPlaceholder {{
        background-color: {SURFACE};
        border: 1px dashed {BORDER_ACCENT};
        border-radius: {CARD_RADIUS}px;
    }}
    QLabel#GraphHint {{
        font-family: {SERIF};
        font-size: 15px;
        color: {TEXT_FAINT};
        background: transparent;
    }}

    /* --- Scroll area / scrollbar ------------------------------------------ */
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: {SURFACE};
        width: 10px;
        margin: 0;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical {{
        background: {ACCENT_DIM};
        min-height: 48px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {ACCENT};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent;
    }}
    """
