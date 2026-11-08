"""A native, schematic literature timeline; all records and actions are plain data."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QGraphicsObject, QGraphicsScene, QGraphicsView, QWidget

from gui import theme

TRACKS = ("Statistics", "Computing", "AI", "Machine learning", "Data science", "Deep learning")
TRACK_COLORS = dict(zip(TRACKS, (theme.PLOT_ALT, "#94a3b8", "#f5bd76", theme.UI_ACCENT,
                               "#69b7ef", "#ef9fca")))
LANE_HEIGHT = 86
CARD_WIDTH = 174
CARD_HEIGHT = 70
SCENE_HEIGHT = 44 + LANE_HEIGHT * len(TRACKS)


def track_of(entry):
    return entry.get("track") if entry.get("track") in TRACKS else "AI"


def _period(year):
    for limit, label in ((1900, "Before 1900"), (1950, "1900–1949"), (1980, "1950–1979"),
                         (2000, "1980–1999"), (2010, "2000–2009"), (2020, "2010–2019")):
        if year < limit:
            return label
    return "2020 onward"


class _PaperCard(QGraphicsObject):
    """A milestone with an explicit reading target, separate from selection."""

    def __init__(self, entry, view):
        super().__init__()
        self.entry = entry
        self.view = view
        self.action_rect = QRectF(CARD_WIDTH - 32, 0, 32, CARD_HEIGHT)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{entry['date']} · {entry['title']}\n{entry.get('people', '')}\n"
                        "Click to inspect; use the arrow to open the reading.")

    def boundingRect(self):
        return QRectF(0, 0, CARD_WIDTH, CARD_HEIGHT)

    def paint(self, painter, option, widget=None):
        selected = self.view.selected_id == self.entry["id"]
        color = QColor(TRACK_COLORS[track_of(self.entry)])
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.view.transform().m11() < .4:
            painter.setPen(QPen(QColor(theme.TEXT) if selected else color, 2))
            painter.setBrush(color)
            painter.drawRoundedRect(self.boundingRect().adjusted(1, 1, -1, -1), 10, 10)
            return
        painter.setPen(QPen(color if selected else QColor(theme.BORDER), 1.8 if selected else 1))
        painter.setBrush(QColor(theme.UI_SELECTION if selected else theme.CARD))
        painter.drawRoundedRect(self.boundingRect().adjusted(1, 1, -1, -1), 5, 5)
        painter.setPen(QPen(color, 3))
        painter.drawLine(QPointF(3, 14), QPointF(3, CARD_HEIGHT - 14))
        painter.setFont(QFont(theme.FONT_FAMILY, 8))
        painter.setPen(color)
        painter.drawText(QRectF(12, 7, 124, 17), Qt.AlignmentFlag.AlignLeft, self.entry["date"])
        painter.setFont(QFont(theme.FONT_FAMILY, 9))
        painter.setPen(QColor(theme.TEXT))
        title = self.entry.get("short_title", self.entry["title"])
        metrics = painter.fontMetrics()
        # Two measured lines keep every card bounded; the full title is in the inspector/tooltip.
        words = title.split()
        line = ""
        lines = []
        for word in words:
            candidate = (line + " " + word).strip()
            if line and metrics.horizontalAdvance(candidate) > 124:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        if len(lines) > 2:
            lines = [lines[0], metrics.elidedText(" ".join(lines[1:]), Qt.TextElideMode.ElideRight, 124)]
        painter.drawText(QRectF(12, 27, 124, 38), Qt.AlignmentFlag.AlignTop, "\n".join(lines))
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.drawLine(QPointF(CARD_WIDTH - 33, 11), QPointF(CARD_WIDTH - 33, CARD_HEIGHT - 11))
        painter.setFont(QFont(theme.FONT_FAMILY, 14))
        painter.setPen(color)
        painter.drawText(self.action_rect, Qt.AlignmentFlag.AlignCenter, "↗")

    def mousePressEvent(self, event):
        if not self.view._busy:
            self.view.select_entry(self.entry["id"], reveal=False)
            self.view.entry_selected.emit(self.entry["id"])
            if self.action_rect.contains(event.pos()):
                self.view.open_entry(self.entry["id"])
        event.accept()


class ResearchTimeline(QGraphicsView):
    """Six literature lanes with chronological spacing and explicit paper actions.

    Spacing follows publication order, not elapsed years. This gives old and
    densely published periods readable cards without inventing influence edges.
    """

    entry_selected = Signal(str)
    source_requested = Signal(str)
    paper_requested = Signal(str)
    viewport_changed = Signal()

    def __init__(self, entries=(), parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor(theme.BG))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setAccessibleName("Interactive literature timeline; arrows select, Enter opens reading")
        self.setMinimumWidth(0)
        self.selected_id = None
        self.node_items = {}
        self._entries = []
        self._papers = {}
        self._busy = False
        self.bands = []
        self.horizontalScrollBar().valueChanged.connect(self.viewport_changed)
        self.verticalScrollBar().valueChanged.connect(self.viewport_changed)
        self.set_entries(entries)

    def set_entries(self, entries):
        self._entries = list(entries)
        self.scene().clear()
        self.node_items = {}
        self.bands = []
        previous_year = None
        x = 30
        last_in_track = dict.fromkeys(TRACKS, -CARD_WIDTH)
        for entry in self._entries:
            track = track_of(entry)
            if previous_year is not None and entry["year"] != previous_year:
                x += 64
            x = max(x, last_in_track[track] + CARD_WIDTH + 18)
            period = _period(entry["year"])
            if not self.bands or self.bands[-1]["era"] != period:
                self.bands.append({"era": period, "x": max(0, x - 15)})
            node = _PaperCard(entry, self)
            node.setPos(x, 47 + TRACKS.index(track) * LANE_HEIGHT)
            self.scene().addItem(node)
            self.node_items[entry["id"]] = node
            last_in_track[track] = x
            previous_year = entry["year"]
        self.scene().setSceneRect(0, 0, max(650, x + CARD_WIDTH + 30), SCENE_HEIGHT)
        self.viewport_changed.emit()

    def set_library(self, papers):
        self._papers = {paper["id"]: paper for paper in papers}

    def select_entry(self, key, *, reveal=True):
        previous = self.node_items.get(self.selected_id)
        self.selected_id = key
        if previous is not None:
            previous.update()
        current = self.node_items.get(key)
        if current is not None:
            current.update()
            if reveal:
                self.ensureVisible(current, 28, 8)
        self.viewport_changed.emit()

    def open_entry(self, key):
        if self._busy or key not in self.node_items:
            return
        entry = self.node_items[key].entry
        # Prefer the original paper, never an unrelated later reading linked for context.
        for paper_id in entry.get("paper_ids", []):
            paper = self._papers.get(paper_id, {})
            if paper.get("available") and (paper_id == key or paper.get("url") in
                                          [source["url"] for source in entry.get("sources", [])]):
                self.paper_requested.emit(paper_id)
                return
        if entry.get("sources"):
            self.source_requested.emit(entry["sources"][0]["url"])

    def set_busy(self, busy):
        self._busy = bool(busy)
        self.setEnabled(not busy)

    def zoom(self, factor):
        if self._busy:
            return
        current = self.transform().m11()
        target = min(1.65, max(.12, current * factor))
        self.scale(target / current, target / current)
        self.viewport_changed.emit()

    def fit_overview(self):
        if not self._busy:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.viewport_changed.emit()

    def focus_fraction(self, fraction):
        if self._busy:
            return
        if self.transform().m11() < .7:
            self.resetTransform()
        self.centerOn(fraction * self.sceneRect().width(), SCENE_HEIGHT / 2)
        self.viewport_changed.emit()

    def view_state(self):
        return {"zoom": self.transform().m11(), "x": self.horizontalScrollBar().value(),
                "y": self.verticalScrollBar().value()}

    def restore_view(self, state):
        if not isinstance(state, dict):
            return
        values = [state.get(key) for key in ("zoom", "x", "y")]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            return
        scale, x, y = values
        self.resetTransform()
        self.scale(min(1.65, max(.02, scale)), min(1.65, max(.02, scale)))
        self.horizontalScrollBar().setValue(round(x))
        self.verticalScrollBar().setValue(round(y))
        self.viewport_changed.emit()

    def keyPressEvent(self, event):
        keys = list(self.node_items)
        if self._busy or not keys:
            event.ignore()
            return
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.open_entry(self.selected_id)
            event.accept()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Up, Qt.Key.Key_Right, Qt.Key.Key_Down,
                   Qt.Key.Key_Home, Qt.Key.Key_End):
            index = keys.index(self.selected_id) if self.selected_id in keys else 0
            if key == Qt.Key.Key_Home:
                index = 0
            elif key == Qt.Key.Key_End:
                index = len(keys) - 1
            else:
                index = max(0, min(len(keys) - 1, index +
                            (-1 if key in (Qt.Key.Key_Left, Qt.Key.Key_Up) else 1)))
            self.select_entry(keys[index])
            self.entry_selected.emit(keys[index])
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
            event.accept()
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - event.angleDelta().y())
            event.accept()
        else:
            super().wheelEvent(event)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        width = self.sceneRect().width()
        for index, _track in enumerate(TRACKS):
            top = 40 + index * LANE_HEIGHT
            painter.fillRect(QRectF(0, top, width, LANE_HEIGHT), QColor(theme.SURFACE if index % 2 == 0 else theme.BG))
            painter.setPen(QPen(QColor(theme.BORDER), .7))
            painter.drawLine(QPointF(0, top + LANE_HEIGHT), QPointF(width, top + LANE_HEIGHT))
            rail = QColor(TRACK_COLORS[_track])
            rail.setAlpha(45)
            painter.setPen(QPen(rail, 1))
            painter.drawLine(QPointF(0, top + LANE_HEIGHT / 2), QPointF(width, top + LANE_HEIGHT / 2))
        painter.setFont(QFont(theme.FONT_FAMILY, 10))
        for band in self.bands:
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(QRectF(band["x"] + 12, 8, 250, 24), band["era"])
            painter.setPen(QPen(QColor(theme.BORDER), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(band["x"], 37), QPointF(band["x"], SCENE_HEIGHT))


class TimelineOverview(QWidget):
    """Whole chronology navigator and current horizontal viewport, without fetching."""

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.setFixedHeight(78)
        self.setMinimumWidth(0)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName("Timeline overview; click a period to navigate")
        self.setToolTip("Click a period to explore. Card spacing follows publication order, not elapsed years.")
        view.viewport_changed.connect(self.update)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.SURFACE))
        area = QRectF(12, 28, max(1, self.width() - 24), 40)
        scale = area.width() / self.view.sceneRect().width()
        painter.setFont(QFont(theme.FONT_FAMILY, 8))
        previous_right = -1
        for band in self.view.bands:
            x = area.left() + band["x"] * scale
            if x > previous_right + 9:
                text = band["era"]
                painter.setPen(QColor(theme.TEXT_MUTED))
                painter.drawText(QPointF(x, 17), text)
                previous_right = x + painter.fontMetrics().horizontalAdvance(text)
        for node in self.view.node_items.values():
            x = area.left() + node.pos().x() * scale
            y = area.top() + TRACKS.index(track_of(node.entry)) * 6.5
            painter.setPen(QPen(QColor(TRACK_COLORS[track_of(node.entry)]), 3))
            painter.drawPoint(QPointF(x, y))
        visible = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        focus = QRectF(area.left() + max(0, visible.left()) * scale, 24,
                       min(area.width(), visible.width() * scale), 48)
        focus.setRight(min(area.right(), focus.right()))
        painter.setBrush(QColor(167, 139, 250, 25))
        painter.setPen(QPen(QColor(theme.UI_ACCENT), 1))
        painter.drawRoundedRect(focus, 3, 3)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            fraction = min(1, max(0, (event.position().x() - 12) / max(1, self.width() - 24)))
            self.view.focus_fraction(fraction)
            event.accept()
