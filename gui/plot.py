"""A small 2-D plot widget drawn with QPainter (no plotting dependency).

Purely presentational: it renders a plain ``spec`` dict handed in by app.py.

    {
        "title": "Actual vs predicted",
        "x_label": "predicted", "y_label": "actual",
        "series": [
            {"kind": "points", "color": "accent", "label": "test rows", "points": [(x, y), ...]},
            {"kind": "line",   "color": "muted",  "label": "perfect fit", "points": [(x, y), ...]},
        ],
    }

``color`` is "accent", "alt", "muted" (dashed when a line), or "c0".."c9" from
the categorical palette in theme.py. Axis ranges are fitted to the data.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from gui import theme

_MARGIN_LEFT, _MARGIN_RIGHT, _MARGIN_TOP, _MARGIN_BOTTOM = 62, 20, 44, 46


def _nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Round-numbered tick positions covering [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw), default=raw)
    first = math.ceil(lo / step) * step
    ticks, value = [], first
    while value <= hi + step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _fmt(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 100 or value == int(value) else f"{value:g}"


class PlotWidget(QWidget):
    def __init__(self, spec: dict, parent=None) -> None:
        super().__init__(parent)
        self._spec = spec
        self.setMinimumHeight(340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(360)

        xs = [p[0] for s in spec["series"] for p in s["points"]]
        ys = [p[1] for s in spec["series"] for p in s["points"]]
        self._x_range = self._padded(min(xs), max(xs)) if xs else (0.0, 1.0)
        self._y_range = self._padded(min(ys), max(ys)) if ys else (0.0, 1.0)

    @staticmethod
    def _padded(lo: float, hi: float) -> tuple[float, float]:
        pad = (hi - lo) * 0.05 or 1.0
        return lo - pad, hi + pad

    def _plot_rect(self, metrics: QFontMetrics) -> QRectF:
        tick_width = max((metrics.horizontalAdvance(_fmt(tick))
                          for tick in _nice_ticks(*self._y_range)), default=0)
        # Keep the rotated caption and the complete tick values in separate bands.
        caption_right = 14 + max(20, metrics.height()) / 2 if self._spec.get("y_label") else 0
        left = max(_MARGIN_LEFT, math.ceil(caption_right + 8 + tick_width + 8))
        return QRectF(left, _MARGIN_TOP, self.width() - left - _MARGIN_RIGHT,
                      self.height() - _MARGIN_TOP - _MARGIN_BOTTOM)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h = self.height()
        small = QFont(theme.FONT_FAMILY, theme.PLOT_POINT_SIZE)
        p.setFont(small)
        plot = self._plot_rect(p.fontMetrics())
        (x0, x1), (y0, y1) = self._x_range, self._y_range

        def to_px(x: float, y: float) -> QPointF:
            return QPointF(
                plot.left() + (x - x0) / (x1 - x0) * plot.width(),
                plot.bottom() - (y - y0) / (y1 - y0) * plot.height(),
            )

        p.fillRect(self.rect(), QColor(theme.SURFACE))

        # Grid + tick labels
        for tx in _nice_ticks(x0, x1):
            px = to_px(tx, y0).x()
            p.setPen(QPen(QColor(theme.BORDER), 1))
            p.drawLine(QPointF(px, plot.top()), QPointF(px, plot.bottom()))
            p.setPen(QColor(theme.TEXT_MUTED))
            p.drawText(QRectF(px - 30, plot.bottom() + 4, 60, 16), Qt.AlignmentFlag.AlignCenter, _fmt(tx))
        for ty in _nice_ticks(y0, y1):
            py = to_px(x0, ty).y()
            p.setPen(QPen(QColor(theme.BORDER), 1))
            p.drawLine(QPointF(plot.left(), py), QPointF(plot.right(), py))
            p.setPen(QColor(theme.TEXT_MUTED))
            p.drawText(
                QRectF(0, py - 8, plot.left() - 8, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                _fmt(ty),
            )
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(plot)

        # Data (clipped to the plot area)
        p.save()
        p.setClipRect(plot)
        for series in self._spec["series"]:
            color = QColor(theme.palette(series.get("color", "accent")))
            pts = [to_px(x, y) for x, y in series["points"]]
            if series["kind"] == "line":
                pen = QPen(color, 2)
                pen.setStyle(Qt.PenStyle.DashLine if series.get("color") == "muted" else Qt.PenStyle.SolidLine)
                p.setPen(pen)
                for a, b in zip(pts, pts[1:]):
                    p.drawLine(a, b)
            else:
                fill = QColor(color)
                fill.setAlpha(150)
                p.setPen(QPen(color, 1))
                p.setBrush(fill)
                for pt in pts:
                    p.drawEllipse(pt, 3.5, 3.5)
        p.restore()

        # Title, axis labels, legend (the title gets whatever width the legend leaves)
        p.setFont(small)
        labels = [s.get("label") for s in self._spec["series"] if s.get("label")]
        legend_w = sum(p.fontMetrics().horizontalAdvance(label) + 26 for label in labels)
        p.setPen(QColor(theme.TEXT))
        p.setFont(QFont(theme.FONT_FAMILY, theme.PLOT_TITLE_POINT_SIZE))
        title_w = max(80.0, plot.right() - legend_w - plot.left() - 12)
        title = p.fontMetrics().elidedText(self._spec.get("title", ""), Qt.TextElideMode.ElideRight, int(title_w))
        p.drawText(QRectF(plot.left(), 8, title_w, 24), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   title)
        p.setFont(small)
        p.setPen(QColor(theme.TEXT_MUTED))
        p.drawText(QRectF(plot.left(), h - 22, plot.width(), 18), Qt.AlignmentFlag.AlignCenter,
                   self._spec.get("x_label", ""))
        p.save()
        p.translate(14, plot.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-plot.height() / 2, -10, plot.height(), 20), Qt.AlignmentFlag.AlignCenter,
                   self._spec.get("y_label", ""))
        p.restore()

        x = plot.right()
        for series in reversed(self._spec["series"]):
            label = series.get("label")
            if not label:
                continue
            width = p.fontMetrics().horizontalAdvance(label)
            x -= width
            p.setPen(QColor(theme.TEXT_MUTED))
            p.drawText(QRectF(x, 12, width + 2, 18), Qt.AlignmentFlag.AlignVCenter, label)
            p.setBrush(QColor(theme.palette(series.get("color", "accent"))))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x - 9, 21), 4, 4)
            x -= 26


class BarList(QWidget):
    """Horizontal bars, one row per item.

    Spec: ``{"items": [(label, value, text), ...]}`` where value is 0..1 (bar
    fill) and text is the already-formatted number shown at the right.
    """

    ROW_H = 26
    LABEL_W = 210

    def __init__(self, spec: dict, parent=None) -> None:
        super().__init__(parent)
        self._items = spec["items"]
        self.setFixedHeight(max(1, len(self._items)) * self.ROW_H + 8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.SURFACE))
        p.setFont(QFont(theme.FONT_FAMILY, theme.PLOT_POINT_SIZE))
        text_w = 56
        bar_left = self.LABEL_W + 12
        bar_w = max(10, self.width() - bar_left - text_w - 16)
        top = max((v for _, v, _ in self._items), default=1) or 1  # scale to the largest bar
        for i, (label, value, text) in enumerate(self._items):
            y = 4 + i * self.ROW_H
            p.setPen(QColor(theme.TEXT))
            elided = p.fontMetrics().elidedText(label, Qt.TextElideMode.ElideRight, self.LABEL_W)
            p.drawText(QRectF(12, y, self.LABEL_W, self.ROW_H), Qt.AlignmentFlag.AlignVCenter, elided)
            track = QRectF(bar_left, y + 8, bar_w, 10)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.BORDER))
            p.drawRoundedRect(track, 5, 5)
            if value > 0:
                p.setBrush(QColor(theme.ACCENT))
                p.drawRoundedRect(QRectF(track.left(), track.top(), max(4.0, track.width() * value / top), track.height()), 5, 5)
            p.setPen(QColor(theme.TEXT_MUTED))
            p.drawText(QRectF(bar_left + bar_w + 8, y, text_w, self.ROW_H),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text)
