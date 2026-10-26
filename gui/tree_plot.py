"""Node-link diagram of a fitted decision tree, drawn with QPainter.

Purely presentational: renders a plain ``spec`` dict handed in by app.py.

    {
        "nodes": [  # nodes[0] is the root; left/right are indices into this list
            {"title": "mean radius ≤ 14.6", "subtitle": "n=426 · gini 0.47",
             "fraction": 0.37,        # share of the node's rows in the positive class
             "positive": False,       # majority class is the positive one
             "left": 1, "right": 2},  # both None for a leaf
            ...
        ],
        "negative_label": "benign", "positive_label": "malignant",
    }

Colors: teal = negative class, violet = positive class (ReportScreen draws the legend).
A node may instead carry ``"color"`` (a theme palette name such as "c2"): its bar
and leaf outline then use that color, e.g. the predicted class of a multiclass
tree or a single accent for regression trees, whose bar shows the predicted value.

Layout: leaves are spaced evenly left to right, each parent centers over its
children, depth runs top to bottom. Boxes show the split rule, sample count,
and a bar of the class mix; leaves are outlined in their predicted class color.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QWidget

from gui import theme

BOX_W, BOX_H = 200, 58
H_GAP, V_GAP = 16, 56
MARGIN = 24
LEGEND_H = 0  # the legend lives outside the scroller (see ReportScreen)


class TreeWidget(QWidget):
    def __init__(self, spec: dict, parent=None) -> None:
        super().__init__(parent)
        self._spec = spec
        self._nodes = spec["nodes"]
        self._pos: dict[int, tuple[float, int]] = {}  # node index -> (center x, depth)
        self._leaf_count = 0
        self._max_depth = 0
        self._layout(0, 0)
        self.setFixedSize(
            int(MARGIN * 2 + self._leaf_count * BOX_W + max(0, self._leaf_count - 1) * H_GAP),
            int(LEGEND_H + MARGIN + (self._max_depth + 1) * BOX_H + self._max_depth * V_GAP + MARGIN),
        )

    def _layout(self, index: int, depth: int) -> float:
        node = self._nodes[index]
        self._max_depth = max(self._max_depth, depth)
        if node["left"] is None:
            x = MARGIN + self._leaf_count * (BOX_W + H_GAP) + BOX_W / 2
            self._leaf_count += 1
        else:
            x = (self._layout(node["left"], depth + 1) + self._layout(node["right"], depth + 1)) / 2
        self._pos[index] = (x, depth)
        return x

    def _box(self, index: int) -> QRectF:
        x, depth = self._pos[index]
        y = LEGEND_H + MARGIN + depth * (BOX_H + V_GAP)
        return QRectF(x - BOX_W / 2, y, BOX_W, BOX_H)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.SURFACE))
        neg, pos = QColor(theme.PLOT_ALT), QColor(theme.ACCENT)
        small = QFont(theme.FONT_FAMILY, theme.PLOT_POINT_SIZE)

        # Edges first, so boxes sit on top of them.
        for i, node in enumerate(self._nodes):
            if node["left"] is None:
                continue
            parent = self._box(i)
            for child, label in ((node["left"], "≤"), (node["right"], ">")):
                box = self._box(child)
                start = QPointF(parent.center().x(), parent.bottom())
                end = QPointF(box.center().x(), box.top())
                p.setPen(QPen(QColor(theme.SCROLL_HANDLE), 1.5))
                p.drawLine(start, end)
                p.setFont(small)
                p.setPen(QColor(theme.TEXT_MUTED))
                mid = QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)
                p.drawText(QRectF(mid.x() - 10, mid.y() - 9, 20, 18), Qt.AlignmentFlag.AlignCenter, label)

        for i, node in enumerate(self._nodes):
            box = self._box(i)
            leaf = node["left"] is None
            color = QColor(theme.palette(node["color"])) if "color" in node else None
            outline = color if color is not None else (pos if node["positive"] else neg)
            p.setBrush(QColor(theme.CARD))
            p.setPen(QPen(outline if leaf else QColor(theme.BORDER), 2 if leaf else 1))
            p.drawRoundedRect(box, 8, 8)

            text_w = int(BOX_W - 16)
            p.setPen(QColor(theme.TEXT))
            p.setFont(QFont(theme.FONT_FAMILY, theme.PLOT_POINT_SIZE))
            title = QFontMetrics(p.font()).elidedText(node["title"], Qt.TextElideMode.ElideRight, text_w)
            p.drawText(QRectF(box.left() + 8, box.top() + 6, text_w, 18), Qt.AlignmentFlag.AlignVCenter, title)
            p.setFont(small)
            p.setPen(QColor(theme.TEXT_MUTED))
            p.drawText(QRectF(box.left() + 8, box.top() + 24, text_w, 16), Qt.AlignmentFlag.AlignVCenter,
                       node["subtitle"])

            # Class-mix bar: negative color, then the positive share in violet.
            bar = QRectF(box.left() + 8, box.bottom() - 13, text_w, 6)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.BORDER) if color is not None else neg)
            p.drawRoundedRect(bar, 3, 3)
            share = max(0.0, min(1.0, node["fraction"]))
            if share > 0:
                p.setBrush(color if color is not None else pos)
                p.drawRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * share, bar.height()), 3, 3)
