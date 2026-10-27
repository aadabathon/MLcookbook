"""Native graph drawing and gestures. All edits leave as plain-data signals."""
import weakref

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from gui import theme

NODE_LABELS = {"dataset": "Dataset snapshot", "select": "Select columns", "comparison": "Model comparison", "report": "Report"}
NODE_SYMBOLS = {"dataset": "D", "select": "π", "comparison": "⇄", "report": "Σ"}
NODE_PORTS = {"dataset": (None, "data"), "select": ("data", "data"),
              "comparison": ("data", "result"), "report": ("result", None)}


class PortItem(QGraphicsEllipseItem):
    def __init__(self, node, name, output):
        super().__init__(-6, -6, 12, 12, node)
        self._node, self.name, self.output = weakref.ref(node), name, output
        self.setPos(210 if output else 0, 65)
        color = theme.UI_ACCENT if name == "data" else theme.PLOT_ALT
        self.setBrush(QColor(color))
        self.setPen(QPen(QColor(theme.BG), 2))
        self.setToolTip(("Output: " if output else "Input: ") + ("dataset" if name == "data" else "comparison result"))
        self.setCursor(Qt.CursorShape.CrossCursor)

    @property
    def node(self):
        return self._node()


class NodeItem(QGraphicsRectItem):
    def __init__(self, node, canvas):
        super().__init__(0, 0, 210, 104)
        self.node, self._canvas, self.state = node, weakref.ref(canvas), "unrun"
        self.ports = {}
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setToolTip(f"{node['id']} · {NODE_LABELS[node['kind']]}")
        incoming, outgoing = NODE_PORTS[node["kind"]]
        for name, output in ((incoming, False), (outgoing, True)):
            if name:
                self.ports[output] = PortItem(self, name, output)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(theme.UI_ACCENT if self.isSelected() else theme.BORDER), 2 if self.isSelected() else 1))
        painter.setBrush(QColor(theme.CARD))
        painter.drawRoundedRect(self.rect(), 5, 5)
        painter.setFont(QFont("Palatino Linotype", 13))
        painter.setPen(QColor(theme.TEXT))
        painter.drawText(QRectF(14, 10, 185, 30), Qt.AlignmentFlag.AlignLeft,
                         NODE_SYMBOLS[self.node["kind"]] + "  " + NODE_LABELS[self.node["kind"]])
        painter.setFont(QFont(theme.FONT_FAMILY, theme.PLOT_POINT_SIZE))
        painter.setPen(QColor(theme.WARNING if self.state in ("stale", "failed", "blocked", "cancelled") else
                              theme.UI_ACCENT if self.state == "running" else theme.TEXT_MUTED))
        painter.drawText(QRectF(15, 43, 180, 22), self.state.capitalize())
        painter.setPen(QColor(theme.TEXT_FAINT))
        painter.drawText(QRectF(15, 73, 180, 20), self.node["id"])

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            return QPointF(max(-100_000, min(100_000, value.x())), max(-100_000, min(100_000, value.y())))
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            canvas = self._canvas()
            if canvas:
                canvas.update_edges()
        return super().itemChange(change, value)


class KitchenCanvas(QGraphicsView):
    selected = Signal(object)
    moved = Signal(dict)
    connect_requested = Signal(dict)
    delete_requested = Signal()
    message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setSceneRect(-1000, -800, 5000, 3000)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Kitchen workflow canvas")
        self.setAccessibleDescription("Left and right select nodes. Delete removes the selection. Drag nodes to move; drag the background to pan. Control plus wheel zooms.")
        self.setToolTip("Connect an output circle to an input circle. Drag the background to pan; Ctrl+wheel to zoom.")
        self.scene().selectionChanged.connect(self._selection)
        self.nodes, self.edges, self._link = {}, [], None
        self._loading = False
        self._positions = {}

    def set_graph(self, graph, layout, selected=None, states=None):
        self._loading = True
        self._cancel_link()
        self.nodes, self.edges = {}, []
        self.scene().clear()
        for index, node in enumerate(graph["nodes"]):
            item = NodeItem(node, self)
            self.nodes[node["id"]] = item
            self.scene().addItem(item)
            item.setPos(*layout.get(node["id"], [index * 255, 0]))
            item.state = (states or {}).get(node["id"], "unrun")
            item.setSelected(node["id"] == selected)
        for index, edge in enumerate(graph["edges"]):
            line = QGraphicsPathItem()
            line.setPen(QPen(QColor(theme.UI_ACCENT if edge["source_port"] == "data" else theme.PLOT_ALT), 2))
            line.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            line.setData(0, index)
            line.setZValue(-1)
            self.scene().addItem(line)
            self.edges.append((line, edge))
        self.update_edges()
        self._positions = self.positions()
        self._loading = False

    @staticmethod
    def curve(start, end):
        path = QPainterPath(start)
        distance = min(80, max(40, abs(end.x() - start.x()) * 0.5))
        path.cubicTo(start + QPointF(distance, 0), end - QPointF(distance, 0), end)
        return path

    def update_edges(self):
        for line, edge in self.edges:
            source, target = self.nodes.get(edge["source"]), self.nodes.get(edge["target"])
            if source and target:
                line.setPath(self.curve(source.ports[True].scenePos(), target.ports[False].scenePos()))
        bounds = self.scene().itemsBoundingRect().adjusted(-400, -300, 400, 300)
        self.setSceneRect(bounds.united(QRectF(-1000, -800, 5000, 3000)))

    def positions(self):
        return {key: [item.pos().x(), item.pos().y()] for key, item in self.nodes.items()}

    def set_states(self, states):
        for key, item in self.nodes.items():
            item.state = states.get(key, "unrun")
            item.update()

    def selected_edges(self):
        return [line.data(0) for line, _ in self.edges if line.isSelected()]

    def _selection(self):
        if not self._loading:
            selected = sorted(key for key, item in self.nodes.items() if item.isSelected())
            self.selected.emit(selected[0] if selected else None)

    def fit_graph(self):
        if self.nodes:
            # Fit drawn curves rather than their much wider Bezier control hull.
            bounds = QRectF()
            for item in self.nodes.values():
                bounds = bounds.united(item.sceneBoundingRect().adjusted(-8, 0, 8, 0))
            for line, _ in self.edges:
                bounds = bounds.united(line.path().boundingRect())
            bounds = bounds.adjusted(-30, -40, 30, 40)
            self.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)
            if self.transform().m11() > 1:
                self.resetTransform()
                self.centerOn(bounds.center())
        else:
            self.resetTransform()
            self.centerOn(200, 0)

    def _cancel_link(self):
        if self._link:
            self.scene().removeItem(self._link[1])
            self._link = None

    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and isinstance(item, PortItem):
            if item.output:
                self._cancel_link()
                line = QGraphicsPathItem()
                line.setPen(QPen(QColor(theme.UI_ACCENT), 2, Qt.PenStyle.DashLine))
                line.setZValue(-0.5)
                self.scene().addItem(line)
                self._link = item, line
                self.message.emit("Choose a matching input port. Escape cancels the connection.")
            elif self._link:
                self._finish_link(item)
            event.accept()
            return
        super().mousePressEvent(event)

    def _finish_link(self, port):
        source = self._link[0]
        edge = {"source": source.node.node["id"], "source_port": source.name,
                "target": port.node.node["id"], "target_port": port.name}
        self._cancel_link()
        self.connect_requested.emit(edge)

    def mouseMoveEvent(self, event):
        if self._link:
            self._link[1].setPath(self.curve(self._link[0].scenePos(), self.mapToScene(event.position().toPoint())))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if self._link and isinstance(item, PortItem) and not item.output:
            self._finish_link(item)
        super().mouseReleaseEvent(event)
        positions = self.positions()
        if positions != self._positions:
            self._positions = positions
            self.moved.emit(positions)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._cancel_link()
            event.accept()
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_requested.emit()
            event.accept()
        elif event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right) and self.nodes:
            keys = list(self.nodes)
            current = next((i for i, key in enumerate(keys) if self.nodes[key].isSelected()), -1)
            key = keys[(current + (-1 if event.key() == Qt.Key.Key_Left else 1)) % len(keys)]
            self.scene().clearSelection()
            self.nodes[key].setSelected(True)
            self.ensureVisible(self.nodes[key])
            event.accept()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            current = self.transform().m11()
            zoom = max(0.1, min(2.5, current * factor))
            self.scale(zoom / current, zoom / current)
            event.accept()
        else:
            super().wheelEvent(event)
