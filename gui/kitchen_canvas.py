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

NODE_LABELS = {"dataset": "Dataset snapshot", "select": "Select columns", "column": "Column vector",
               "map": "Vector map", "combine": "Combine vectors", "build": "Build dataset", "add_column": "Add column",
               "filter": "Filter rows", "linear": "Linear regression", "comparison": "Model comparison",
               "summary": "Vector summary", "correlation": "Correlation", "report": "Report"}
NODE_SYMBOLS = {"dataset": "D", "select": "π", "column": "v", "map": "ƒ", "combine": "⊕", "build": "⊞",
                "add_column": "+", "filter": "σ", "linear": "β", "comparison": "⇄", "summary": "μ", "correlation": "ρ", "report": "Σ"}
NODE_PORTS = {
    "dataset": ({}, {"data": "dataset"}), "select": ({"data": "dataset"}, {"data": "dataset"}),
    "column": ({"data": "dataset"}, {"vector": "vector", "missing": "vector"}),
    "map": ({"vector": "vector"}, {"vector": "vector"}),
    "combine": ({"left": "vector", "right": "vector"}, {"vector": "vector"}),
    "build": ({"vector": "vector"}, {"data": "dataset"}),
    "add_column": ({"data": "dataset", "vector": "vector"}, {"data": "dataset"}),
    "filter": ({"data": "dataset", "mask": "vector"}, {"data": "dataset"}),
    "linear": ({"data": "dataset"}, {"result": "analysis"}),
    "comparison": ({"data": "dataset"}, {"result": "analysis"}),
    "summary": ({"vector": "vector"}, {"result": "analysis"}),
    "correlation": ({"left": "vector", "right": "vector"}, {"result": "analysis"}),
    "report": ({"result": "analysis"}, {}),
}
PORT_COLORS = {"dataset": theme.UI_ACCENT, "vector": theme.PLOT_ALT, "analysis": theme.ACCENT_SOFT}


def node_caption(node):
    """A short description of the configured operation, without analytical work."""
    kind, settings = node["kind"], node["settings"]
    if kind == "dataset":
        return "Dataset bound" if settings["snapshot"] else "No dataset bound"
    if kind == "column":
        return settings["column"] or "No column selected"
    if kind == "select":
        return ", ".join(settings["columns"]) or "No columns selected"
    if kind in ("build", "add_column"):
        return settings["name"] or "Column name required"
    if kind in ("linear", "comparison"):
        return (settings["target"] + " ← " + ", ".join(settings["features"])) if settings["target"] else "No target selected"
    if kind == "map":
        operation, value = settings["operation"], settings["value"]
        constants = {"add": "+", "multiply": "×", "greater_than": ">", "less_than": "<", "equal": "="}
        return f"vector {constants[operation]} {value:g}" if operation in constants else f"{operation}(vector)"
    if kind == "combine":
        symbol = {"add": "+", "subtract": "−", "multiply": "×", "divide": "/", "greater_than": ">",
                  "less_than": "<", "equal": "="}[settings["operation"]]
        return f"left {symbol} right"
    if kind == "correlation":
        return f"{settings['method'].capitalize()} · {settings['confidence_level']:.0%} confidence"
    return {"filter": "Keep rows where mask is true", "summary": "Describe connected vector",
            "report": "Inspect connected analysis"}.get(kind, "")


class PortItem(QGraphicsEllipseItem):
    def __init__(self, node, name, kind, output, index):
        super().__init__(-6, -6, 12, 12, node)
        self._node, self.name, self.output = weakref.ref(node), name, output
        self.setPos(node.rect().width() if output else 0, 82 + index * 26)
        color = PORT_COLORS[kind]
        self.setBrush(QColor(color))
        self.setPen(QPen(QColor(theme.BG), 2))
        self.setToolTip(("Output " if output else "Input ") + name + " · " + kind)
        self.setCursor(Qt.CursorShape.CrossCursor)

    @property
    def node(self):
        return self._node()


class NodeItem(QGraphicsRectItem):
    def __init__(self, node, canvas):
        incoming, outgoing = NODE_PORTS[node["kind"]]
        super().__init__(0, 0, 230, 126 + 26 * max(0, max(len(incoming), len(outgoing)) - 1))
        self.node, self._canvas, self.state = node, weakref.ref(canvas), "unrun"
        self.caption = node_caption(node)
        self.ports, self.input_ports, self.output_ports = {}, {}, {}
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.set_caption(self.caption)
        for definitions, output in ((incoming, False), (outgoing, True)):
            mapping = self.output_ports if output else self.input_ports
            for index, (name, kind) in enumerate(definitions.items()):
                mapping[name] = PortItem(self, name, kind, output, index)
                self.ports.setdefault(output, mapping[name])  # first-port compatibility for gestures/tests

    def set_caption(self, caption):
        self.caption = caption
        self.setToolTip(f"{self.node['id']} · {NODE_LABELS[self.node['kind']]}\n{caption}")
        self.update()

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(theme.UI_ACCENT if self.isSelected() else theme.BORDER), 2 if self.isSelected() else 1))
        painter.setBrush(QColor(theme.CARD))
        painter.drawRoundedRect(self.rect(), 5, 5)
        painter.setFont(QFont("Palatino Linotype", 13))
        painter.setPen(QColor(theme.TEXT))
        title = NODE_SYMBOLS[self.node["kind"]] + "  " + NODE_LABELS[self.node["kind"]]
        painter.drawText(QRectF(14, 8, 202, 27), Qt.AlignmentFlag.AlignLeft,
                         painter.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, 202))
        painter.setFont(QFont(theme.FONT_FAMILY, theme.PLOT_POINT_SIZE))
        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.drawText(QRectF(14, 38, 202, 20),
                         painter.fontMetrics().elidedText(self.caption, Qt.TextElideMode.ElideRight, 202))
        painter.setPen(QColor(theme.WARNING if self.state in ("stale", "failed", "blocked", "cancelled") else
                              theme.UI_ACCENT if self.state == "running" else theme.TEXT_MUTED))
        painter.drawText(QRectF(14, self.rect().height() - 22, 202, 18),
                         f"{self.node['id']} · {self.state.capitalize()}")
        painter.setPen(QColor(theme.TEXT_FAINT))
        for mapping, output in ((self.input_ports, False), (self.output_ports, True)):
            for name, port in mapping.items():
                rect = QRectF(14 if not output else 110, port.pos().y() - 8, 106, 20)
                painter.drawText(rect, Qt.AlignmentFlag.AlignRight if output else Qt.AlignmentFlag.AlignLeft, name)

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
        self._captions = {}

    def set_graph(self, graph, layout, selected=None, states=None):
        self._loading = True
        self._cancel_link()
        self.nodes, self.edges = {}, []
        self.scene().clear()
        for index, node in enumerate(graph["nodes"]):
            item = NodeItem(node, self)
            self.nodes[node["id"]] = item
            self.scene().addItem(item)
            item.set_caption(self._captions.get(node["id"], node_caption(node)))
            item.setPos(*layout.get(node["id"], [index * 255, 0]))
            item.state = (states or {}).get(node["id"], "unrun")
            item.setSelected(node["id"] == selected)
        for index, edge in enumerate(graph["edges"]):
            line = QGraphicsPathItem()
            kind = NODE_PORTS[self.nodes[edge["source"]].node["kind"]][1][edge["source_port"]]
            line.setPen(QPen(QColor(PORT_COLORS[kind]), 2))
            line.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            line.setData(0, index)
            line.setZValue(-1)
            self.scene().addItem(line)
            self.edges.append((line, edge))
        self.update_edges()
        self._positions = self.positions()
        self._loading = False

    def update_nodes(self, nodes):
        """Refresh edited captions without replacing items or losing selection."""
        for node in nodes:
            item = self.nodes.get(node["id"])
            if item is not None:
                item.node = node
                item.set_caption(self._captions.get(node["id"], node_caption(node)))

    def set_captions(self, captions):
        self._captions = dict(captions)
        self.update_nodes([item.node for item in self.nodes.values()])

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
                line.setPath(self.curve(source.output_ports[edge["source_port"]].scenePos(), target.input_ports[edge["target_port"]].scenePos()))
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
            backward = event.key() == Qt.Key.Key_Left
            position = (-1 if backward else 0) if current < 0 else (current + (-1 if backward else 1)) % len(keys)
            key = keys[position]
            self.scene().clearSelection()
            self.nodes[key].setSelected(True)
            self.ensureVisible(self.nodes[key])
            event.accept()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if not event.angleDelta().y():
                event.accept()
                return
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            current = self.transform().m11()
            zoom = max(min(0.001, current), min(2.5, current * factor))
            self.scale(zoom / current, zoom / current)
            event.accept()
        else:
            super().wheelEvent(event)
