"""Native graph drawing and gestures. All edits leave as plain-data signals."""
import weakref

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
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


def connection_error(graph, edge):
    """The editor's wiring rules, shared by previews and committed edits."""
    nodes = {node["id"]: node for node in graph["nodes"]}
    if edge["source"] not in nodes or edge["target"] not in nodes:
        return "Choose two modules in this graph."
    output = NODE_PORTS[nodes[edge["source"]]["kind"]][1].get(edge["source_port"])
    incoming = NODE_PORTS[nodes[edge["target"]]["kind"]][0].get(edge["target_port"])
    if not output or output != incoming:
        return "Incompatible ports: connect datasets, vectors or analysis results to the same input type."
    if any(item["target"] == edge["target"] and item["target_port"] == edge["target_port"] for item in graph["edges"]):
        return "That input is already connected. Select its connection and Delete it first."
    children = {}
    for existing in graph["edges"]:
        children.setdefault(existing["source"], []).append(existing["target"])
    pending, seen = [edge["target"]], set()
    while pending:
        key = pending.pop()
        if key == edge["source"]:
            return "This connection would create a cycle."
        if key not in seen:
            seen.add(key)
            pending.extend(children.get(key, []))
    if len(graph["edges"]) >= 200:
        return "This Kitchen supports at most 200 connections."
    return ""


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
        self._node, self.name, self.output, self.kind = weakref.ref(node), name, output, kind
        self.setPos(node.rect().width() if output else 0, 82 + index * 26)
        self.set_hint()
        self.setToolTip(("Output " if output else "Input ") + name + " · " + kind)
        self.setCursor(Qt.CursorShape.CrossCursor)

    @property
    def node(self):
        return self._node()

    def set_hint(self, state="normal"):
        self.highlighted = state in ("available", "target")
        radius = 10 if state == "target" else 8 if state == "available" else 6
        self.setRect(-radius, -radius, radius * 2, radius * 2)
        self.setBrush(QColor(PORT_COLORS[self.kind]))
        self.setPen(QPen(QColor(theme.TEXT if state == "target" else theme.BG), 2))
        self.setOpacity(.25 if state == "unavailable" else 1)


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
    connect_mode_changed = Signal(bool)
    SNAP_DISTANCE = 16  # viewport pixels, independent of graph zoom

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setSceneRect(-1000, -800, 5000, 3000)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName("Kitchen workflow canvas")
        self.setAccessibleDescription("Connect enables module wiring. Drag or click matching output and input ports. Escape cancels. Left and right select nodes; Delete removes the selection. Drag the background to pan; Control plus wheel zooms.")
        self.setToolTip("Drag between ports, or use Connect to link modules. Drag the background to pan; Ctrl+wheel to zoom.")
        self.scene().selectionChanged.connect(self._selection)
        self.nodes, self.edges, self._link = {}, [], None
        self._loading = False
        self._positions = {}
        self._captions = {}
        self._graph = {"nodes": [], "edges": []}
        self.connect_mode = False
        self._valid_targets = []
        self._target_port = None
        self._press_position = None
        self._dragging = False

    def set_graph(self, graph, layout, selected=None, states=None):
        self._loading = True
        self._cancel_link()
        self._graph = graph
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
        self._port_hints()

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

    def set_connect_mode(self, enabled):
        enabled = bool(enabled) and self.isEnabled()
        changed, pending = self.connect_mode != enabled, self._link is not None
        self._cancel_link()
        if changed:
            self.connect_mode = enabled
            self.connect_mode_changed.emit(enabled)
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self._port_hints()
        if changed or pending:
            self.message.emit("Click or drag from a module to a matching input. Escape returns to moving modules." if enabled else "")

    def _ports(self):
        return [port for node in self.nodes.values() for ports in (node.input_ports, node.output_ports) for port in ports.values()]

    def _port_hints(self):
        for port in self._ports():
            if port is self._target_port:
                state = "target"
            elif self._link:
                state = "available" if port is self._link[0] or port in self._valid_targets else "unavailable"
            else:
                state = "available" if self.connect_mode and port.output else "normal"
            port.set_hint(state)

    def _cancel_link(self):
        if self._link is not None:
            self.scene().removeItem(self._link[1])
            self._link = None
        self._valid_targets = []
        self._target_port = None
        self._press_position = None
        self._dragging = False
        self._port_hints()

    def _port_near(self, position):
        candidates = []
        for port in self._ports():
            offset = self.mapFromScene(port.scenePos()) - position
            distance = offset.x() ** 2 + offset.y() ** 2
            if distance <= self.SNAP_DISTANCE ** 2:
                candidates.append((distance, port))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _node_at(self, position):
        for item in self.items(position):
            if isinstance(item, PortItem):
                return item.node
            if isinstance(item, NodeItem):
                return item
        return None

    @staticmethod
    def _edge(source, target):
        return {"source": source.node.node["id"], "source_port": source.name,
                "target": target.node.node["id"], "target_port": target.name}

    def _begin_link(self, source, position):
        self._cancel_link()
        self._valid_targets = [port for port in self._ports() if not port.output and
                               not connection_error(self._graph, self._edge(source, port))]
        if not self._valid_targets:
            self.message.emit("No free compatible inputs. Add a matching module or remove an existing connection.")
            return
        line = QGraphicsPathItem()
        line.setPen(QPen(QColor(PORT_COLORS[source.kind]), 2, Qt.PenStyle.DashLine))
        line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        line.setZValue(-.5)
        line.hide()
        self.scene().addItem(line)
        self._link = source, line
        self._press_position = position
        self._port_hints()
        self.message.emit(f"Connect {source.name} ({source.kind}) to a highlighted input. Escape cancels.")

    def _resolve_target(self, position):
        port = self._port_near(position)
        if port is not None:
            if port.output:
                return None, "Choose an input port."
            return (port, "") if port in self._valid_targets else (None, connection_error(self._graph, self._edge(self._link[0], port)))
        node = self._node_at(position)
        if node is None:
            return None, ""
        choices = [port for port in self._valid_targets if port.node is node]
        if len(choices) == 1:
            return choices[0], ""
        if choices:
            return None, "Several inputs match. Choose a highlighted input port."
        return None, "This module has no free compatible input."

    def mousePressEvent(self, event):
        position = event.position().toPoint()
        if event.button() == Qt.MouseButton.RightButton:
            self._cancel_link()
        if event.button() == Qt.MouseButton.LeftButton:
            if self._link:
                target, reason = self._resolve_target(position)
                if target:
                    self._finish_link(target)
                    event.accept()
                    return
                port, node = self._port_near(position), self._node_at(position)
                if port is not None and port.output:
                    self._begin_link(port, position)
                elif node is not None:
                    self._press_position, self._dragging = position, False
                    self._target_port = None
                    self._link[1].hide()
                    self._port_hints()
                    self.message.emit(reason)
                else:
                    self._cancel_link()
                    self.message.emit("")
                event.accept()
                return
            port = self._port_near(position)
            node = self._node_at(position)
            if port is not None and port.output:
                self._begin_link(port, position)
            elif self.connect_mode and node is not None:
                if len(node.output_ports) == 1:
                    self._begin_link(next(iter(node.output_ports.values())), position)
                else:
                    self.message.emit("Choose an output port on this module." if node.output_ports else "This module has no output.")
            else:
                super().mousePressEvent(event)
                return
            event.accept()
            return
        super().mousePressEvent(event)

    def _finish_link(self, port):
        edge = self._edge(self._link[0], port)
        self._cancel_link()
        self.connect_requested.emit(edge)

    def mouseMoveEvent(self, event):
        if self._link:
            position = event.position().toPoint()
            if event.buttons() & Qt.MouseButton.LeftButton and (position - self._press_position).manhattanLength() >= QApplication.startDragDistance():
                self._dragging = True
            target, _ = self._resolve_target(position)
            if target is not self._target_port:
                self._target_port = target
                self._port_hints()
            end = target.scenePos() if target else self.mapToScene(position)
            self._link[1].setPath(self.curve(self._link[0].scenePos(), end))
            self._link[1].setVisible(self._dragging or target is not None)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._link and event.button() == Qt.MouseButton.LeftButton:
            position = event.position().toPoint()
            target, reason = self._resolve_target(position)
            if target:
                self._finish_link(target)
            elif self._dragging or (position - self._press_position).manhattanLength() >= QApplication.startDragDistance():
                self._cancel_link()
                self.message.emit(reason)
            else:
                self._link[1].hide()
            event.accept()
            return
        super().mouseReleaseEvent(event)
        positions = self.positions()
        if positions != self._positions:
            self._positions = positions
            self.moved.emit(positions)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.set_connect_mode(False)
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

    def changeEvent(self, event):
        if event.type() == QEvent.Type.EnabledChange and not self.isEnabled():
            self._cancel_link()
            if self.connect_mode:
                self.connect_mode = False
                self.connect_mode_changed.emit(False)
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().changeEvent(event)

    def hideEvent(self, event):
        self._cancel_link()
        super().hideEvent(event)

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
