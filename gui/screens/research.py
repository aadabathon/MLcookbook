"""Offline research chronology: plain catalog data in, explicit reading actions out."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from gui import theme


def _label(text: str = "", name: str = "") -> QLabel:
    widget = QLabel(text)
    widget.setObjectName(name)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(True)
    widget.setMinimumWidth(0)
    widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return widget


class _TimelineDelegate(QStyledItemDelegate):
    """Paint a chronology rail while retaining native list selection and navigation."""

    def __init__(self, entries, parent=None):
        super().__init__(parent)
        self.entries = entries

    def _layout(self, option, index):
        entry = self.entries[index.data(Qt.ItemDataRole.UserRole)]
        width = max(80, option.rect.width() - 127)
        metrics = option.fontMetrics
        title = metrics.boundingRect(QRectF(0, 0, width, 1000).toRect(),
                                     Qt.TextFlag.TextWordWrap, entry["title"])
        summary = metrics.boundingRect(QRectF(0, 0, width, 1000).toRect(),
                                       Qt.TextFlag.TextWordWrap, entry["summary"])
        return entry, width, title.height(), summary.height()

    def sizeHint(self, option, index):
        option = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)
        if self.parent() is not None:
            option.rect.setWidth(self.parent().viewport().width())
        _, _, title_height, summary_height = self._layout(option, index)
        return QSize(240, max(100, title_height + summary_height + 46))

    def paint(self, painter, option, index):
        from PySide6.QtWidgets import QStyle

        self.initStyleOption(option, index)
        entry, width, title_height, summary_height = self._layout(option, index)
        rect = QRectF(option.rect)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.save()
        painter.setClipRect(rect)
        painter.fillRect(rect, QColor(theme.UI_SELECTION if selected else theme.BG))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rail_x = rect.left() + 102
        painter.setPen(QPen(QColor(theme.BORDER), 2))
        painter.drawLine(QPointF(rail_x, rect.top()), QPointF(rail_x, rect.bottom()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.UI_ACCENT if selected else theme.TEXT_FAINT))
        painter.drawEllipse(QPointF(rail_x, rect.top() + 23), 4, 4)
        painter.setFont(option.font)
        painter.setPen(QColor(theme.UI_ACCENT if selected else theme.TEXT_MUTED))
        painter.drawText(QRectF(rect.left() + 8, rect.top() + 12, 80, 50),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                         entry["date"])
        text_left = rect.left() + 118
        painter.setPen(QColor(theme.TEXT))
        painter.drawText(QRectF(text_left, rect.top() + 12, width, title_height + 3),
                         Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, entry["title"])
        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.drawText(QRectF(text_left, rect.top() + title_height + 20, width, summary_height + 3),
                         Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, entry["summary"])
        if option.state & QStyle.StateFlag.State_HasFocus:
            painter.setPen(QPen(QColor(theme.UI_BORDER), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect.adjusted(.5, .5, -.5, -.5))
        painter.restore()


class ResearchScreen(QWidget):
    """Browse supplied history and readings without opening files or connections."""

    source_requested = Signal(str)
    paper_requested = Signal(str)
    library_requested = Signal()

    def __init__(self, entries: list[dict], papers: list[dict], parent=None):
        super().__init__(parent)
        self.entries = deepcopy(entries)
        self._entries = {entry["id"]: entry for entry in self.entries}
        self.papers = deepcopy(papers)
        self._papers = {paper["id"]: paper for paper in self.papers}
        self._preferred_selection = self.entries[0]["id"] if self.entries else None
        self._rendering = False
        self._busy = False
        self._status_message = ""
        self._directory = ""
        self._pending_scroll = 0
        self._scroll_restore = QTimer(self)
        self._scroll_restore.setSingleShot(True)
        self._scroll_restore.timeout.connect(self._apply_scroll)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_MD)
        outer.setSpacing(theme.SPACE_SM)
        heading = QHBoxLayout()
        heading.addWidget(_label("Research", "PageTitle"), stretch=1)
        self.library_button = QPushButton("Papers folder…")
        self.library_button.clicked.connect(self.library_requested)
        heading.addWidget(self.library_button)
        outer.addLayout(heading)

        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search ideas, people, papers or connections")
        self.search.setAccessibleName("Search research timeline")
        self.search.setMinimumWidth(0)
        filters.addWidget(self.search, stretch=1)
        self.era = QComboBox()
        self.era.setAccessibleName("Historical era")
        self.era.setMinimumWidth(0)
        self.era.setMaximumWidth(195)
        self.era.addItem("All eras", "")
        for era in dict.fromkeys(entry["era"] for entry in self.entries):
            self.era.addItem(era, era)
        filters.addWidget(self.era)
        outer.addLayout(filters)
        context = QHBoxLayout()
        self.papers_only = QCheckBox("Linked readings")
        self.papers_only.setToolTip("Show milestones connected to readings in this catalog, including unavailable local files.")
        context.addWidget(self.papers_only)
        self.count = _label("", "HelpText")
        context.addWidget(self.count, stretch=1)
        outer.addLayout(context)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        timeline_panel = QWidget()
        timeline_panel.setMinimumWidth(270)
        timeline_layout = QVBoxLayout(timeline_panel)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(theme.SPACE_XS)
        self.timeline = QListWidget()
        self.timeline.setAccessibleName("Chronological AI research timeline")
        self.timeline.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.timeline.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.timeline.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.timeline.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.timeline.setMinimumWidth(0)
        self.timeline.setStyleSheet(f"QListWidget {{ border: none; background: {theme.BG}; outline: 0; }}")
        self.timeline.setItemDelegate(_TimelineDelegate(self._entries, self.timeline))
        timeline_layout.addWidget(self.timeline, stretch=1)
        self.empty = _label("No milestones match. Clear the search or choose another era.", "HelpText")
        timeline_layout.addWidget(self.empty)
        self.splitter.addWidget(timeline_panel)

        self.detail_scroll = QScrollArea()
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.detail_scroll.setMinimumWidth(270)
        self.detail_scroll.setAccessibleName("Selected milestone details and sources")
        self.detail = QWidget()
        self.detail.setMinimumWidth(0)
        self.detail_layout = QVBoxLayout(self.detail)
        self.detail_layout.setContentsMargins(theme.SPACE_MD, 0, theme.SPACE_XS, theme.SPACE_MD)
        self.detail_layout.setSpacing(theme.SPACE_SM)
        self.detail_scroll.setWidget(self.detail)
        self.splitter.addWidget(self.detail_scroll)
        self.splitter.setSizes([470, 390])
        outer.addWidget(self.splitter, stretch=1)

        footer = QFrame()
        footer.setObjectName("ActionBar")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(0, theme.SPACE_SM, 0, 0)
        footer_layout.setSpacing(theme.SPACE_XS)
        self.library_summary = _label("", "HelpText")
        footer_layout.addWidget(self.library_summary)
        self.status = _label("", "HelpText")
        self.status.hide()
        footer_layout.addWidget(self.status)
        outer.addWidget(footer)

        self.search.textChanged.connect(self._filter)
        self.era.currentIndexChanged.connect(self._filter)
        self.papers_only.toggled.connect(self._filter)
        self.timeline.currentItemChanged.connect(self._selected)
        self.timeline.itemClicked.connect(self._remember_selection)
        self._filter()
        self.set_library(self.papers, "")

    def _search_text(self, entry):
        readings = [self._papers[key] for key in entry.get("paper_ids", []) if key in self._papers]
        return " ".join(str(value) for value in [entry["date"], entry["title"], entry["era"],
                         entry.get("people", ""), entry["summary"], entry.get("significance", ""),
                         *entry.get("connections", []), *[paper["title"] for paper in readings],
                         *[paper.get("summary", "") for paper in readings]]).casefold()

    def _filter(self, *_):
        terms = self.search.text().casefold().split()
        era = self.era.currentData()
        matching = [entry for entry in self.entries
                    if (not era or entry["era"] == era)
                    and (not self.papers_only.isChecked() or entry.get("paper_ids"))
                    and all(term in self._search_text(entry) for term in terms)]
        self._rendering = True
        self.timeline.clear()
        selected = None
        for entry in matching:
            item = QListWidgetItem(f"{entry['date']}: {entry['title']}. {entry['summary']}")
            item.setData(Qt.ItemDataRole.UserRole, entry["id"])
            item.setToolTip(entry["title"])
            self.timeline.addItem(item)
            if entry["id"] == self._preferred_selection:
                selected = item
        if matching:
            self.timeline.setCurrentItem(selected or self.timeline.item(0))
        self._rendering = False
        self.count.setText(f"{len(matching)} of {len(self.entries)} milestones")
        self.empty.setVisible(not matching)
        self.timeline.setVisible(bool(matching))
        self._show_entry(self.timeline.currentItem())

    def _selected(self, current, previous=None):
        if self._rendering:
            return
        if current is not None:
            self._preferred_selection = current.data(Qt.ItemDataRole.UserRole)
        self._show_entry(current)

    def _remember_selection(self, item):
        # Clicking the current fallback row is still an explicit selection.
        if item is not None:
            self._preferred_selection = item.data(Qt.ItemDataRole.UserRole)

    def _show_entry(self, item):
        while self.detail_layout.count():
            old = self.detail_layout.takeAt(0)
            if old.widget() is not None:
                old.widget().hide()
                old.widget().deleteLater()
        self.source_buttons = []
        self.paper_buttons = []
        if item is None:
            self.detail_layout.addWidget(_label("Select a milestone to read its context and sources.", "HelpText"))
            self.detail_layout.addStretch()
            return
        entry = self._entries[item.data(Qt.ItemDataRole.UserRole)]
        self.detail_layout.addWidget(_label(entry["date"] + " · " + entry["era"], "HelpText"))
        self.detail_layout.addWidget(_label(entry["title"], "PageTitle"))
        if entry.get("people"):
            self.detail_layout.addWidget(_label(str(entry["people"]), "HelpText"))
        self.detail_layout.addWidget(_label(entry["summary"]))
        if entry.get("significance"):
            self.detail_layout.addWidget(_label("Why it matters", "FieldLabel"))
            self.detail_layout.addWidget(_label(entry["significance"]))
        if entry.get("connections"):
            self.detail_layout.addWidget(_label("Connections to this workbench", "FieldLabel"))
            self.detail_layout.addWidget(_label("\n\n".join(entry["connections"])))
        if entry.get("sources"):
            self.detail_layout.addWidget(_label("Sources", "FieldLabel"))
            for source in entry["sources"]:
                button = self._reading_row("Open source", source["title"], source["url"])
                button.setAccessibleName("Open source: " + source["title"])
                button.clicked.connect(lambda checked=False, url=source["url"]: self.source_requested.emit(url))
                self.source_buttons.append(button)
        related = [self._papers[key] for key in entry.get("paper_ids", []) if key in self._papers]
        if related:
            self.detail_layout.addWidget(_label("Related readings", "FieldLabel"))
            for paper in related:
                available = bool(paper.get("available"))
                caption = paper["title"] + f" ({paper['year']}) · " + paper.get("kind", "Paper")
                button = self._reading_row("Open PDF" if available else "PDF unavailable", caption,
                                           paper.get("path", paper["filename"]))
                button.setAccessibleName("Open local PDF: " + paper["title"])
                button.setEnabled(available and not self._busy)
                button.clicked.connect(lambda checked=False, key=paper["id"]: self.paper_requested.emit(key))
                self.paper_buttons.append(button)
        self.detail_layout.addStretch()
        self.detail_scroll.verticalScrollBar().setValue(0)

    def _reading_row(self, action, title, tooltip):
        row = QWidget()
        row.setMinimumWidth(0)
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, theme.SPACE_XS)
        layout.setSpacing(theme.SPACE_XS)
        layout.addWidget(_label(title, "HelpText"))
        button = QPushButton(action)
        button.setToolTip(str(tooltip))
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        button.setEnabled(not self._busy)
        layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.detail_layout.addWidget(row)
        return button

    def set_library(self, papers: list[dict], directory: str):
        """Render coordinator-supplied file availability; do not inspect paths."""
        self.papers = deepcopy(papers)
        self._papers = {paper["id"]: paper for paper in self.papers}
        self._directory = directory
        available = sum(bool(paper.get("available")) for paper in self.papers)
        self.library_summary.setText(f"{available} of {len(self.papers)} local PDFs available")
        self.library_summary.setToolTip(directory)
        self.library_button.setToolTip(directory or "Choose the folder containing your AI/ML PDFs.")
        self._filter()

    def session_state(self):
        return {"query": self.search.text(), "era": self.era.currentData(),
                "papers_only": self.papers_only.isChecked(), "selected": self._preferred_selection,
                "scroll": self.timeline.verticalScrollBar().value()}

    def restore_session(self, state: dict):
        if not isinstance(state, dict):
            return
        for control in (self.search, self.era, self.papers_only):
            control.blockSignals(True)
        self.search.setText(str(state.get("query", "")))
        index = self.era.findData(state.get("era", ""))
        self.era.setCurrentIndex(max(0, index))
        self.papers_only.setChecked(bool(state.get("papers_only", False)))
        preferred = state.get("selected")
        if isinstance(preferred, str) and preferred in self._entries:
            self._preferred_selection = preferred
        for control in (self.search, self.era, self.papers_only):
            control.blockSignals(False)
        self._filter()
        scroll = state.get("scroll", 0)
        if isinstance(scroll, int):
            self._pending_scroll = max(0, scroll)
            self._apply_scroll()
            self._scroll_restore.start(0)

    def _apply_scroll(self):
        self.timeline.verticalScrollBar().setValue(self._pending_scroll)

    def set_busy(self, busy: bool):
        self._busy = bool(busy)
        for widget in (self.search, self.era, self.papers_only, self.library_button, self.timeline):
            widget.setEnabled(not busy)
        self._show_entry(self.timeline.currentItem())

    def set_status(self, message: str):
        self._status_message = message
        self.status.setText(message)
        self.status.setVisible(bool(message))
