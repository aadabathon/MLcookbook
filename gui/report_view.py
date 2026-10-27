"""Read-only report renderer shared by recipes and Kitchen results."""
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.plot import BarList, PlotWidget
from gui.tree_plot import TreeWidget

METRIC_CAPTIONS = {
    "ACCURACY": "Accuracy", "BALANCED ACC.": "Balanced accuracy", "PRECISION": "Precision", "RECALL": "Recall",
    "MACRO F1": "Macro F1", "ARCHITECTURE": "Architecture", "QUANTIZATION": "Quantization",
    "ROWS": "Rows", "TREES": "Trees", "CLUSTERS": "Clusters", "COMPONENTS": "Components",
    "CONVERGED": "Converged", "EPOCHS": "Epochs", "PARAMETERS": "Parameters", "SUPPORT VECTORS": "Support vectors",
    "INERTIA": "Inertia", "SILHOUETTE": "Silhouette", "BEST k (SILHOUETTE)": "Best k (silhouette)",
    "DEPTH RANGE": "Depth range", "LEAF COUNT RANGE": "Leaf count range", "SHARE OF TRAINING ROWS": "Share of training rows",
    "OUTPUT TOKENS": "Output tokens", "MODEL LOAD": "Model load", "TOKENS / SECOND": "Tokens / second",
    "TOKENS / WALL SECOND": "Tokens / wall second", "VALID ROWS": "Valid rows", "WALL TIME": "Wall time",
    "CRAMÉR'S V": "Cramér's V", "NORMALITY p": "Normality p", "SPREAD p": "Spread p", "SPREAD RHO": "Spread RHO",
    "KRUSKAL p": "Kruskal p", "FISHER p": "Fisher p", "SKEW": "Skew", "EXCESS KURTOSIS": "Excess kurtosis",
    "TRUST": "Trust", "PC1 SHARE": "PC1 share", "FOR 90%": "For 90%", "EIGENVALUE > 1": "Eigenvalue > 1",
}


class ReportView(QScrollArea):
    def __init__(self, report: dict, *, checks=(), parent=None):
        super().__init__(parent)
        self.view = {"notes": list(checks)}
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidget(self._build_content(report))

    def _build_content(self, report: dict) -> QWidget:
        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(0, theme.SPACE_SM, theme.SPACE_SM, 0)
        column.setSpacing(theme.SPACE_MD)

        if not report:
            checks = list(self.view.get("notes", [])) + list(self.view.get("issues", []))
            if checks:
                column.addWidget(self._section_label("Data checks"))
                for message in checks:
                    note = QLabel(f"•  {message}")
                    note.setWordWrap(True)
                    column.addWidget(note)
        else:
            lines = ["Dataset: " + report.get("dataset", "")]
            if report.get("target"):
                lines.append("Target: " + report["target"])
            lines.append(("Features: " if report.get("target") else "Columns: ") + report.get("features", ""))
            if report.get("settings"):
                lines.append("Settings: " + report["settings"])
            lines.extend(report.get("run_details", []))
            context = QLabel("\n".join(lines))
            context.setTextFormat(Qt.TextFormat.PlainText)
            context.setWordWrap(True)
            context.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            details = QPushButton("▸  Run details")
            details.setObjectName("RunDetailsToggle")
            details.setCheckable(True)
            details.setProperty("disclosure_title", "Run details")
            details.setToolTip("Inspect the displayed run's data, settings and recorded provenance.")
            details.toggled.connect(context.setVisible)
            details.toggled.connect(self._toggle_disclosure)
            column.addWidget(details)
            column.addWidget(context)
            context.hide()
            if report.get("summary"):
                summary = QLabel(report["summary"])
                summary.setObjectName("ResultSummary")
                summary.setTextFormat(Qt.TextFormat.PlainText)
                summary.setWordWrap(True)
                column.addWidget(summary)
        for section in report.get("text_sections", []):
            column.addWidget(self._section_label(section["title"]))
            text = QPlainTextEdit()
            text.setPlainText(section["text"])
            text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            text.setReadOnly(True)
            text.setMinimumHeight(140)
            text.setMaximumHeight(280)
            column.addWidget(text)
        if "tree" in report:  # the diagram leads the report, in its own horizontal scroller
            column.addWidget(self._tree_view(report["tree"]))
        for spec in report.get("plots", []):  # graphs next
            column.addWidget(PlotWidget(spec))
        for group in report.get("metric_groups", []):
            column.addWidget(self._metric_group(group))
        for bars in report.get("bars", []):
            column.addWidget(self._section_label(bars["title"]))
            column.addWidget(BarList(bars))
        for table in report.get("tables", []):
            column.addWidget(self._table(table))
        for warning in report.get("warnings", []):
            note = QLabel(f"⚠  {warning}")
            note.setObjectName("WarningNote")
            note.setWordWrap(True)
            column.addWidget(note)
        column.addStretch()

        return content


    @staticmethod
    def _tree_view(spec: dict) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        if "legend" in spec:
            text = spec["legend"] + " &nbsp;·&nbsp; scroll sideways for the full tree"
        else:
            text = (
                f"<span style='color:{theme.PLOT_ALT}'>●</span> {spec.get('negative_label', '')} &nbsp;&nbsp; "
                f"<span style='color:{theme.ACCENT}'>●</span> {spec.get('positive_label', '')} &nbsp;&nbsp;·&nbsp;&nbsp; "
                "bar = class mix in the node &nbsp;·&nbsp; left branch = condition true (≤) &nbsp;·&nbsp; "
                "scroll sideways for the full tree"
            )
        legend = QLabel(text)
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setObjectName("FieldLabel")
        legend.setWordWrap(True)
        layout.addWidget(legend)
        tree = TreeWidget(spec)
        scroll = QScrollArea()
        scroll.setWidget(tree)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(tree.height() + 18)
        # Open centered on the root once the scroller has its real width.
        QTimer.singleShot(0, scroll, lambda: scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().maximum() // 2))
        layout.addWidget(scroll)
        return box


    def _metric_group(self, group: dict) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        layout.addWidget(self._section_label(group["title"]))

        row = QHBoxLayout()
        row.setSpacing(theme.SPACE_SM)
        for label, value in group["items"]:
            tile = QFrame()
            tile.setObjectName("MetricTile")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setSpacing(2)
            number = QLabel(value)
            number.setObjectName("MetricValue")
            number.setWordWrap(True)
            caption = QLabel(METRIC_CAPTIONS.get(label, label))
            caption.setObjectName("MetricLabel")
            caption.setWordWrap(True)
            tile_layout.addWidget(number)
            tile_layout.addWidget(caption)
            row.addWidget(tile)
        layout.addLayout(row)
        return box


    def _table(self, spec: dict) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        layout.addWidget(self._section_label(spec["title"]))

        headers, rows = spec["headers"], spec["rows"]
        table = QTableWidget(len(rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setShowGrid(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        stretch = 0 if spec.get("stretch_first") else spec.get("stretch_column")
        if stretch is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
        for r, row_values in enumerate(rows):
            for c, text in enumerate(row_values):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                if c > 0 and c != stretch:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(r, c, item)
        row_height = table.verticalHeader().defaultSectionSize()
        table.setFixedHeight(min(330, table.horizontalHeader().height() + row_height * len(rows) + 4))
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(table)
        return box


    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("FieldLabel")
        label.setWordWrap(True)
        return label


    @Slot(bool)
    def _toggle_disclosure(self, opened: bool):
        button = self.sender()
        button.setText(("▾  " if opened else "▸  ") + button.property("disclosure_title"))

