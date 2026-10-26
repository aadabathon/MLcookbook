"""Movement-study controls; all validation/evaluation belongs to the core."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QLabel, QPushButton

from gui.screens.report_screen import ReportScreen

PARAMS = [
    {"key": "cv_folds", "label": "Training group folds", "kind": "choice",
     "options": [("Up to 5 groups", 5), ("Up to 3 groups", 3), ("Up to 10 groups", 10), ("Disabled", 0)]},
    {"key": "final_test", "label": "Reserved test groups", "kind": "choice",
     "options": [("Keep sealed (validation only)", False), ("Evaluate frozen final test", True)],
     "help": "Select the final test only after freezing model and threshold decisions."},
    {"key": "random_state", "label": "Random seed", "kind": "int", "min": 0, "max": 2147483647, "value": 42},
]


class WearableScreen(ReportScreen):
    auto_run_demo = False
    study_requested = Signal(str)
    evaluation_export_requested = Signal(str)

    def __init__(self, engine, report, parent=None):
        super().__init__("Movement Study", engine, report, parent, task="unsupervised", params=PARAMS,
                         min_features=0, cross_validation=False, preprocessing=False,
                         settings_title="Grouped evaluation settings")
        self.load_button.setText("Open movement study…")
        self.sections["import"].hide()
        self.columns.hide()
        self.columns_help.hide()
        self.columns_label.setText("Fixed study contract")
        note = QLabel("Open the export.json produced by the wearable pipeline. Human labels, predictor columns "
                      "and whole-person/session splits stay fixed. IDs and times cannot become predictors. "
                      "Logistic regression and random forest are compared on validation groups.")
        note.setWordWrap(True)
        layout = self.columns_label.parentWidget().layout()
        layout.insertWidget(layout.indexOf(self.columns_label) + 1, note)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.run_button.setText("Run grouped baselines")
        self.export_button = QPushButton("Export evaluation…")
        self.export_button.clicked.connect(self._export)
        self.result_layout.insertWidget(2, self.export_button)
        self.export_button.setEnabled(False)

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open movement study export.json", "", "Study manifest (export.json)")
        if path:
            self.study_requested.emit(path)

    def _problem(self):
        if "__wearable_split" not in {c["name"] for c in self._columns}:
            return "Open a movement study before running grouped baselines."
        return None

    def _update_run(self):
        self.run_button.setEnabled(self._problem() is None and not self._busy)
        self.run_button.setToolTip("Uses fixed group holdouts and training-only preprocessing.")

    def set_busy(self, busy):
        super().set_busy(busy)
        self.export_button.setEnabled(not busy and bool(self._report.get("wearable_evaluation")))

    def show_report(self, report, request=None):
        super().show_report(report, request=request)
        if hasattr(self, "export_button"):
            self.export_button.setEnabled(not self._busy and bool(report.get("wearable_evaluation")))

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export evaluation with predictions and lineage", "movement-evaluation.json", "JSON (*.json)")
        if path:
            self.evaluation_export_requested.emit(path)
