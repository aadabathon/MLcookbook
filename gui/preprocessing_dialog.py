"""Dialog for preprocessing and validation choices.

Dumb like every widget: it edits a plain dict whose keys and values match the
core's ``PreprocessingConfig`` fields (app.py builds the config) plus
``cv_folds``. It never imports mlcookbook.
"""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout

# (key, label, [(option label, value), ...]); the first option is the default.
PREPROCESSING_OPTIONS = [
    ("numeric_impute", "Numeric missing values", [("Fill with training median", "median"),
                                                   ("Fill with training mean", "mean")]),
    ("scaling", "Numeric scaling", [("Recipe default", None), ("Standard (z-score)", "standard"),
                                    ("Robust (median / IQR)", "robust"), ("Min-max to [0, 1]", "minmax"),
                                    ("None (real units)", "none")]),
    ("categorical_impute", "Categorical missing values", [("Fill with most frequent level", "most_frequent"),
                                                          ("Keep as level '(missing)'", "missing_category")]),
    ("categorical_encoding", "Categorical encoding", [("One-hot", "onehot"), ("Ordinal codes", "ordinal")]),
    ("min_category_frequency", "Rare levels", [("Keep every level", None), ("Pool levels under 1%", 0.01),
                                               ("Pool levels under 5%", 0.05), ("Pool levels with < 10 rows", 10)]),
    ("missing_indicators", "Missing-value flags", [("Off", False), ("Add a 0/1 flag per numeric feature", True)]),
]
CV_OPTIONS = [("Off", 0), ("3 folds", 3), ("5 folds", 5), ("10 folds", 10)]
DEFAULT_CV_FOLDS = 5


def default_preprocessing() -> dict:
    return {key: options[0][1] for key, _, options in PREPROCESSING_OPTIONS}


def summarize(preprocessing: dict, cv_folds: int | None) -> str:
    """One line describing the current choices, for the form."""
    labels = {key: dict((v, label) for label, v in options) for key, _, options in PREPROCESSING_OPTIONS}
    scaling = preprocessing.get("scaling")
    parts = [
        f"{preprocessing['numeric_impute']} imputation",
        "recipe scaling" if scaling is None else f"{scaling} scaling",
        preprocessing["categorical_encoding"].replace("onehot", "one-hot"),
    ]
    if preprocessing.get("categorical_impute") == "missing_category":
        parts.append("missing as level")
    if preprocessing.get("min_category_frequency") is not None:
        parts.append(labels["min_category_frequency"][preprocessing["min_category_frequency"]].lower())
    if preprocessing.get("missing_indicators"):
        parts.append("missing flags")
    if cv_folds is not None:
        parts.append(f"CV {cv_folds} folds" if cv_folds else "no CV")
    return " · ".join(parts)


class PreprocessingDialog(QDialog):
    def __init__(self, preprocessing: dict, cv_folds: int | None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preprocessing & validation")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        note = QLabel("Preprocessing is fitted on training rows and refitted within each training fold. "
                      "The test holdout stays separate." if cv_folds is not None else
                      "Preprocessing is fitted on all selected rows. This unsupervised analysis describes "
                      "the supplied dataset; it has no test holdout or training cross-validation.")
        note.setWordWrap(True)
        note.setObjectName("HelpText")
        layout.addWidget(note)
        form = QFormLayout()
        self.combos: dict[str, QComboBox] = {}
        for key, label, options in PREPROCESSING_OPTIONS:
            shown_options = options if cv_folds is not None else [(text.replace("training ", ""), value)
                                                                 for text, value in options]
            self.combos[key] = self._combo(shown_options, preprocessing.get(key))
            form.addRow(label, self.combos[key])
        self.cv = None
        if cv_folds is not None:  # unsupervised recipes have no held-out evaluation to cross-validate
            self.cv = self._combo(CV_OPTIONS, cv_folds)
            form.addRow("Cross-validation (training rows)", self.cv)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _combo(options, current) -> QComboBox:
        combo = QComboBox()
        for label, value in options:
            combo.addItem(label, value)
        for i, (_, value) in enumerate(options):
            if value == current and type(value) is type(current):
                combo.setCurrentIndex(i)
        return combo

    def values(self) -> tuple[dict, int | None]:
        preprocessing = {key: combo.currentData() for key, combo in self.combos.items()}
        return preprocessing, (self.cv.currentData() if self.cv is not None else None)
