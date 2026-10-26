"""Concise recipe reference for the desktop UI; no analysis or core imports."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui import theme

SECTION_TITLES = ("Applicability", "Data and assumptions", "Settings", "Interpreting results", "Limitations")

# The coordinator supplies the callable's identity. Display titles and summaries
# may change independently without selecting a different statistical reference.
CALLABLE_RECIPES = {
    "mlcookbook.regression.linear.linear_regression": "linear-regression",
    "mlcookbook.classification.logistic.logistic_regression": "logistic-regression",
    "mlcookbook.classification.tree.decision_tree": "decision-tree",
    "mlcookbook.regression.tree.decision_tree_regression": "decision-tree",
    "mlcookbook.classification.forest.random_forest": "random-forest",
    "mlcookbook.regression.forest.random_forest_regression": "random-forest",
    "mlcookbook.classification.svm.svm": "svm",
    "mlcookbook.regression.svm.svm_regression": "svm",
    "mlcookbook.classification.neural_network.neural_network": "neural-network",
    "mlcookbook.regression.neural_network.neural_network_regression": "neural-network",
    "mlcookbook.clustering.kmeans.kmeans": "kmeans",
    "mlcookbook.dimensionality.pca.pca": "pca",
    "mlcookbook.statistics.scan.association_scan": "association-scan",
    "mlcookbook.comparison.compare_regressors": "model-comparison",
    "mlcookbook.llm.generate_text": "llm",
    "mlcookbook.wearable.evaluate_wearable": "wearable-study",
}

REFERENCE_CONTENT = {
    "linear-regression": (
        "Predict a numeric outcome with an additive linear model. Ordinary least squares provides a baseline; "
        "ridge and lasso shrink coefficients when predictors are numerous or correlated.",
        "Choose a numeric target and exclude identifiers or information unavailable at prediction time. "
        "Rows with missing targets are dropped. Imputation, encoding and scaling are fitted on training rows "
        "and refitted within each training cross-validation fold.",
        "Penalty selects ordinary least squares, ridge (L2) or lasso (L1). Larger alpha means stronger shrinkage "
        "and has no effect without a penalty. Scaling changes coefficient units and the relative penalty on features.",
        "Lower held-out MAE and RMSE indicate smaller prediction errors in target units. R² compares squared error "
        "with a constant-mean baseline and can be negative. Actual-versus-predicted points should follow the diagonal. "
        "With default standard scaling, numeric coefficients describe a one-standard-deviation feature change.",
        "Coefficients describe conditional associations, not causal effects. Correlated predictors can make them "
        "unstable. Residual diagnostics use training residuals; normality and constant spread matter for classical "
        "inference, but this recipe does not report coefficient confidence intervals. Row-wise splits do not account "
        "for repeated subjects or time order.",
    ),
    "logistic-regression": (
        "Classify a categorical outcome with binary logistic or multinomial regression. Use it when a linear "
        "relationship in log-odds is a useful baseline and coefficient interpretation matters.",
        "This interface accepts targets with 2–20 classes and enough rows for stratified splitting. For two classes, select the "
        "positive class to define precision, recall and ROC metrics. Missing targets are dropped; feature "
        "preprocessing is fitted only on training rows and within training cross-validation folds.",
        "Select predictors, statistical roles, preprocessing and training cross-validation. Standard scaling is "
        "the default. The fitted sklearn logistic model uses its default L2 regularization; this screen does not "
        "expose its penalty strength.",
        "Read the confusion matrix alongside precision, recall and F1. Multiclass metrics use macro averages, "
        "giving each class equal weight. ROC AUC describes ranking, not probability calibration. Binary coefficients "
        "are log-odds; exponentiating gives an odds ratio. Multiclass coefficient differences describe pairwise log-odds.",
        "Accuracy can hide poor minority-class performance. Probabilities are not guaranteed calibrated, and "
        "coefficients are not causal effects. Numeric coefficients depend on scaling and correlated inputs. "
        "Random row splits do not protect against subject or temporal leakage.",
    ),
    "decision-tree": (
        "Fit an interpretable sequence of feature thresholds for classification or numeric regression. "
        "A shallow tree makes nonlinear rules and interactions visible.",
        "Choose the task to match the target: 2–20 classes for classification in this interface, a numeric target for regression. "
        "Missing targets are dropped; preprocessing is learned from training rows only. The default leaves numeric "
        "features unscaled, so their thresholds retain original units.",
        "Maximum depth limits rule complexity. Minimum rows per leaf prevents small terminal groups. "
        "The depth scan uses training-only cross-validation; it does not automatically change the chosen depth. "
        "Custom scaling and encoding change the feature names and units used by the tree.",
        "Follow a root-to-leaf path to read a prediction and its training support. Compare training, held-out "
        "and cross-validation metrics: a large train/test gap suggests overfitting. Classification reports include "
        "a confusion matrix; regression errors remain in target units.",
        "Small data changes can produce different splits. Deep trees can memorize rare patterns, and leaf support "
        "is not a confidence interval. Regression trees predict piecewise constants and do not extrapolate trends. "
        "Random row evaluation does not account for repeated subjects or time order.",
    ),
    "random-forest": (
        "Average many randomized trees for nonlinear classification or regression. Forests capture interactions "
        "without choosing a single readable decision path.",
        "Choose a categorical target with 2–20 classes in this interface or a numeric regression target. Trees use "
        "bootstrap training samples. The classifier considers random feature subsets at splits; the regressor's "
        "default considers all encoded features. Imputation and encoding are learned from training rows; "
        "numeric scaling defaults to none. Cross-validation refits the complete pipeline.",
        "More trees generally stabilize the average and increase runtime. Maximum depth and minimum rows per leaf "
        "control individual-tree complexity; zero maximum depth in the screen means unlimited. The positive class "
        "defines binary classification metrics.",
        "Compare held-out metrics with training and training-CV results. Read the confusion matrix for per-class "
        "errors or actual-versus-predicted plot for regression. Feature importance measures impurity reduction "
        "per encoded feature, so categorical levels may appear separately.",
        "Impurity importance can favor high-cardinality predictors and split credit across correlated features; "
        "it is not a causal effect. More trees do not remove leakage or guarantee calibrated probabilities. "
        "Regression forests do not extrapolate trends beyond learned leaf predictions. Random row splits are "
        "unsuitable for evaluating dependent subjects or future-time performance.",
    ),
    "svm": (
        "Fit a margin-based classifier or support vector regressor. A linear kernel gives a linear boundary; "
        "RBF and polynomial kernels allow nonlinear relationships.",
        "Use a categorical target with 2–20 classes in this interface or a numeric regression target. Distances and margins depend "
        "on feature units, so standard scaling is the default. All preprocessing is trained inside the split "
        "and each CV fold. Regression also standardizes the target inside its training pipeline.",
        "Kernel controls boundary shape. Larger C penalizes training errors more strongly. In regression, epsilon "
        "sets the no-penalty error band in target standard deviations; reported predictions and errors return to "
        "original target units. Cross-validation evaluates the selected settings rather than tuning them automatically.",
        "Read support-vector count as a description of the fitted boundary, not a quality score. Compare held-out "
        "and cross-validation metrics. Classification uses confusion and probability-based metrics; regression "
        "uses MAE, RMSE and R² with an actual-versus-predicted plot.",
        "Kernel training can become expensive as row count grows. Results are sensitive to scaling, C and kernel "
        "choice. Class probabilities are estimates rather than a calibration guarantee. Random row evaluation "
        "does not handle subject or temporal dependence.",
    ),
    "neural-network": (
        "Fit a multilayer perceptron to tabular classification or regression data. This recipe uses sklearn's "
        "MLP, with training diagnostics for the selected network architecture.",
        "Use a target with 2–20 classes in this interface or a numeric regression target. Standard scaling helps optimization; "
        "preprocessing is fitted on training rows and within training CV folds. Early stopping reserves an "
        "internal fraction of the training data, separate from the outer held-out test rows.",
        "Hidden layers specify units per layer. Activation controls the nonlinearity. Alpha sets the L2 penalty, "
        "and maximum epochs bounds training passes. Early stopping can stop before that limit when its internal "
        "validation score stops improving.",
        "Inspect loss and any early-stopping validation curve alongside final held-out and training-CV metrics. "
        "Low training loss alone does not establish generalization. Classification uses confusion and "
        "class-sensitive scores; regression uses errors in target units.",
        "Optimization can stop without convergence, and larger networks can overfit small datasets. The internal "
        "early-stopping curve and outer test metrics describe different rows. A fixed seed aids repeatability "
        "without guaranteeing identical results across runtime versions. This recipe does not define arbitrary "
        "PyTorch or TensorFlow architectures.",
    ),
    "kmeans": (
        "Explore groups of similar rows without a target label. K-means is most useful when Euclidean distance "
        "and compact clusters have a meaningful interpretation for the selected features.",
        "Select numeric, categorical, boolean or datetime predictors; preprocessing encodes them into a numeric "
        "space. Standard scaling is the default. Preprocessing and clustering use all selected rows; there is "
        "no held-out test or training CV in this exploratory workflow.",
        "Cluster count is the number actually fitted. The maximum-k scan compares other counts but does not "
        "automatically replace your selection. Scaling, one-hot encoding and irrelevant features change the "
        "distance geometry and therefore the groups.",
        "Higher silhouette indicates better separation in the encoded space; values near zero suggest overlap. "
        "Inertia is within-cluster squared distance and generally falls as k increases. Compare cluster profiles "
        "in original units. The two-component PCA view is a projection for display, not the space used to fit clusters.",
        "Cluster numbers are arbitrary labels and are not discovered ground truth. K-means can force groups into "
        "continuous or irregular data and is sensitive to outliers. Distinct groups in a 2D projection need not "
        "be well separated in the full feature space. A high silhouette does not validate a downstream prediction task.",
    ),
    "pca": (
        "Summarize a feature matrix with orthogonal linear components that retain variance. Use PCA to inspect "
        "redundancy, dominant directions or a low-dimensional projection without a target label.",
        "Selected numeric, categorical, boolean and datetime features are encoded and imputed before PCA. "
        "Standard scaling is the default; otherwise large-variance units can dominate. The analysis fits all "
        "selected rows and reports no held-out predictive performance.",
        "This recipe computes all available components, up to the smaller of row count and encoded feature count; "
        "there is no component-count control in this interface. Changing preprocessing changes the variance PCA "
        "sees. Inspect the recorded encoded feature names "
        "when interpreting categorical or datetime loadings.",
        "The scree plot shows variance by component; cumulative variance shows how much the retained subspace "
        "captures. The 90% count is the fewest components reaching that cumulative variance. "
        "Loadings describe component directions in the preprocessed space, and scores place rows "
        "along those directions. Component signs can flip without changing the represented subspace.",
        "Variance is not necessarily useful signal or predictive importance. PCA describes linear structure "
        "and can be dominated by outliers. One-hot categorical loadings are per level. Reusing this all-row "
        "exploration as preprocessing for a supervised test would leak information unless PCA is refitted "
        "inside that model's training pipeline.",
    ),
    "association-scan": (
        "Screen pairwise relationships across a mixed table. Numeric pairs use Spearman correlation, "
        "categorical/boolean pairs use chi-square, and mixed numeric/group pairs use Kruskal–Wallis.",
        "Select 2–60 numeric, categorical or boolean columns. Missing values are dropped separately for each "
        "pair, so sample sizes can differ. Constant pairs and pairs with too few complete observations are skipped. "
        "Independent observations are assumed; sparse contingency tables can undermine chi-square approximations.",
        "Column roles determine the test; changing a role does not convert values. There is no model, train/test "
        "split or preprocessing pipeline. The scan adjusts the completed tests' p-values with the "
        "Benjamini–Hochberg false discovery rate procedure.",
        "Read the test name, effect measure, complete-pair count and adjusted p-value together. Numeric effects "
        "use absolute Spearman rho, categorical effects use Cramér's V, and mixed effects use epsilon squared. "
        "The signed Spearman statistic retains direction even though the ranking uses its magnitude.",
        "Effect sizes from different tests are not directly comparable despite sharing a 0–1 scale. "
        "Small adjusted p-values do not establish practical importance or causation. Pairwise deletion can "
        "change the population represented by each result; the scan does not adjust for confounders.",
    ),
    "model-comparison": (
        "Compare ridge and random forest regression under one evaluation design. The selected model is the "
        "candidate with the lowest mean training-CV RMSE; only that model is evaluated on the test holdout.",
        "Choose a numeric target and predictors available at prediction time. Both candidates use the same "
        "training/test rows and materialized training folds. Each fold refits preprocessing on its own training "
        "rows. Both candidates share one preprocessing configuration, with standard scaling by default.",
        "Test fraction reserves the final holdout. The split/model seed supports repeatability. Set at least "
        "three training CV folds in this screen, then choose ridge alpha and forest tree/depth/leaf settings. "
        "These define two candidates rather than an automatic hyperparameter search.",
        "Compare candidate CV RMSE means and fold variability in target units. The held-out plot and test metrics "
        "belong to the CV-selected model. Lower CV error guided selection; the test estimates that selected "
        "pipeline's performance on rows excluded from selection.",
        "This is a comparison of two specified regressors, not evidence that the winner is best among all models. "
        "Repeated decisions based on the same test report erode its independence. The random row split does "
        "not provide grouped or forward-time evaluation.",
    ),
    "llm": (
        "Generate text from a local model, or extract, summarize, rewrite and classify dataset rows into validated "
        "fields. PyTorch/Transformers reads a local model folder; Ollama and llama.cpp connect to existing "
        "loopback servers. The app does not install models or start servers.",
        "A typed prompt needs no dataset. Free row generation needs one string-valued text/categorical column; "
        "structured workflows need explicit column bindings and nullable output fields. Reference-label columns "
        "are for evaluation and cannot enter prompts. Structured tasks use Ollama or llama.cpp.",
        "Choose the engine and observed local model before configuring prompts, output limits, row limits and "
        "sampling. Positive temperature enables sampling; top-p limits its probability mass. Context and "
        "device controls differ by engine: llama.cpp context, GPU offload and thinking are server startup choices. "
        "An optional GGUF path hashes a file and does not load or select the server's weights.",
        "A live preview is incomplete until the run finishes. Inspect each row's raw output and validation outcome "
        "before accepting derived columns. Validation rate measures schema compliance; reference agreement uses "
        "valid non-null pairs and must be read with coverage. Timing and token counts use recorded engine metadata "
        "where available; local wall time includes client-side work.",
        "Valid JSON is not evidence of factual accuracy. Requested settings can differ from what a server applies, "
        "and server-rendered prompts or truncation may be unverified. A selected GGUF hash does not prove which "
        "weights the server loaded. Seeds do not guarantee cross-engine reproducibility. Saved results reopen "
        "offline; reruns still need the external model or server.",
    ),
    "wearable-study": (
        "Compare logistic regression and random forest baselines on a verified wearable movement study. "
        "This workflow preserves the study's fixed train, validation and reserved test groups.",
        "Open the wearable pipeline's export.json or a saved study experiment. Only the known physical-unit "
        "summary features are predictors; identifiers, timestamps and window lineage remain metadata. "
        "Whole-group training CV refits preprocessing per fold rather than splitting related windows at random.",
        "Training group folds controls within-training evaluation. Validation-group macro F1 selects the model. "
        "Reserved test groups remain sealed by default; evaluate them explicitly after freezing modeling "
        "decisions. The seed controls randomized fitting, not the fixed study splits.",
        "Read validation metrics separately from grouped training CV and any explicit final-test report. "
        "Macro F1 gives equal weight to each class. Per-window predictions, probabilities and retained group, "
        "session and window lineage support error inspection and evaluation export.",
        "Many overlapping windows do not create the same number of independent subjects. Performance depends "
        "on the study's held-out groups and labeling protocol. Synthetic fixtures verify integration rather "
        "than real-world accuracy. Repeated final-test inspection compromises its role as a reserved evaluation.",
    ),
}


def recipe_reference(engine: dict) -> tuple[tuple[str, str], ...]:
    """Return display-only reference text for the supplied callable identity."""
    key = CALLABLE_RECIPES.get(engine.get("callable", ""))
    if key is None:
        summary = engine.get("description") or engine.get("summary") or "No recipe reference is available."
        return (("Recipe reference", summary),)
    return tuple(zip(SECTION_TITLES, REFERENCE_CONTENT[key]))


class RecipeReferenceDialog(QDialog):
    def __init__(self, title: str, engine: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{title} reference")
        self.resize(700, 640)
        layout = QVBoxLayout(self)
        layout.setSpacing(theme.SPACE_MD)
        heading = self._label(title)
        heading.setObjectName("ScreenTitle")
        layout.addWidget(heading)

        content = QWidget()
        notes = QVBoxLayout(content)
        notes.setContentsMargins(0, 0, theme.SPACE_SM, 0)
        notes.setSpacing(theme.SPACE_SM)
        self.reference_labels: dict[str, QLabel] = {}
        for section, text in recipe_reference(engine):
            caption = self._label(section)
            caption.setObjectName("FieldLabel")
            notes.addWidget(caption)
            label = self._label(text)
            self.reference_labels[section] = label
            notes.addWidget(label)
            notes.addSpacing(theme.SPACE_XS)

        self.api_toggle = QPushButton("▸  Python API")
        self.api_toggle.setObjectName("SectionToggle")
        self.api_toggle.setCheckable(True)
        self.api_toggle.setAccessibleName("Python API")
        self.api_text = QPlainTextEdit()
        self.api_text.setReadOnly(True)
        self.api_text.setPlainText(engine.get("callable", "") + engine.get("signature", ""))
        self.api_text.setAccessibleName("Python callable and signature")
        self.api_text.setMinimumHeight(110)
        self.api_text.setMaximumHeight(180)
        self.api_text.hide()
        self.api_toggle.toggled.connect(self._toggle_api)
        notes.addWidget(self.api_toggle)
        notes.addWidget(self.api_text)
        notes.addStretch()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(content)
        layout.addWidget(self.scroll, stretch=1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @Slot(bool)
    def _toggle_api(self, opened: bool) -> None:
        self.api_text.setVisible(opened)
        self.api_toggle.setText(("▾  " if opened else "▸  ") + "Python API")

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        return label
