# ML Cookbook

A statistics, data science, and classical machine-learning workbench for arbitrary tabular datasets — and a learning environment for building one deliberately.

The core idea: you hand the library a pandas DataFrame, it gives you a typed `Dataset`, and every analysis returns a **structured, JSON-serializable result object** instead of printing or plotting. That keeps the analytical core independent of notebooks, CLIs, and web frontends, so it can later sit unchanged behind a FastAPI service.

```text
DataFrame
   ↓
Dataset / Schema          (core)
   ↓
Preprocessing             (sklearn Pipeline / ColumnTransformer, leakage-safe)
   ↓
Statistics / ML workflows (statistics, regression, ...)
   ↓
Structured Result Objects (frozen dataclasses with .to_dict())
   ↓
Python / Notebook / future REST API / future Web UI
```

## Quick start

```python
import pandas as pd
from mlcookbook import Dataset, correlation, decision_tree, linear_regression, logistic_regression, random_forest

data = Dataset(df)                      # infers a VarType for every column
data.schema()                           # numeric / categorical / boolean / datetime / text / unsupported
data.describe()                         # missingness, cardinality, constant columns
data.set_type("zip_code", "categorical")  # override a wrong inference

corr = correlation(data, "sqm", "price", method="spearman")
corr.statistic, corr.p_value, corr.warnings

result = linear_regression(data, target="price")
result.test_metrics.r2
result.to_dict()                        # plain dict, ready for json.dumps / FastAPI

clf = logistic_regression(data, target="churned", positive_class=True)
clf.test_metrics.roc_auc, clf.odds_ratios

tree = decision_tree(data, target="churned", max_depth=3)
tree.nodes, tree.rules, tree.depth_curve   # the fitted tree as plain data

forest = random_forest(data, target="churned", positive_class=True, n_estimators=100)
forest.test_metrics.roc_auc, forest.feature_importances
forest.roc_fpr, forest.roc_tpr             # held-out ROC coordinates
```

Run `python examples/basic_usage.py` for a complete tour.

## Installation

Requires Python 3.12+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (`source .venv/bin/activate` on Unix)
pip install -e ".[dev]"
pytest
```

## Architecture

`src/`-layout package; each module owns one concern:

| Module | Status | Responsibility |
|---|---|---|
| `core` | ✅ | `Dataset`, schema/type inference (`VarType`), overrides, metadata, result serialization (`Result`, `to_jsonable`) |
| `io` | Ready | `load_dataset(path, delimiter=",", encoding="utf-8-sig")` reads flat CSV/Parquet into a `Dataset`; Parquet uses PyArrow |
| `preprocessing` | ✅ minimal | `build_preprocessor(dataset, columns)` → unfitted sklearn `ColumnTransformer` |
| `statistics` | ✅ one workflow | `correlation` (Pearson/Spearman) — the template for future association tests |
| `regression` | ✅ one workflow | `linear_regression` — the template for future supervised workflows |
| `classification` | ✅ three workflows | `logistic_regression`, `decision_tree`, `random_forest` (all binary, stratified split; trees expose rules/depth diagnostics; forests expose held-out ROC data, importances, and per-tree sizes) |
| `evaluation` | ✅ minimal | `regression_metrics` → `RegressionMetrics`, `classification_metrics` → `ClassificationMetrics` |
| `clustering`, `dimensionality`, `visualization` | 🚧 skeletal | Documented as future work; see `ROADMAP.md` |

### Design rules

1. **Types drive applicability.** Every column gets a `VarType` (a statistical role, not a pandas dtype). Analyses validate their inputs against the schema and refuse or warn rather than silently coercing. Users can override any inference.
2. **Results are data.** Every analysis returns a frozen dataclass mixing in `Result`, which provides `to_dict()` via a recursive serializer (`core/results.py`) that handles nested dataclasses, numpy scalars/arrays, enums, and NaN → `null`. No analysis prints or plots.
3. **Warnings are part of the result.** Dropped rows, skipped columns, constant variables, and assumption issues are surfaced in `result.warnings`, not swallowed.
4. **No leakage.** Preprocessors are returned unfitted and are fitted inside a `Pipeline` on training data only (`regression/linear.py` shows the pattern).
5. **sklearn/scipy do the math.** This library orchestrates and validates; it does not reimplement numerical algorithms.
6. **Small public API.** Only `Dataset`, `VarType`, `correlation`, `linear_regression`, `logistic_regression`, `decision_tree`, `random_forest` (and their result types) are exported at the top level.

### Adding a new workflow

Copy the shape of an existing one:

- A statistical test → `statistics/association.py`: validate types from the schema, compute with scipy, return a frozen result dataclass with warnings.
- A supervised model → `regression/linear.py` (or `classification/logistic.py`): validate target/features, build a `Pipeline` with `build_preprocessor`, train/test split, evaluate both splits via `evaluation`, return a result dataclass.

## Desktop app

A PySide6 desktop shell, launched with a single entry point:

```bash
pip install -e ".[gui]"
python -m mlcookbook
```

Three layers, one seam:

- `gui/` — dumb presentation: widgets render data and emit signals (`FrontPage.recipe_selected`, screens' `back_requested`). Never imports the core. Visual constants live in `gui/theme.py`; the card grid is data-driven from `gui/recipes.py`.
- `src/mlcookbook/` — the analytical core, UI-free as ever (`mlcookbook/__main__.py` is only a trampoline to the app).
- `app.py` (repo root) — **the only module that knows both sides.** Owns the window and navigation (delete-and-redraw screen swapping) and maps a ready recipe's `key` to a screen factory in `SCREEN_FACTORIES`, where core entry points get bound to screens.

Until a user dataset is imported, ready recipes open by running their core function on a small built-in scikit-learn dataset (diabetes for linear, breast cancer for logistic, trees, and forests) and render the result through the shared `gui/screens/report_screen.py` layout (a `QPainter` visual from `gui/plot.py` / `gui/tree_plot.py` leads the report: actual-vs-predicted for linear, the fitted sigmoid for logistic, the drawn tree plus a depth-vs-overfit curve for decision trees, and a held-out ROC curve for random forests); `app.py` formats the result into plain strings first, so screens stay core-free. All four recipes also accept local CSV and Parquet files through the shared import controls.

### Random forests

`random_forest` in `classification/forest.py` returns a frozen `RandomForestResult`. It reuses binary-target validation, stratified holdout splitting, classification metrics, and an unfitted preprocessor inside a training-only sklearn pipeline. Numeric inputs are not scaled. CSV/Parquet input, type overrides, and target/features/positive-class controls work exactly as in the other ready recipes.

Defaults: 100 bootstrap trees, `max_depth=None`, `min_samples_leaf=5`, `max_features="sqrt"`, `test_size=0.25`, and `random_state=0`. These parameters are configurable through the Python API; the desktop screen uses those defaults. The forest runs with one worker inside the app's background analysis job.

The result includes train/test metrics, held-out labels/predictions/positive-class probabilities, ROC coordinates, per-encoded-feature impurity importances, and each tree's depth and leaf count. The screen renders the ROC curve, metric tiles, forest-size summary, importance bars, confusion matrix, and warnings. Importance values sum to one when splits contribute importance, or zero for a forest of unsplit trees; they may favor high-cardinality features and do not establish causation. See [sklearn's RandomForestClassifier reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Missing targets, unsupported auto-selected columns, imbalance, missing/constant features, sklearn preprocessing warnings, and a train/test accuracy gap over 0.10 are surfaced through result warnings. If the held-out split has one class, ROC coordinates are empty and ROC AUC serializes as `null`. This initial forest recipe is binary classification; forest regression and multiclass evaluation remain future work.

### Analyze your own data

The current file picker accepts `.csv` and `.parquet` only. CSV supports comma, semicolon, tab, and pipe delimiters, with UTF-8 (including BOM), Windows-1252, or UTF-16 encoding. Selecting Tab does not yet enable files named `.tsv`; compressed CSV and other formats are future work.

1. Open one of the four ready recipes and click **Open CSV / Parquet**. For CSV, choose its delimiter and encoding before opening (defaults: comma, UTF-8).
2. Preview the first 20 rows and review each column's inferred statistical type and missing-value count. Types are editable roles, not conversions of the underlying values.
3. Choose your target, check the input features, and exclude IDs or columns that reveal the answer. Linear regression needs a numeric target; logistic regression, decision trees, and random forests need exactly two classes and let you select the positive class.
4. Click **Run Analysis**. Import and training run in background threads; reports show the dataset, target, and features used. Existing train/test splits and training-only preprocessing remain unchanged.

The loaded dataset stays available when switching recipes during the current session; each screen asks you to choose its target and features again. Column-role overrides and positive-class choices also reset on navigation. Nothing is saved across app restarts; project saving and result export are not implemented yet. Import errors preserve the previous dataset. Failed runs show an error and preserve the previous report, whose settings remain labeled.

CSV files need a header with unique, non-empty column names. Standard pandas missing-value markers (including blank fields and `NA`) apply; CSV type inference can remove leading zeros from numeric-looking fields, so use Parquet string columns when that distinction matters. CSV date strings are not automatically parsed as dates. Parquet preserves stored types; nested list/struct columns must be flattened before import. Files are loaded into memory, so this first version is intended for datasets that fit in RAM. Model hyperparameters and the 25% test split retain the existing defaults.

Library use:

```python
from mlcookbook.io import load_dataset

data = load_dataset("measurements.parquet")
# Or: load_dataset("measurements.csv", delimiter=";", encoding="cp1252")
result = linear_regression(data, target="price", features=["area", "rooms"])
```

Import validation rejects empty datasets, duplicate or blank column names, CSV records with extra fields, and nested Parquet columns. Missing target rows are handled by the existing analyses and reported in their warnings.

PyArrow is a required dependency. Nullable categorical/string/boolean missing values are normalized inside the preprocessing pipeline before training-only imputation.

To bring a recipe alive: set its `variant="ready"` and `key` in `gui/recipes.py`, build its screen in `gui/screens/`, and register a factory in `app.py`.

Screens share one layout (`gui/screens/report_screen.py`) that renders a plain report dict; visuals are drawn with `QPainter` (`gui/plot.py`, `gui/tree_plot.py`), so there is no plotting dependency. See "Desktop GUI conventions" in `AGENTS.md` for the full contract.

A standalone Windows build (PyInstaller) is produced into `dist/MLcookbook/MLcookbook.exe` — see `packaging/` notes in ADAM.md.

## Web portability

The core has no UI or web dependency. A future service is just:

```python
@app.post("/analyze/correlation")
def analyze(...):
    result = mlcookbook.correlation(dataset, x, y)
    return result.to_dict()
```

## Where to start reading

1. `src/mlcookbook/core/schema.py` — `VarType` and inference rules
2. `src/mlcookbook/core/dataset.py` — the `Dataset` entry point
3. `src/mlcookbook/core/results.py` — how results serialize
4. `src/mlcookbook/statistics/association.py` — the statistical-workflow template
5. `src/mlcookbook/regression/linear.py` — the ML-workflow template
6. `src/mlcookbook/classification/logistic.py`, `tree.py`, and `forest.py` — binary classifiers (shared validation in `_binary.py`)

Then `ROADMAP.md` for the learning curriculum and the proposed data/workflow directions under discussion.
