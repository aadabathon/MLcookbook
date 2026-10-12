# ML Cookbook

A statistics, data science, and classical machine-learning workbench for arbitrary tabular datasets — and a learning environment for building one deliberately.

The core idea: you hand the library a pandas DataFrame, it gives you a typed `Dataset`, and every analysis returns a **structured, JSON-serializable result object** instead of printing or plotting. That keeps the analytical core independent of notebooks, CLIs, and web frontends, so it can later sit unchanged behind a FastAPI service.

```text
CSV / Parquet file  ──  load_dataset: leading zeros kept, ISO dates parsed, missing markers, ImportReport
   ↓
Dataset / Schema          (core: VarType per column, overrides, data-quality issues)
   ↓
Preprocessing             (PreprocessingConfig → unfitted ColumnTransformer, fitted on training rows only)
   ↓
Statistics / ML workflows (tests, regression, classification, clustering, PCA)
   ↓
Structured Result Objects (frozen dataclasses with .to_dict(): metrics, CV folds, fitted preprocessing, warnings)
   ↓
Python / Notebook / desktop app / future REST API
```

## Quick start

```python
import pandas as pd
from mlcookbook import (Dataset, PreprocessingConfig, load_dataset, correlation, chi_square, anova,
                        association_scan, linear_regression, logistic_regression, decision_tree,
                        decision_tree_regression, random_forest, random_forest_regression, kmeans, pca)

data = Dataset(df)                      # infers a VarType for every column
data.schema()                           # numeric / categorical / boolean / datetime / text / unsupported
data.describe()                         # missingness, cardinality, constant columns, quality issues
data.set_type("zip_code", "categorical")  # override a wrong inference

# Statistics
correlation(data, "sqm", "price").ci_low       # Pearson/Spearman with a Fisher-z confidence interval
chi_square(data, "smoker", "cough").cramers_v  # plus Fisher's exact test for 2x2 tables
anova(data, "price", "city").recommended_test  # ANOVA and Kruskal-Wallis, with assumption checks
association_scan(data).pairs[0]                # every column pair, test chosen by type, FDR-adjusted

# Supervised learning: every workflow takes preprocessing=... and cv_folds=...
config = PreprocessingConfig(scaling="robust", min_category_frequency=0.05, missing_indicators=True)
result = linear_regression(data, target="price", penalty="ridge", alpha=1.0, preprocessing=config)
result.test_metrics.r2, result.cross_validation.mean["r2"], result.cross_validation.std["r2"]
result.residual_diagnostics.normality_p_value
result.preprocessing.columns[0].steps   # what was learned from the training rows, column by column
result.to_dict()                        # plain dict, ready for json.dumps / FastAPI

logistic_regression(data, target="churned", positive_class=True)   # binary
logistic_regression(data, target="species")                       # 3+ classes: multinomial
decision_tree(data, target="species", max_depth=3).rules
decision_tree_regression(data, target="price").nodes
random_forest(data, target="churned").roc_fpr
random_forest_regression(data, target="price").feature_importances

# Unsupervised learning
kmeans(data, n_clusters=3).profiles     # cluster profiles in original units, k scan, silhouette
pca(data).components[0].loadings        # explained variance, loadings, 2-D scores
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

| Module | Responsibility |
|---|---|
| `core` | `Dataset`, schema/type inference (`VarType`), overrides, data-quality checks (`quality.py`), result serialization (`Result`, `to_jsonable`) |
| `io` | `load_dataset` / `load_dataset_with_report`: flat CSV/Parquet into a `Dataset`, with an `ImportReport` of every conversion |
| `preprocessing` | `PreprocessingConfig`, `build_preprocessor` (unfitted `ColumnTransformer`), `describe_preprocessor` (fitted values per column), `resolve_features` (shared feature selection + quality warnings) |
| `evaluation` | `regression_metrics`, `classification_metrics` (binary or macro-averaged multiclass, per-class metrics), `cross_validate_pipeline` |
| `statistics` | `correlation`, `chi_square`, `anova`, `association_scan` |
| `regression` | `linear_regression` (OLS/ridge/lasso + residual diagnostics), `decision_tree_regression`, `random_forest_regression` |
| `classification` | `logistic_regression`, `decision_tree`, `random_forest` — binary or multiclass; shared validation in `_problem.py` |
| `clustering` | `kmeans` |
| `dimensionality` | `pca` |
| `visualization` | 🚧 skeletal — the desktop app draws its own charts from result data |

### Design rules

1. **Types drive applicability.** Every column gets a `VarType` (a statistical role, not a pandas dtype). Analyses validate their inputs against the schema and refuse or warn rather than silently coercing. Users can override any inference.
2. **Results are data.** Every analysis returns a frozen dataclass mixing in `Result`, which provides `to_dict()` via a recursive serializer (`core/results.py`) that handles nested dataclasses, numpy scalars/arrays, enums, and NaN → `null`. No analysis prints or plots.
3. **Warnings are part of the result.** Dropped rows, skipped columns, constant/ID-like/mostly-missing features, class imbalance, overfitting, and failed assumptions are surfaced in `result.warnings`, not swallowed.
4. **No leakage.** Preprocessors are returned unfitted and are fitted inside a `Pipeline` on training data only. Cross-validation runs on the training rows and refits the whole pipeline, preprocessing included, inside every fold; the test split is touched once.
5. **sklearn/scipy do the math.** This library orchestrates and validates; it does not reimplement numerical algorithms.
6. **Small public API.** The workflows, their result types, `Dataset`, `VarType`, `PreprocessingConfig`, and `load_dataset` are exported at the top level; everything else is imported from submodules.

### The data pipeline

`PreprocessingConfig` chooses how each column type is prepared; every workflow takes one as `preprocessing=`:

| Option | Choices (default first) |
|---|---|
| `numeric_impute` | `"median"`, `"mean"` |
| `scaling` | `None` = workflow default (standard for linear models, k-means, PCA; none for trees), `"standard"`, `"robust"`, `"minmax"`, `"none"` |
| `categorical_impute` | `"most_frequent"`, `"missing_category"` (missing becomes its own level) |
| `categorical_encoding` | `"onehot"`, `"ordinal"` (linear models warn: codes imply an order) |
| `min_category_frequency` | `None`, a count, or a share: rarer levels are pooled into one "infrequent" level, which also absorbs unseen levels |
| `missing_indicators` | `False`, `True` (0/1 "was missing" column per numeric feature with training gaps) |

Datetime columns become days-since-1970, year, month, and day of week. Every supervised result carries `preprocessing` (a `PreprocessingSummary`: per column, the fitted steps such as "missing values filled with training median 42.5" and the model columns it became) and `cross_validation` (`CrossValidationResult`: per-fold scores, mean, and sample SD; `cv_folds=0` disables it).

Data-quality checks (`core/quality.py`) flag constant, entirely or mostly (>50%) missing, row-identifier-like, and very high-cardinality columns. They appear in `Dataset.describe().issues`, in workflow warnings for the chosen features, and in the app's data checks. They are reported, never acted on.

### Adding a new workflow

Copy the shape of an existing one:

- A statistical test → `statistics/association.py` or `statistics/groups.py`: validate types from the schema, compute with scipy, return a frozen result dataclass with warnings.
- A supervised model → `regression/linear.py` (or `classification/logistic.py`): validate target/features (`_problem.py` helpers), build a `Pipeline` with `build_preprocessor(..., config=...)`, train/test split, evaluate both splits via `evaluation`, cross-validate the training rows, attach `describe_preprocessor(...)`, return a result dataclass.
- An unsupervised method → `clustering/kmeans.py`: no target and no held-out set, so preprocessing is fitted on all rows.

## Desktop app

A PySide6 desktop shell, launched with a single entry point:

```bash
pip install -e ".[gui]"
python -m mlcookbook
```

Three layers, one seam:

- `gui/` — dumb presentation: widgets render data and emit signals (`FrontPage.recipe_selected`, screens' `back_requested`, `load_requested`, `run_requested`). Never imports the core. Visual constants (including a categorical palette) live in `gui/theme.py`; the card grid is data-driven from `gui/recipes.py`.
- `src/mlcookbook/` — the analytical core, UI-free as ever (`mlcookbook/__main__.py` is only a trampoline to the app).
- `app.py` (repo root) — **the only module that knows both sides.** Owns the window, navigation, background jobs, the built-in demo datasets (`DEMOS`), and the report builders (`REPORT_BUILDERS`) that turn core results into plain report dicts.

Seven recipes are ready. Each opens on a built-in scikit-learn dataset, then accepts your own CSV/Parquet file:

| Recipe | Demo data | Report |
|---|---|---|
| Linear Regression | diabetes | actual vs predicted, residual plot, test/train/CV metrics, residual checks, coefficients (OLS, ridge, or lasso) |
| Logistic Regression | breast cancer | fitted sigmoid (binary) or recall by class (multiclass), metrics, confusion matrix, odds ratios or softmax coefficients |
| Decision Trees | breast cancer | the drawn tree (colored by class, or by predicted value for regression), depth-vs-overfit curve, importances, rules |
| Random Forests | breast cancer | ROC curve (binary), recall by class (multiclass), or actual vs predicted (regression); forest structure; importances |
| K-Means | wine measurements | clusters on two principal components, elbow and silhouette scans, cluster sizes and profiles |
| PCA | wine measurements | scree plot, rows on PC1/PC2, loadings |
| Association Scan | breast cancer | strongest associations, every pair with adjusted p-values, a closer look at the strongest pair |

Every report ends with the preprocessing table (what was fitted on which rows) and any warnings. Supervised reports include cross-validation (mean ± SD over folds of the training rows).

### Analyze your own data

1. Open a recipe. For CSV, choose the delimiter and encoding, and optionally extra missing-value markers (e.g. `?, -999`), then click **Open CSV / Parquet**.
2. Review the **data checks** (import conversions and quality issues) and each column's type. Columns that look like IDs, are constant, or are entirely missing start unchecked. Types are editable roles, not conversions of the underlying values.
3. Choose the target and features. Decision trees and random forests pick classification or regression from the target (a numeric target with more than 20 values means regression) and let you switch. Classification works with 2 to 20 classes: two classes let you pick the positive class; more run as multiclass.
4. Adjust **model settings** (penalty, depth, trees, k, ...) and **Preprocessing & validation** (imputation, scaling, encoding, rare levels, missing flags, CV folds).
5. Click **Run Analysis**. Import and training run in background threads; reports show the dataset, target, features, and settings used.

The loaded dataset stays available when switching recipes during the session; each screen asks for its selections again. Nothing is saved across app restarts. Import errors preserve the previous dataset; failed runs show an error and keep the previous report.

Import rules: CSV files need a header with unique, non-empty column names. Numeric-looking CSV columns with leading zeros (`01234`) stay text; text columns whose every value is an ISO-8601 date/time are parsed as dates (offsets converted to UTC); ambiguous formats such as `05/06/2024` stay text. Parquet preserves stored types; nested list/struct columns must be flattened. Files are loaded into memory, so datasets must fit in RAM.

Library use:

```python
from mlcookbook.io import load_dataset, load_dataset_with_report

data = load_dataset("measurements.csv", delimiter=";", encoding="cp1252", missing_markers=["?", "-999"])
data, report = load_dataset_with_report("measurements.csv")
report.kept_as_text, report.parsed_dates, report.notes
```

Screens share one layout (`gui/screens/report_screen.py`) that renders a plain report dict; visuals are drawn with `QPainter` (`gui/plot.py`, `gui/tree_plot.py`), so there is no plotting dependency. See "Desktop GUI conventions" in `AGENTS.md` for the full contract.

A standalone Windows build (PyInstaller) is produced into `dist/MLcookbook/MLcookbook.exe` — see the packaging notes in ADAM.md.

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
4. `src/mlcookbook/preprocessing/` — config, builder, fitted summary: the leakage-safe pipeline
5. `src/mlcookbook/evaluation/cross_validation.py` — why one split is not enough
6. `src/mlcookbook/statistics/` — the statistical tests and the type-driven scan
7. `src/mlcookbook/regression/linear.py` — the ML-workflow template
8. `src/mlcookbook/classification/` and `clustering/`, `dimensionality/` — the remaining workflows

Then `ROADMAP.md` for what is still open.
