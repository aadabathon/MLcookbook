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
                        decision_tree_regression, random_forest, random_forest_regression, svm,
                        svm_regression, neural_network, neural_network_regression, kmeans, pca)

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
svm(data, target="churned", kernel="rbf", C=1.0).support_vectors
neural_network(data, target="species", hidden_layers=(64, 32)).training.loss_curve
neural_network_regression(data, target="price").training.validation_curve

# Unsupervised learning
kmeans(data, n_clusters=3).profiles     # cluster profiles in original units, k scan, silhouette
pca(data).components[0].loadings        # explained variance, loadings, 2-D scores
```

Run `python examples/basic_usage.py` for a complete tour.

## Installation

### Windows desktop download

Download `MLcookbook-v1.0.0-windows-x64.zip` from the [v1.0.0 release](https://github.com/aadabathon/MLcookbook/releases/tag/v1.0.0), extract the entire archive, and open `MLcookbook/MLcookbook.exe`. Keep the adjacent `_internal` folder with the executable. Python is bundled; no separate Python installation is needed.

Start with a built-in demo or import CSV/Parquet data. **Save experiment** writes a `.mlcb` snapshot of the dataset, current recipe settings, and last report; **Open experiment** restores it. See the [release notes](CHANGELOG.md) for the v1 scope.

### From source

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
| `regression` | `linear_regression` (OLS/ridge/lasso + residual diagnostics), `decision_tree_regression`, `random_forest_regression`, `svm_regression`, `neural_network_regression` |
| `classification` | `logistic_regression`, `decision_tree`, `random_forest`, `svm`, `neural_network` — binary or multiclass; shared validation in `_problem.py` |
| `clustering` | `kmeans` |
| `dimensionality` | `pca` |
| `llm` | `generate_text`, optional PyTorch/Transformers inference from local GPT-2/Llama/Qwen2/Qwen3 checkpoints, frozen text-generation results |
| `visualization` | 🚧 skeletal — the desktop app draws its own charts from result data |

### Design rules

1. **Types drive applicability.** Every column gets a `VarType` (a statistical role, not a pandas dtype). Analyses validate their inputs against the schema and refuse or warn rather than silently coercing. Users can override any inference.
2. **Results are data.** Every analysis returns a frozen dataclass mixing in `Result`, which provides `to_dict()` via a recursive serializer (`core/results.py`) that handles nested dataclasses, numpy scalars/arrays, enums, and NaN → `null`. No analysis prints or plots.
3. **Warnings are part of the result.** Dropped rows, skipped columns, constant/ID-like/mostly-missing features, class imbalance, overfitting, and failed assumptions are surfaced in `result.warnings`, not swallowed.
4. **No leakage.** Preprocessors are returned unfitted and are fitted inside a `Pipeline` on training data only. Cross-validation runs on the training rows and refits the whole pipeline, preprocessing included, inside every fold; the test split is touched once.
5. **Libraries do the math.** Classical workflows use sklearn/scipy; the optional LLM recipe uses PyTorch and Transformers. This library orchestrates and validates.
6. **Small public API.** The workflows, their result types, `Dataset`, `VarType`, `PreprocessingConfig`, and `load_dataset` are exported at the top level; everything else is imported from submodules.

### The data pipeline

`PreprocessingConfig` chooses how each column type is prepared for classical ML workflows via `preprocessing=`. LLM generation instead uses the chosen model's tokenizer and chat template:

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

Eleven recipes are ready. The ten classical recipes open on built-in scikit-learn datasets, then accept your own CSV/Parquet file. The LLM recipe opens a setup screen and waits for an explicitly chosen local model:

| Recipe | Demo data | Report |
|---|---|---|
| Local LLM · PyTorch | typed prompt or example question rows; local model required | generated text, token counts, throughput, stop reasons, model fingerprint, and runtime versions |
| Model Comparison | diabetes | ridge vs random forest regression on identical training folds, selected model's final test metrics, exact row/fold membership saved with the report |
| Linear Regression | diabetes | actual vs predicted, residual plot, test/train/CV metrics, residual checks, coefficients (OLS, ridge, or lasso) |
| Logistic Regression | breast cancer | fitted sigmoid (binary) or recall by class (multiclass), metrics, confusion matrix, odds ratios or softmax coefficients |
| Decision Trees | breast cancer | the drawn tree (colored by class, or by predicted value for regression), depth-vs-overfit curve, importances, rules |
| Random Forests | breast cancer | ROC curve (binary), recall by class (multiclass), or actual vs predicted (regression); forest structure; importances |
| Support Vector Machines | breast cancer | ROC curve (binary), recall by class, or actual vs predicted (regression); support vectors; linear-kernel weights |
| Neural Networks | wine cultivars (3 classes) | training-loss and validation curves, metrics, architecture and parameter count |
| K-Means | wine measurements | clusters on two principal components, elbow and silhouette scans, cluster sizes and profiles |
| PCA | wine measurements | scree plot, rows on PC1/PC2, loadings |
| Association Scan | breast cancer | strongest associations, every pair with adjusted p-values, a closer look at the strongest pair |

Reports include a short explanation of the result, followed by the recipe's visuals, metrics, and detailed tables. Every report ends with the preprocessing table (what was fitted on which rows) and any warnings; a warning summary above the report also opens the complete warning list. Supervised reports include cross-validation (mean ± SD over folds of the training rows). Long tables scroll within a bounded area.

### Analyze your own data

1. Open a recipe. Its built-in example runs in the background. For your own CSV, expand **CSV import options** if you need a different delimiter, encoding, or extra missing-value markers (e.g. `?, -999`), then click **Open CSV / Parquet**.
2. Review the **data checks** (import conversions and quality issues) and each column's type. Columns that look like IDs, are constant, or are entirely missing start unchecked. Types are editable roles, not conversions of the underlying values.
3. Choose the target and features. Decision trees, random forests, SVMs, and neural networks pick classification or regression from the target (a numeric target with more than 20 values means regression) and let you switch. Classification works with 2 to 20 classes: two classes let you pick the positive class; more run as multiclass.
4. Expand **Model & validation settings** to adjust model settings (penalty, depth, trees, k, ...) or open **Preprocessing & validation** (imputation, scaling, encoding, rare levels, missing flags, CV folds).
5. Click **Run Analysis** in the fixed bottom action bar. Progress and errors stay visible while the form and report scroll. Import and training run in background threads; reports show the dataset, target, features, and settings used. A banner marks results from previous settings until you rerun (or revert your changes).

The loaded dataset stays available when switching recipes. Each recipe remembers its selections, all model parameters, preprocessing, import options, expanded controls, scroll positions, and last report for the session. Returning restores the report without retraining, including any unrun changes and their stale-result banner. A successful file replacement clears all previous recipe sessions; a failed import preserves them. Use **Save experiment** to retain the current recipe across app restarts. Failed runs show an error and keep the previous report and the settings that produced it.

**Data checks** remains accessible after a run. **About this recipe** contains the Python callable and signature. Recipe cards support Tab navigation and Enter/Space activation.

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

### Save, reopen, and compare

1. Open **Model Comparison**, import your CSV/Parquet, and select a numeric target and features.
2. Expand **Model & validation settings** to choose the split/model seed, test fraction, ridge penalty, and forest settings. Choose CV folds and shared preprocessing in **Preprocessing & validation**. Comparison requires CV.
3. Run. Ridge and random forest use one materialized split and the same training-only folds. Each pipeline fits preprocessing inside each fold. The lowest mean CV RMSE selects the model (ridge wins an exact tie); only the selected model is refitted on all training rows and evaluated on the held-out test rows. Mean ± sample SD and per-fold curves show the variation in validation scores.
4. Click **Save experiment** in the app header. The `.mlcb` file includes a copy of the full dataset, current controls (including unrun changes), the last completed report and its request, original import options, package versions, run timestamp/seed, and a SHA-256 fingerprint of the data snapshot. Comparison reports also retain the actual train/test row positions and CV folds.
5. After restarting the app, use **Open experiment** from the header. The report reappears without training. The original CSV/Parquet may have moved or been removed; reopening uses the embedded snapshot. Click **Run Analysis** to rerun with the restored settings. Data checks flag package-version differences because results can vary across versions.

Saving is explicit and captures **one recipe**, not every open recipe session. It also works for all other recipes and for configurations that have not been run yet. Files use a versioned ZIP containing JSON metadata/results and a typed Parquet snapshot; no fitted model objects or executable Python are stored. Saves replace existing files atomically. Failed opens leave your current dataset and report intact. Successfully opening an experiment clears previous in-memory recipe sessions and makes its dataset available to the other recipes. A saved file contains the dataset itself, so its size and contents follow your data.

The comparison currently supports ridge and random forest regression. Both use the same preprocessing choices (standard scaling by default), unlike the standalone forest recipe's default of no scaling. Repeated tuning after inspecting the test score weakens its independence; reserve new held-out data for a final assessment. Classification comparison, arbitrary workflow graphs, fitted-model persistence, and CSV result export remain future work.

### Local LLM generation with PyTorch

This optional recipe runs pretrained causal language models locally. Install the runtime in your project environment; for a CPU-only Windows build:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e ".[dev,gui,llm]"
.venv/Scripts/python.exe -m mlcookbook
```

For CUDA, install the appropriate PyTorch build using the [official installation selector](https://pytorch.org/get-started/locally/). The portable build produced by `build_windows.ps1` includes the runtime installed in `.venv`; the tested build uses CPU PyTorch.

Choose a local Hugging Face model folder with `config.json`, tokenizer files, and `model.safetensors` (or a sharded safetensors index). Supported `model_type` values are `gpt2`, `llama`, `qwen2`, and `qwen3`; checkpoints must be unquantized. A small starting model is [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), which uses the Llama architecture. To download that specific snapshot once, separately from app generation:

```powershell
.venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM2-135M-Instruct', revision='12fd25f77366fa6b3b4b768ec3050bf629380bac', local_dir='models/SmolLM2-135M-Instruct', allow_patterns=['config.json','generation_config.json','tokenizer*','special_tokens_map.json','vocab.json','merges.txt','model.safetensors'])"
```

The `models/` directory is gitignored. Opening the recipe never downloads or loads weights automatically. The app loads local files only, uses safetensors, and does not execute model-provided Python. GGUF files and Hub IDs are not model-folder inputs.

1. Open **Local LLM · PyTorch**, choose the model folder, and edit **Prompt**. **Auto** applies the tokenizer's chat template if present; **Plain text** sends the raw prompt.
2. Choose an output token limit, temperature (zero means greedy decoding), sampling top-p/seed, input token limit, and device. Oversized inputs fail with an explicit message; they are never silently truncated. Input plus output tokens must fit the model's context window.
3. Click **Generate text**. The report contains selectable/copyable output text, token counts, generation throughput, model load time, stop reasons, and the model fingerprint and runtime versions. **Cancel** stops after loading or at a decoding step and preserves the previous report. The time limit is checked between decoding steps; it cannot interrupt loading or a single forward pass.
4. For row-wise generation, select **Dataset rows**, import CSV/Parquet, and check exactly one string-valued text/categorical column. **Prompt** becomes an instruction prepended to each row. Blank/missing rows are skipped with warnings; at most ten rows are processed per run, with zero-based source positions recorded independently of dataframe index labels.
5. **Save experiment** retains the prompt, generated text, settings, typed dataset, exact formatted prompts, model fingerprint, and package versions. Reopening displays the saved output without loading weights. Reruns require the local model folder; weights are not embedded in `.mlcb` files.

```python
from mlcookbook import generate_text

result = generate_text(
    model_path="models/SmolLM2-135M-Instruct",
    prompt="Explain training-only preprocessing in one sentence.",
    max_new_tokens=64, temperature=0.0, random_state=0,
)
result.outputs[0].text
result.to_dict()  # JSON-safe data; no tensors or fitted model objects

# Optional row-wise generation; data is a Dataset with a string-valued column.
result = generate_text(data, model_path="models/SmolLM2-135M-Instruct",
                       prompt="Summarize:", text_column="notes", max_rows=3)
```

This is an inference workflow: no sklearn preprocessing, train/test split, cross-validation, training, or accuracy evaluation is performed. Generated text can be wrong. CPU uses float32 and CUDA uses float16; hardware/software changes and time limits can affect repeatability even with a recorded seed. Training/fine-tuning, embeddings, quantized runtimes, and graph composition remain separate future work. The existing Neural Networks recipe continues to use scikit-learn's MLP.

```python
from mlcookbook import compare_regressors

comparison = compare_regressors(data, target="y", random_state=17, cv_folds=5)
comparison.selected_model, comparison.test_metrics
comparison.train_rows, comparison.test_rows, comparison.folds
comparison.to_dict()  # JSON-safe, including both candidates' CV metrics
```

`mlcookbook.comparison` owns shared-fold comparison and its frozen results. `mlcookbook.experiments.save_experiment` / `load_experiment` handle the snapshot format without importing GUI code; the coordinator supplies and restores the plain screen state. The shared CV helper accepts explicit positional splits for comparison and rejects overlapping training/validation rows.

## Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests: `ruff check .` (rules configured in `pyproject.toml`), the library tests on Ubuntu with Python 3.12 and 3.13 (plus the example tour), and the full suite including the PySide6 app tests on Windows with Qt's offscreen platform. Locally: `ruff check .` and `pytest`.

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
8. `src/mlcookbook/classification/` and `clustering/`, `dimensionality/` — the remaining workflows (`classification/neural_network.py` for the MLP and its training curves)

Then `ROADMAP.md` for what is still open.
