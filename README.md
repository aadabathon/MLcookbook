# ML Cookbook

A statistics, data science, and classical machine-learning workbench for arbitrary tabular datasets — and a learning environment for building one deliberately.

**New to the app? Start with the [recipe tutorial](TUTORIAL.md):** all twelve recipes, one reusable practice CSV, how to read results, all three local LLM engines, and a repeatable prompt-research walkthrough.

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

That release predates the current Data-first UI and local LLM workflows. Current repository builds include them.

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
| `wearable` | Verified movement-study import, whole-group baseline comparison, optional fixed test, probabilities and source-window lineage |
| `llm` | `generate_text`, optional PyTorch/Transformers inference from local GPT-2/Llama/Qwen2/Qwen3 checkpoints, frozen text-generation results |
| `ollama` | `ollama_models`, `describe_ollama_model`, `generate_ollama`: installed local model discovery and streamed generation via standard-library HTTP; no PyTorch required |
| `llamacpp` | `llamacpp_models`, `generate_llamacpp`: direct local llama-server discovery and chat generation over SSE, optional explicitly selected GGUF file hashing; no PyTorch required |
| `llm_workflow` | `run_workflow`, `derive_dataset`, `compare_runs`, `export_workflow`: bound templates, strict JSON validation, row lineage, typed derived columns and repeatable server-backed research runs |
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

The [UI roadmap](UI_ROADMAP.md) now includes a persistent sidebar with **Data** first, followed by **Recipes**, **Local LLMs** and **Movement Study**. The app opens on Data: import CSV/Parquet, inspect the first 20 rows, source roles, missing values and data checks, then choose a recipe. The sidebar identifies the active dataset; each recipe also identifies its own input. Typed prompts and fixed-group movement studies retain their separate input context.

Data shows source roles read-only. Adjust role overrides inside each recipe; those edits remain scoped to that analysis and its saved experiment. Navigation restores unrun edits and completed reports without retraining. A failed import preserves the active data and recipe sessions; a successful replacement invalidates prior recipe results. Opening an ordinary saved experiment makes its embedded snapshot active, while opening a Movement Study snapshot retains any existing ordinary dataset. Data's CSV options persist separately from recipe controls.

Opening a recipe never constructs a preset or runs an analysis. It restores your last view or waits for data and settings. **Run example** explicitly loads and runs a bundled dataset in the background. Afterward, **View example** restores that example's settings/results without rerunning, so even invalid unrun edits can be reopened and repaired. **Use my data** restores the recipe's own-data results and unrun edits. Examples stay separate from the active dataset; saved examples reopen offline without becoming ordinary input. Typed LLM prompts work without imported rows; row and structured workflows require a dataset.

The shared visual system keeps sans-serif typography, neutral surfaces, sentence-case headings, clear focus states and restrained accent colors. Ready cards omit decorative symbols and repeated badges.

Recipe forms separate data/columns, model settings and preprocessing/evaluation. Preprocessing, the training-CV setting and the split/seed context stay accessible when model settings are collapsed. Model Comparison's editable split and seed sit beside evaluation controls. The result banner identifies stale results; the footer reports readiness, progress or an actionable error. **Run details** expands the displayed report's data/settings and any recorded timestamp/runtime information. This disclosure is temporary and does not change saved experiment settings.

**Recipe reference** explains applicability, data assumptions, controls, interpretation and limitations for every ready recipe. Its Python API section starts collapsed. Local LLM settings are grouped by task, runtime, inputs, generation and limits; Movement Study opens its grouped evaluation controls with the final test sealed. These presentation changes retain the existing `.mlcb` format and analytical results.

The Data shell opens before scientific libraries load. First use of an analysis prepares the existing runtime in a background job, with a visible status and navigation locks; it does not run a model. File/snapshot imports also initialize their core work in a worker. A failed initialization retains the current view and can be retried. The Python package keeps the same public names and real functions/classes, resolving each export when requested; `import mlcookbook` alone does not import the scientific stack or optional model runtimes.

Adam's existing Windows shortcut now uses the refreshed UI at `dist/MLcookbook/MLcookbook.exe`. The complete pre-redesign app is preserved at `dist/ui-original/MLcookbook`; earlier UI previews remain in `dist/ui-stage1` through `dist/ui-stage4`. Keep every executable with its matching `_internal` folder.

To review future UI changes separately, run `./build_windows.ps1 -Preview`, then open `dist/ui-preview/MLcookbook/MLcookbook.exe`. This preview build leaves the installed `dist/MLcookbook` bundle intact.

A PySide6 desktop shell, launched with a single entry point:

```bash
pip install -e ".[gui]"
python -m mlcookbook
```

Three layers, one seam:

- `gui/` — dumb presentation: widgets render data and emit signals (`FrontPage.recipe_selected`, screens' `back_requested`, `load_requested`, `run_requested`). Never imports the core. Visual constants (including a categorical palette) live in `gui/theme.py`; the card grid is data-driven from `gui/recipes.py`.
- `src/mlcookbook/` — the analytical core, UI-free as ever (`mlcookbook/__main__.py` is only a trampoline to the app).
- `app.py` (repo root) — **the only module that knows both sides.** Owns the window, navigation, background jobs, the built-in demo datasets (`DEMOS`), and the report builders (`REPORT_BUILDERS`) that turn core results into plain report dicts.

Twelve recipes are ready. Ten classical recipes accept CSV/Parquet files and offer explicit bundled examples. Local LLMs waits for a chosen local model. Movement Study waits for a verified wearable export and preserves its fixed group splits:

| Recipe | Demo data | Report |
|---|---|---|
| Movement Study | explicitly selected wearable export.json | fixed-group baseline comparison, sealed-test choice, offline snapshots and prediction/source-window lineage |
| Local LLMs | typed prompt or text rows; PyTorch model folder, Ollama or llama.cpp server | free text or validated nullable fields, derived datasets, saved-run comparisons, usage/timings, model provenance and export |
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

1. Import CSV/Parquet from **Data**, then open a recipe. Nothing runs automatically. For another CSV delimiter, encoding or missing-value markers (e.g. `?, -999`), expand **CSV import options** before opening the file. **Run example** is a separate explicit action.
2. Review the **data checks** (import conversions and quality issues) and each column's type. Columns that look like IDs, are constant, or are entirely missing start unchecked. Types are editable roles, not conversions of the underlying values.
3. Choose the target and features. Decision trees, random forests, SVMs, and neural networks pick classification or regression from the target (a numeric target with more than 20 values means regression) and let you switch. Classification works with 2 to 20 classes: two classes let you pick the positive class; more run as multiclass.
4. Expand **Model settings** for penalty, depth, trees or cluster count. **Preprocessing & validation** stays outside that collapsed group; use it for imputation, scaling, encoding, rare levels, missing flags and training-CV folds. The evaluation summary identifies the test split and seed, or full-data fitting for unsupervised recipes.
5. Click **Run Analysis** in the fixed bottom action bar. Progress and errors stay visible while the form and report scroll. Import and training run in background threads; reports show the dataset, target, features, and settings used. A banner marks results from previous settings until you rerun (or revert your changes).

The loaded dataset stays available when switching recipes. Each recipe remembers its selections, all model parameters, preprocessing, import options, expanded controls, scroll positions, and last report for the session. Returning restores the report without retraining, including any unrun changes and their stale-result banner. A successful file replacement clears all previous recipe sessions; a failed import preserves them. Use **Save experiment** to retain the current recipe across app restarts. Failed runs show an error and keep the previous report and the settings that produced it.

**Data checks** remains accessible after a run. **Recipe reference** explains the method and its limitations, with the Python callable and signature in a collapsed section. Recipe cards support Tab navigation and Enter/Space activation.

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

```python
from mlcookbook import compare_regressors

comparison = compare_regressors(data, target="y", random_state=17, cv_folds=5)
comparison.selected_model, comparison.test_metrics
comparison.train_rows, comparison.test_rows, comparison.folds
comparison.to_dict()  # JSON-safe, including both candidates' CV metrics
```

`mlcookbook.comparison` owns shared-fold comparison and its frozen results. `mlcookbook.experiments.save_experiment` / `load_experiment` handle the snapshot format without importing GUI code; the coordinator supplies and restores the plain screen state. The shared CV helper accepts explicit positional splits for comparison and rejects overlapping training/validation rows.

### Local LLM generation with PyTorch

This recipe is available in current source/local builds. The published v1.0.0 Windows download contains the ten classical recipes.

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

1. Open **Local LLMs**, select **PyTorch / Transformers**, choose the model folder, and edit **Prompt**. **Auto** applies the tokenizer's chat template if present; **Plain text** sends the raw prompt.
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

This is an inference workflow: no sklearn preprocessing, train/test split, cross-validation, training, or accuracy evaluation is performed. Generated text can be wrong. PyTorch uses float32 on CPU and float16 on CUDA; hardware/software changes and time limits can affect repeatability even with a recorded seed. Training/fine-tuning, embeddings, and graph composition remain separate future work. The existing Neural Networks recipe continues to use scikit-learn's MLP.

### Local LLM generation with Ollama

The same **Local LLMs** screen also connects to an existing local [Ollama server](https://docs.ollama.com/api/introduction). This connector uses Python's standard library and needs neither the `llm` extra nor a browser. Ollama itself and its models are installed separately.

1. Start Ollama and choose **Inference engine → Ollama**. Keep `http://127.0.0.1:11434` unless your local server uses a different port.
2. Click **Test / refresh models**, then select an installed model. The connection panel shows runtime version, model size, quantization, and model context when supplied. Only loopback HTTP endpoints and installed local models are accepted; cloud references are excluded. The app never pulls models or manages the server process.
3. Write a prompt and click **Generate text**. **Auto** and **Chat template** submit a user message for the server to format; **Plain text** requests raw generation. Output tokens, temperature/top-p, seed, context allocation, thinking, row limit, and time limit are request controls. Device, precision, GPU offload, and model loading are managed by Ollama. Thinking **On/Off** require boolean support; choose **Model default** for models exposing thinking levels.
4. Watch **Live preview** while the request runs. **Cancel** also interrupts an idle request during model loading; the server remains running. The per-row deadline includes loading (the screen defaults to 90 seconds). Failed, timed-out, cancelled, or incomplete streams discard transient output and preserve the previous completed report.
5. Use **Dataset rows** for the same bounded text-column workflow described above. **Save experiment** retains outputs, submitted prompts/messages, requested options, source positions, model digest/metadata, Ollama version, and server token/timing data. Reopening needs no server or weights, and restores unrun edits. Older PyTorch experiment files remain supported.

```python
from mlcookbook.ollama import ollama_models, generate_ollama

connection = ollama_models()  # discovery only; does not load model weights
result = generate_ollama(
    model=connection.models[0].name,  # choose an installed local model
    prompt="Explain training-only preprocessing in one sentence.",
    max_new_tokens=64, max_seconds=90, thinking="off",
    on_progress=lambda event: None,  # accumulated text/thinking, output number, source position
)
result.outputs[0].text
result.to_dict()
```

Ollama's reported token counts and server durations are distinct from local wall time; missing metrics remain unavailable. Options are requests, not verified applied settings. The server manages tokenization/context, so the client cannot guarantee no input truncation or recover the exact rendered chat prompt. Model names, seeds, and digests do not guarantee identical output across runtime/hardware changes. Loopback restricts the connection destination; Ollama's own cloud configuration remains outside this app.

### Local LLM generation with llama.cpp

**Inference engine → llama.cpp server** connects to an existing single-model [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), using `/health`, `/v1/models`, read-only `/props`, and streamed `/v1/chat/completions`. It uses the standard library, needs no browser or `llm` extra, and never installs, starts/stops, loads, or downloads a server/model. Router mode, authenticated endpoints, tools, and multimodal requests are outside this milestone.

Start a server separately with your local GGUF. This machine's verified CPU example uses the QA checkpoint already saved in the ignored `models/` folder:

```powershell
llama-server -m models/SmolLM2-135M-Instruct.Q4_K_M.gguf --alias smollm2-135m --host 127.0.0.1 --port 8080 --ctx-size 2048 --parallel 1 --n-gpu-layers 0 --threads 4 --jinja --offline
```

The example checkpoint is [QuantFactory/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF/tree/476854d00ede130660aba430d15f9347ad2e7d0e), file `SmolLM2-135M-Instruct.Q4_K_M.gguf`, SHA-256 `8030f04528538d47bda434f6f0bdf3952c40a58123e4d5e755332f23731a8684`. It is a small inference smoke-test model, not a quality recommendation. Checkpoints and server executables stay outside the app bundle. The installed server build `b11193-4e7481175` could not load the existing Ollama Qwen GGUF because its rope metadata format differed; compatibility depends on both model and runtime.

1. Open **Local LLMs → llama.cpp server**, set the local address (default `http://127.0.0.1:8080`), then click **Test / refresh models**. Loading status is visible; refresh when ready. The selector uses the loaded model's exact identity. A changed identity is rejected before generation.
2. Write a prompt or select **Dataset rows**. The server formats one user chat message per input; the PyTorch raw/template controls are hidden. Output limit, temperature/top-p, seed, row limit and per-row deadline are request controls. Context, GPU offload, precision and thinking remain server settings. SSE streams may carry separate reasoning text.
3. Generate, watch the transient preview, or cancel. Cancellation interrupts idle reads without stopping the external process. Invalid/incomplete streams, timeouts and failures retain the previous completed result. Usage-only terminal events and the SSE completion marker must finish the response before it is committed.
4. Optionally select **GGUF file** to hash that local file during a run. The app never reads a filesystem path merely because the server supplied it. This hash identifies only your selected file; its binding to the server's loaded bytes, other shards, adapters or chat template is unverified. Without a selected file, the report explicitly retains an unverified model reference.
5. Save/Open retains submitted messages, outputs, source positions, requested and explicitly reported applied settings, model/server metadata, reported runtime version, actual usage/timings, and optional file provenance. Reopening contacts no server and reads no model file. Reruns need the server/model, plus the selected GGUF if hashing is enabled (or clear that optional field). Older PyTorch and Ollama experiments migrate before validation.

```python
from mlcookbook.llamacpp import llamacpp_models, generate_llamacpp

connection = llamacpp_models()  # Does not load weights; status can be "loading".
result = generate_llamacpp(
    model=connection.models[0].name, prompt="Explain gravity in one sentence.",
    max_new_tokens=64, max_seconds=90,
    # gguf_path="models/my-model.gguf",  # optional file hash; never loads a model
)
result.outputs[0].text
result.to_dict()
```

Server-reported token counts and millisecond timings remain distinct from local wall time. Unreported usage, versions, or timings stay unavailable. Exact rendered prompts and server input truncation remain unverified. Both connectors share a small private `_local_http` module for bounded loopback-only HTTP; Ollama NDJSON and llama.cpp SSE parsing remain separate.

### Repeatable LLM text-to-dataset workflows

In **Local LLMs**, choose **Extract fields**, **Summarize rows**, **Rewrite rows**, or **Classify rows** with Ollama or llama.cpp. Import text data, select the prompt input columns, and click **Use task template with selected columns**. Templates bind column names literally, such as `{{notes}}`; unchecked columns never enter the request. Define one output per line:

```text
extracted_bedrooms: numeric
summary: text
sentiment: categorical: positive | neutral | negative
urgent: boolean
```

Names must be new columns. Roles determine nullable output storage (`Float64`, string, boolean); no string-to-number conversion or JSON repair occurs. Every field is required in the JSON object, with `null` permitted for unknown facts. Ollama receives a `format` schema and llama.cpp a `response_format` schema; the core independently rejects missing/extra/duplicate fields, wrong types and unknown labels. The [tutorial](TUTORIAL.md#structured-text-workflows) includes a complete walkthrough. Free text generation remains available on all three engines.

The row limit is **1–10 source positions**, including blank/skipped rows, for structured workflows. Each row records its submitted prompt, raw answer, validated values, errors, timings and backend provenance. Cancellation retains completed rows and marks unfinished rows. **Review rows / retry / use as dataset** shows source text beside generated fields, plus exact prompts/raw output. Selected/failed-row retries require matching source/configuration/model/runtime identity; changed controls disable retry/acceptance until a full run. llama.cpp identities remain unverified server references even with an explicitly selected GGUF hash; its response timestamp is not a model fingerprint.

**Use as derived dataset** explicitly copies all source rows/columns and appends nullable generated columns by source position plus a full typed dataset fingerprint. Duplicate index labels are preserved. Invalid, skipped, cancelled and unprocessed rows stay missing. Click Back and open another recipe to use this dataset; Local LLMs retains the original source run. Save an analysis experiment to persist the derived dataset and lineage, or save the LLM experiment to retain source data, templates, unrun edits and up to ten recent runs. Both reopen without inference servers or original input files.

**Compare saved runs** selects two or more runs on the same source positions, output schema and reference bindings. Native charts show validation rate, wall time, tokens per wall second, server-reported load/decode durations where available, and output length. Wall time includes discovery, optional file hashing and failed attempts; inherited retry rows retain their original timings. It is an artifact comparison, not a fresh benchmark. Optional **Reference labels** uses `output_field=source_column`, one per line, and reports exact field agreement on valid, non-null pairs with evaluated-row counts. Reference columns cannot be prompt bindings. Validation rate uses all selected source rows and does not measure semantic accuracy.

**Export current run** writes UTF-8 JSON with source values, row outcomes, metadata and metrics. `.mlcb` remains the typed, checksummed source snapshot and offline-reopening format. Generated columns can still reveal target information: inspect inputs and freeze prompt/model choices before assessing held-out predictions.

The [roadmap](ROADMAP.md#local-llm-and-workflow-graphs) continues with graph execution and a native Qt canvas for connecting these operations to existing recipes.

The same operations are available without Qt:

```python
from mlcookbook.llm_workflow import run_workflow, derive_dataset

run = run_workflow(
    data, backend="ollama", model="your-installed-model", base_url="http://127.0.0.1:11434",
    template="Extract the bedroom count from this description:\n{{notes}}",
    bindings=["notes"], field_spec="extracted_bedrooms: numeric", max_rows=3,
    settings={"max_new_tokens": 128, "temperature": 0.0},
)
# Inspect run.rows first. Accept explicitly after reviewing raw text/errors:
derived, lineage = derive_dataset(data, run.to_dict())
```

## Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests: `ruff check .` (rules configured in `pyproject.toml`), the library tests on Ubuntu with Python 3.12 and 3.13 (plus the example tour), and the full suite including the PySide6 app tests on Windows with Qt's offscreen platform. A separate Ubuntu CPU LLM job installs `.[dev,gui,llm]` and Qt runtime libraries, then runs tiny-checkpoint and fake Ollama/llama.cpp HTTP tests entirely offline. No external inference server or model download is needed for tests. Locally: `ruff check .` and `pytest` using the project environment.

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


## Wearable movement studies

**Movement Study → Open movement study…** opens the `export.json` made by the
PSoC wearable analysis pipeline (`wearable.py export-mlcookbook`). Keep its six
split/index CSV files beside the manifest. Import verifies hashes, feature/label
alignment, physical units, whole-person/session assignments and class coverage.
IDs and clock fields are retained as provenance and cannot become predictors.

**Run grouped baselines** compares standard-scaled logistic regression and a
bounded random forest on the fixed validation groups. CV keeps training groups
together and fits preprocessing inside each fold; missing group/class coverage
produces a warning, never a row-wise fallback. **Reserved test groups** defaults
to **Keep sealed**. Evaluate that holdout only after freezing study decisions.

Save/Open experiment retains the complete typed feature/lineage snapshot,
settings and last report; reopen/rerun needs no original CSVs or PSoC repository.
**Export evaluation…** writes JSON containing metrics, probabilities and exact
source-window/session/time lineage. Loading this study does not expose reserved
holdouts to other recipes. The companion CLI calls the same native evaluator.

```python
from mlcookbook import load_wearable_study, evaluate_wearable

dataset, imported = load_wearable_study("study/export.json")
result = evaluate_wearable(dataset, cv_folds=5, final_test=False, random_state=42)
result.to_dict()  # JSON-safe metrics, groups, folds and predictions
```

These are classical window baselines. DEEPCRAFT still trains/exports the temporal
CNN; window scores do not establish fall-event recall, false alerts per hour or
on-device performance. Ordinary recipe row-wise scores remain exploratory for
overlapping movement windows. Synthetic studies are explicitly marked and warn
that their scores only check software integration. No new MLcookbook dependency
or separate app entry point was added.


The same desktop entry point accepts direct offline handoffs:

```powershell
python -m mlcookbook --movement-study
python -m mlcookbook --study C:/path/to/study/mlcookbook/export.json
python -m mlcookbook --experiment C:/path/to/study.mlcb
# The portable EXE accepts the same arguments.
```

Opening verifies/imports the study or restores the saved snapshot without fitting
a model. The empty Movement Study card and new imports default to sealed test
groups; saved experiments retain their explicit unrun choices without running
them. Qt command-line options are forwarded separately. Failed imports leave
the app usable with the error in its status area. The PSoC companion now provides
`wearable.py study` to assemble prepared windows/CNN tracks, feature exports and
native grouped baselines in one new folder, plus `verify-study` and a PowerShell
launcher. Its `study.json` hashes link all derived outputs; raw sessions remain
separate evidence. Synthetic provenance propagates from sessions automatically,
and mixed real/synthetic preparation is rejected.
