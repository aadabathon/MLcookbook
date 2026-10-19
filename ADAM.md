# Adam's Handbook

Personal notes for the author. The scaffold is done — from here on, this project is yours. This file is your map: where things are, how to work, and what to build in what order. Edit it freely; it's a living document.

## Local LLM / PyTorch recipe - 2026-10-19

Implemented at my request, with explicit approval for optional `torch` and `transformers` dependencies. **Local LLM · PyTorch** is the eleventh recipe. Choose a local Hugging Face safetensors model, edit a prompt, and generate; **Dataset rows** prepends the instruction to one selected string-valued column. Prompt format, token limits, temperature/top-p, seed, CPU/CUDA, row limit, and decoding time are visible controls. Cancel/failure keeps the previous report. Opening the recipe does not load or download a model.

Start with the local folder `models/SmolLM2-135M-Instruct` (gitignored), downloaded from revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The real model generated 23 tokens in about 1.3 seconds on CPU in the initial check; that is a local observation, not a general performance guarantee. The frozen build includes CPU PyTorch, so use CPU there. A CUDA-enabled source environment can select CUDA.

`llm.py` returns frozen JSON-safe `TextGenerationResult` data: generated and formatted prompt text, input/output token counts, source row positions, stop reasons, timing, model fingerprint, parameter/context size, precision/device, settings, and runtime versions. Save/Open retains outputs and unrun edits without loading weights; reruns require the external model folder. Generated answers are not evaluated analytical results. The existing MLP recipe is unchanged.

Learning points: a tokenizer maps text to token IDs; chat templates add the model's expected role/special-token format; causal decoding appends one token at a time; `eval()` and `torch.inference_mode()` select inference behavior; temperature zero is greedy, while temperature/top-p sample from next-token probabilities. The model context holds both input and generated tokens. A seed helps reproduce sampling on the same runtime, but hardware and decoding time limits matter. Training/fine-tuning, embeddings, quantized inference, and graph composition are separate learning tasks.

The isolated `.venv` has the LLM, dev, GUI, and packaging dependencies. Use `.venv/Scripts/python.exe -m mlcookbook` / `-m pytest`. The optional extra is lazy: classical imports and the setup screen do not initialize PyTorch. Offline tests construct tiny real checkpoints; the separate LLM CI job needs no pretrained-model download. Build with `build_windows.ps1`, as described below. This feature has not been committed or added to the published v1.0.0 download.

## v1.0.0 release - 2026-10-19

The first tagged release packages all ten recipes, the usability improvements, saved experiments, and regression model comparison. The package and experiment provenance now report **1.0.0**. [Release notes](CHANGELOG.md) define the scope; the [GitHub release](https://github.com/aadabathon/MLcookbook/releases/tag/v1.0.0) hosts the portable Windows x64 ZIP and its SHA-256 checksum. Extract the entire ZIP and keep `_internal` beside `MLcookbook.exe`. The local shortcut continues to use `dist/MLcookbook/MLcookbook.exe`.

Release verification: **299 tests passed** on Python 3.12 with pandas 2.2.3; Ruff and `examples/basic_usage.py` passed. The preceding milestone also passed the full suite on pandas 3.0.6. GitHub CI checks the release commit on Linux Python 3.12/3.13 and Windows Python 3.12. PyTorch and graph composition remain separate next steps.

## Saved experiments and comparison - 2026-10-17

Implemented at my request: **Save experiment / Open experiment** in the header, plus **Model Comparison** as a tenth recipe. Try the comparison demo, change a setting, save a `.mlcb`, close the app, and reopen it: the prior report and the unrun change return without retraining. The file embeds the full parsed dataset, so it still opens if the original CSV/Parquet moves. Saving is explicit and captures the current recipe only.

For your own regression study: import data, choose a numeric target/features, choose seed/test fraction and model settings, and run. Ridge and random forest receive identical training folds, with preprocessing fitted separately inside each fold. Lowest mean CV RMSE selects the pipeline, which is then refitted on training rows and evaluated on the reserved test rows. The report shows both candidates' CV curves/table and the selected model's test metrics. Change settings using CV; repeatedly consulting the test score weakens its independence.

The core additions are `comparison.py` (plain frozen comparison results, exact source row/fold positions) and `experiments.py` (versioned JSON + Parquet snapshots, fingerprints, package-version notes, atomic writes). Original import options and report run timestamp/seed/versions are retained. Current controls can differ from the saved report's request, and the stale-result banner survives reopening. Opening a valid experiment replaces the active dataset and clears other in-memory recipe sessions; failures preserve your work.

Learning points: shared seeds alone are less auditable than recorded row membership; CV selects a model while the test set estimates its performance; every fold needs a newly fitted preprocessor; repeatability depends on data, settings, and software versions. Experiment files are readable data, not pickled trained models. Classification comparison, result export, fitted-model persistence, graph composition, and PyTorch remain future work. No dependencies were added.

Validation: **299 tests passed** with the installed pandas 2.2.3 and again with isolated pandas 3.0.6 using CI's Qt offscreen mode. Ruff passed. Native Windows renders covered the home page, comparison curves/table, minimum size, reopened unrun changes, and using the saved dataset in another recipe. The final display refinements passed seven focused GUI checks.

The Windows executable was rebuilt with the isolated build `PATH`, verified to contain the current versions of all nine changed application modules, and launch-tested: a responsive Qt window opened and closed normally. No commit was made for this milestone.

## Quality-of-life update - 2026-10-16

The nine existing recipes now share a fixed Run/status/progress bar, collapsed CSV and model/validation controls, wider column-name space, an About dialog for Python details, and a persistent warning summary. Data checks remain available after a run. Short result explanations use the existing computed metrics; long tables retain all rows in a bounded scroller. Recipe cards work with Tab and Enter/Space.

Navigation remembers each recipe's configuration and last report during the session, including unrun changes, expanded controls, import options, and scroll positions. The result banner tells you when settings differ from the displayed run. Returning does not retrain. Successful file replacements clear old recipe sessions; failed imports and runs preserve previous work. First-visit demos now train in the background.

This is an in-session usability pass: reopening the app still starts fresh. Saved projects/export, repeatable workflow design, and new recipes remain separate future work. The analytical core and dependencies are unchanged.

Verification: **253 pytest tests passed** and `ruff check .` passed. All nine recipe demos plus imported-data, error, dialog, and minimum-window layouts were rendered and inspected on native Windows. The executable was rebuilt with an isolated `PATH` after diagnosing an incompatible Poppler ICU DLL in the earlier bundle. Its changed modules match the source, and the rebuilt app opened a responsive window and closed normally. No commit was made.

## Handoff - 2026-10-12

Shipped at my request in three local commits (not pushed), in this order:

1. **Data pipeline** - `PreprocessingConfig` (median/mean imputation; standard/robust/min-max/no scaling; one-hot/ordinal encoding; rare-level pooling; "(missing)" as a level; numeric missing-value flags), datetime features (days since 1970, year, month, weekday), a `PreprocessingSummary` on every supervised result showing what was fitted on which rows, training-only k-fold cross-validation in every supervised workflow (`cv_folds`, default 5), data-quality checks (constant, mostly missing, ID-like, high-cardinality), and import quality (leading zeros kept, extra missing markers, ISO dates parsed, `ImportReport`).
2. **Modules** - multiclass for all three classifiers, `decision_tree_regression`, `random_forest_regression`, ridge/lasso on `linear_regression` plus residual diagnostics, `kmeans`, `pca`, `chi_square`, `anova` (with Kruskal-Wallis), correlation confidence intervals and normality warnings, and `association_scan`.
3. **GUI** - seven ready recipes (K-Means, PCA, Association Scan are new), a Task switch on trees/forests, multiclass targets, model settings, the Preprocessing & validation dialog, extra missing markers on import, data checks, and cross-validation/preprocessing tables in every report.

Verification at the three-phase handoff: **207 pytest tests passed**; `examples/basic_usage.py` ran end to end; every recipe's demo plus multiclass-tree, regression-tree/forest, multiclass-logistic, and data-check screens were rendered and inspected; the Windows exe was rebuilt and launch-tested.

Interview prep (the parts worth being able to explain): why preprocessing must be refitted inside each CV fold; why CV runs on training rows only; ridge vs lasso geometry; what residual normality/heteroscedasticity tests do and don't imply; macro vs per-class metrics and one-vs-rest AUC; silhouette vs inertia for choosing k; why PCA needs scaling; Cramér's V vs the chi-square p-value; when Kruskal-Wallis beats ANOVA; why many tests need Benjamini-Hochberg.

Follow-up the same day: **SVM** (`svm`, `svm_regression`: rbf/linear/poly kernels, C, support vectors, Platt-scaled probabilities, target standardized for SVR so C and epsilon are unit-free) and **neural networks** (`neural_network`, `neural_network_regression`: scikit-learn MLP, Adam with batches of 32, L2 alpha, early stopping on 10% validation rows, loss and validation curves, parameter count) with recipes for both, plus **CI** (`.github/workflows/ci.yml`: ruff, library tests on Ubuntu 3.12/3.13, full app suite on Windows offscreen). 229 tests and `ruff check .` pass; the exe was rebuilt and launch-tested. Interview angles: margin vs C, the kernel trick, why SVR and MLPs want a standardized target, what early stopping's validation curve shows, parameters vs rows.

Limits: in-memory CSV/Parquet only, nothing saved across sessions, no result export, no hyperparameter search or model comparison table, no mypy.

## Daily workflow

```powershell
.venv\Scripts\activate            # if using a venv (recommended)
# No venv? This machine has Python 3.11/3.12/3.13 and bare `python` is 3.11 WITHOUT the project.
# Prefix everything with `py -3.12` instead, e.g.  py -3.12 -m pytest   /   py -3.12 -m mlcookbook
pip install -e ".[dev,gui]"       # once, or after changing pyproject.toml
pytest                            # run the full suite
pytest tests/test_dataset.py -k schema   # run one thing while iterating
python examples/basic_usage.py    # smoke-test the public API end to end
python -m mlcookbook              # launch the desktop app
```

App architecture: `gui/` is dumb (widgets + signals), `src/mlcookbook` is UI-free, and root `app.py` is the single seam that knows both — it owns the window, delete-and-redraw navigation, demo datasets (`DEMOS`), and report builders (`REPORT_BUILDERS`). To wire a new recipe: flip it to `variant="ready"` with a `key` in `gui/recipes.py`, build a screen in `gui/screens/` (declaring its task and params), and add a `DEMOS` entry and a report builder in `app.py`.

Packaging: close the app, then run this from the repository root in PowerShell. Use an isolated `PATH` for the build so PyInstaller cannot collect incompatible DLLs from other tools. A Poppler `icuuc.dll` on the tooling `PATH` caused the QtCore "specified procedure could not be found" startup error; a clean build with only Python and Windows directories fixes that collision.

```powershell
# First-time setup only; this session's .venv already has these dependencies.
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e ".[dev,gui,llm]" pyinstaller

# Rebuild after source changes, with the app closed.
./build_windows.ps1
```

The script verifies build targets, isolates `PATH`, includes the supported causal-model implementations/fast tokenizers and runtime metadata, then restores the environment. The older `--exclude-module torch` command would omit the LLM runtime; use the script now. Model weights remain in external folders, outside the bundle.

The output is `dist/MLcookbook/MLcookbook.exe`; launch it once after each build and verify generation with a local model. **The exe is a frozen snapshot**: editing `.py` files never changes it. If your shortcut shows old behavior, rebuild — or run from source with `.venv/Scripts/python.exe -m mlcookbook`. The desktop/Start Menu shortcuts point at that exe, so keep the path stable. `dist/`, `build/`, `.venv/`, and `models/` are gitignored.

Work in small loops: write the test first (or alongside), run it red, make it green, run the full suite, commit.

## What's built (status)

| Recipe | Core | GUI report (built-in example until user data is imported) |
|---|---|---|
| Local LLM · PyTorch | `llm.py` | local prompt or text-column generation, decoded text, timing/tokens, model/runtime provenance; explicit model setup |
| Linear Regression | `regression/linear.py`, `regression/diagnostics.py` | actual vs predicted, residuals, CV, residual checks, coefficients; OLS/ridge/lasso (diabetes) |
| Logistic Regression | `classification/logistic.py` | sigmoid (binary) or recall by class (multiclass), metrics, confusion matrix, coefficients (breast cancer) |
| Decision Trees | `classification/tree.py`, `regression/tree.py` | drawn tree, depth curve, importances, rules; classification or regression (breast cancer) |
| Random Forests | `classification/forest.py`, `regression/forest.py` | ROC / recall by class / actual vs predicted, structure, importances (breast cancer) |
| K-Means | `clustering/kmeans.py` | clusters on PC1/PC2, elbow + silhouette scans, sizes, profiles (wine) |
| PCA | `dimensionality/pca.py` | scree plot, scores, loadings (wine) |
| Association Scan | `statistics/scan.py` (+ `association.py`, `categorical.py`, `groups.py`) | ranked pairs with adjusted p-values, detail for the strongest pair (breast cancer) |

Every supervised report also shows cross-validation (mean ± SD) and the fitted preprocessing table.

## Random forests milestone - 2026-09-27

Implemented at my request as the fourth ready recipe. `random_forest` / `RandomForestResult` are exported from `mlcookbook` and `mlcookbook.classification`; `classification/forest.py` reuses shared binary validation, training-only preprocessing (without numeric scaling), and stratified evaluation. No new dependencies.

Defaults are 100 bootstrap trees, sqrt feature sampling, no depth cap, at least 5 training samples per leaf, a 25% test split, and seed 0. Python callers can change these settings; the GUI retains the defaults and uses its existing CSV/Parquet and column-selection controls. The report starts with a held-out ROC curve, followed by train/test metrics, a forest-size summary, feature importances, confusion matrix, and warnings.

Learning points: a forest averages many randomized trees' class probabilities; the positive-class probability must be selected by its label, not assumed to be column 1. Importance values are for encoded columns and reflect impurity reduction, with a warning about high-cardinality bias and causal interpretation. ROC coordinates are computed in the core with sklearn. The GUI never fits models or computes ROC curves.

Tests in `tests/test_forest.py` cover predictive signal, leakage prevention, class ordering, reproducibility, settings/target validation, warnings, and frozen/JSON results. Shared file and GUI integration tests exercise forest runs from both CSV and Parquet, including nullable features. Regression forests and multiclass support remain future learning work.

Verification at completion: **110 pytest tests passed**; the public API demo completed; the built-in and imported-Parquet forest reports were visually inspected; the Windows executable was rebuilt, confirmed responsive on launch, and closed normally. No commit was made.

## CSV/Parquet milestone - 2026-09-27

Implemented at my request across linear regression, logistic regression, and decision trees:

- `src/mlcookbook/io.py`: shared `load_dataset` reader, explicit CSV delimiter/encoding, PyArrow Parquet loading, and validation for headers, malformed records, empty data, and nested values.
- `gui/screens/report_screen.py`: file picker, first-20-row preview, column-role controls and missing counts, target/features, positive class, and an enabled Run Analysis flow. Reports identify the dataset and selections used.
- `app.py`: imported-data session state, report builders reused for demo/custom data, and background loading/training with recoverable errors. Switching recipes after import does not train a demo model.
- `preprocessing/builders.py`: normalize nullable categorical/string/boolean missing markers inside the pipeline, retaining training-only imputation.
- PyArrow added to `pyproject.toml` and included in the Windows build. Packaging excludes unused Torch/TensorFlow/Keras packages from the global environment; revisit those exclusions if a future workflow actually needs them.

Verification at completion: **80 pytest tests passed**, all three imported-data reports and the preview were visually inspected on native Windows, and the rebuilt `dist/MLcookbook/MLcookbook.exe` was launch-tested, confirmed responsive, and closed normally. No commit was made.

Coverage lives in `tests/test_io.py` (formats, invalid files, all three workflows, nullable data), `tests/test_app_import.py` (controls, worker completion, recovery, navigation, and deferred tree callbacks), and `tests/test_preprocessing.py` (nullable imputation fitted on training rows only). GUI integration tests require the GUI extra and skip when PySide6 is unavailable.

Limits at the CSV/Parquet milestone: local flat tables that fit in RAM; `.csv` and `.parquet` extensions only; no saved sessions or result-export UI; selections reset on navigation; existing model parameters and the 25% test split remain fixed. The two classifiers are still binary-only. Statistical-role overrides do not repair or convert source values.

Future format and workflow ideas are recorded in [ROADMAP.md](ROADMAP.md#directions-under-discussion); they are proposals, not an implementation queue or a replacement for my curriculum.

## The codebase in one paragraph

`Dataset` (core/dataset.py) wraps a DataFrame and infers a `VarType` per column (core/schema.py) — that type, not the pandas dtype, decides which analyses apply. Every analysis is a plain function taking a `Dataset`, validating against the schema, computing with scipy/sklearn, and returning a frozen dataclass that mixes in `Result` (core/results.py) so `.to_dict()` is always JSON-safe. `preprocessing/build_preprocessor` returns an *unfitted* ColumnTransformer that gets fitted inside a Pipeline on training data only — that's the leakage guarantee. Two reference implementations exist: `statistics/association.py` (correlation) and `regression/linear.py` (with `classification/logistic.py` as the binary-classifier variant, sharing `classification/_binary.py`). Everything you build is a variation on one of those files.

## Rules I set for myself

- Analyses **return objects, never print**. Warnings go in `result.warnings`.
- Validate against `schema()`, don't guess from dtypes.
- Fit preprocessing on train only. If I ever call `.fit` before `train_test_split`, stop.
- scipy/sklearn do the math; I do orchestration, validation, and interpretation.
- Every new result type gets a `json.dumps(result.to_dict())` test.
- Keep `mlcookbook/__init__.py` small — the public API should fit in one glance.
- Files are UTF-8 (a UTF-16 README once broke the build on this machine).

## Suggested build order (my curriculum)

_Status 2026-10-12: items 1-5 below were built at my request; see the handoff above. Kept as the study order for understanding the code._

Roughly increasing difficulty; full list with hints in [ROADMAP.md](ROADMAP.md).

1. **Chi-square + Cramér's V** — first solo feature; pure copy of the correlation template. Learn: contingency tables, expected counts, effect size vs p-value.
2. ✅ (agent-built) **Logistic regression + classification metrics** — copy of the regression template plus a new `evaluation/` result. Learn: classification metrics, class imbalance, probability vs label predictions.
3. **One-way ANOVA + Kruskal–Wallis** — first analysis with an assumption-based fallback. Learn: when parametric assumptions matter, eta-squared.
4. **Pairwise association scan** — dispatch the right test per column pair from `VarType`. This is the payoff of the schema design and the first "meta" feature.
5. **Cross-validation utility** — structured per-fold results. Learn: variance of estimates, why a single split lies.

After those: regularized regression, regression forests (binary decision trees and random forests already exist), k-means, PCA, residual diagnostics, then the FastAPI layer.

## When I use AI agents

Agents read `AGENTS.md` / `CLAUDE.md` and know not to build roadmap features for me — unless I explicitly say "build it", in which case they go end to end (core, tests, GUI, docs, exe rebuild) and I review. Good uses: reviewing my implementation, explaining a statistical concept, writing tedious test cases after I've written the logic, debugging. Bad use: "implement ANOVA" — that defeats the purpose.

## Scratch space for ideas

- (jot things here)
