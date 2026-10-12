# Agent Guidelines — ML Cookbook

This repo is a **learning environment owned by Adam, the author**. Agents assist; they do not take over. The default posture is: small, surgical changes that preserve the existing patterns, and no unsolicited feature expansion.

## Prime directive

**Do not implement ROADMAP.md items unless explicitly asked.** The roadmap is Adam's personal curriculum — building those features is the point of the project. If asked to "help" with one, prefer explaining, reviewing, or pairing over writing the whole thing.

When Adam *does* explicitly say to build something ("implement it", "go ahead"), that is the green light: build it end to end (core + tests + GUI + docs) rather than handing back pieces. Adam reviews the work afterward, so leave the repo green and the docs current. Logistic regression, decision trees, and random forests were built this way.

## Project shape

- `src/` layout, package `mlcookbook`, Python 3.12+, hatchling build.
- Install: `pip install -e ".[dev]"` · Test: `pytest` · Demo: `python examples/basic_usage.py` · App: `python -m mlcookbook`
- **Which Python:** this machine has 3.11, 3.12 and 3.13; bare `python` is 3.11 and does NOT have the project installed. Use `py -3.12 -m pytest` / `py -3.12 -m mlcookbook` (or activate a 3.12 venv). If you see "No module named mlcookbook/pytest", it's the wrong interpreter.
- Desktop app architecture: `gui/` is dumb presentation (signals out, data in; never imports the core), `src/mlcookbook` is the UI-free core, and repo-root `app.py` is the ONLY module that may import both. Keep it that way. Navigation is delete-and-redraw via `MainWindow.show_screen`; recipes route through `SCREEN_FACTORIES` in `app.py`. There is exactly one entry point (`python -m mlcookbook`); do not add others.
- Dependencies: pandas, numpy, scipy, scikit-learn, pyarrow, pytest. Do not add dependencies (statsmodels, matplotlib, etc.) without asking.
- Built so far: statistics (`correlation`, `chi_square`, `anova`, `association_scan`), regression (`linear_regression` with ridge/lasso and residual diagnostics, `decision_tree_regression`, `random_forest_regression`), classification (`logistic_regression`, `decision_tree`, `random_forest`, binary or multiclass), `kmeans`, and `pca`, all on a configurable, recorded preprocessing pipeline with training-only cross-validation. `visualization/` is intentionally skeletal (the app draws from result data); leave it that way unless told otherwise.
- The user launches the app from a Windows shortcut pointing at the PyInstaller build `dist/MLcookbook/MLcookbook.exe`, which is a **frozen snapshot**: source edits do not appear there until it is rebuilt (command in ADAM.md). After GUI or core changes the user will want to see, rebuild it and check the exe launches. `dist/` and `build/` are gitignored.

## Architecture rules (non-negotiable)

1. **Types drive applicability.** Columns get a `VarType` (`core/schema.py`) — a statistical role, not a pandas dtype. Analyses validate inputs against `Dataset.schema()` and raise or warn; never silently coerce.
2. **Results are data.** Every analysis returns a frozen dataclass mixing in `Result` (`core/results.py`) with a `warnings: list[str]` field. No analysis prints, plots, or logs. `result.to_dict()` must survive `json.dumps`.
3. **Warnings are part of the result.** Dropped rows, skipped columns, constant variables, assumption issues → `result.warnings`.
4. **No leakage.** Preprocessors come from `preprocessing.build_preprocessor()` unfitted and are fitted inside a sklearn `Pipeline` on training data only. Never fit anything on the full dataset before splitting.
5. **sklearn/scipy do the math.** Orchestrate and validate; never reimplement numerical algorithms.
6. **Small public API.** Only add to `mlcookbook/__init__.py` deliberately. Everything else is imported from submodules.
7. **No enterprise patterns.** No factories, DI containers, deep inheritance, or clever metaprogramming. Type hints, dataclasses, composition, small modules.

## Desktop GUI conventions

- **Recipes are data + a factory.** A `Recipe` in `gui/recipes.py` (`variant="ready"`, a `key`), a thin screen class in `gui/screens/`, a demo entry in `DEMOS` and a builder in `REPORT_BUILDERS` in `app.py` (`SCREEN_FACTORIES` is derived from `DEMOS`). Ready screens (linear/logistic regression, decision tree, random forest, k-means, PCA, association scan) are ~10-line subclasses of `gui/screens/report_screen.py::ReportScreen` that declare their `task` (`"regression"`, `"classification"`, `"unsupervised"`) or `tasks` (tree/forest switch between classification and regression), and `params` specs (`int`/`float`/`choice` widgets whose keys app.py maps onto core arguments). `screen.request()` returns task, target, features, overrides, positive class (None = multiclass), params, preprocessing dict, and cv_folds.
- **Screens render a plain `report` dict** (documented at the top of `report_screen.py`): `dataset/target/features/settings` strings, optional `tree`, `plots`, `metric_groups`, `bars`, `tables`, `warnings`. All numbers arrive pre-formatted as strings (or raw point lists for plots). **`app.py` does all result-to-report formatting**; `gui/` never imports `mlcookbook`.
- **Without an imported dataset, every ready screen opens by running its report builder on a built-in demo** (`DEMOS` in `app.py`: diabetes, breast cancer, wine; scikit-learn's bundled data, so no new dependency) using the screen's own default `request()`. All screens support CSV/Parquet import with extra missing markers, a preview, data checks (import notes + quality issues; likely IDs/constant columns start unchecked), column-role overrides, target/feature selection, model settings, and the Preprocessing & validation dialog (`gui/preprocessing_dialog.py`). `mlcookbook.io.load_dataset_with_report` reads files; `app.py` owns the imported dataset and background jobs. Screens only emit load/run requests and render plain data. Imported data persists across recipes for the session; selections reset on navigation. Session state is not saved to disk. On imported-data navigation, screens wait for Run Analysis instead of training a demo. Thread completion and errors return to the GUI; the application waits for its active job when exiting.
- **Report order:** the recipe's main visual goes first (tree diagram, actual-vs-predicted scatter, fitted sigmoid, ROC curve, cluster projection, scree plot, strongest-pair detail), then plots, metric tiles (test, train, cross-validation mean ± SD), bars, tables (the preprocessing table last), warnings. Shared formatters live in `app.py` (`_cv_group`, `_preprocessing_table`, `_confusion_table`, ...). Metric captions are upper-cased by the stylesheet, so avoid Greek letters that look Latin in capitals (write RHO, ETA²).
- **Visuals are drawn with `QPainter`** (`gui/plot.py`: `PlotWidget`, `BarList`; `gui/tree_plot.py`: `TreeWidget`). Do not add matplotlib/plotly. Colors come from `gui/theme.py` (`ACCENT` violet, `PLOT_ALT` teal, `CATEGORICAL` palette addressed as `"c0"`..`"c9"` via `theme.palette`). To plot something, the **core result must carry the data** (e.g. `test_actual`/`test_predicted`, `test_probability`, tree `nodes`, k-means `projection`, PCA `scores`) — never recompute a model in `app.py`.
- **Verifying visually without a display:** construct `MLCookbookApp`, call `window.show_screen(SCREEN_FACTORIES[key]())`, `processEvents()`, then `window.grab().save("x.png")` and look at the PNG. Use the native Windows platform, not `QT_QPA_PLATFORM=offscreen` (offscreen has no fonts and renders tofu boxes). Flush `DeferredDelete` events between screens or the previous screen bleeds through.

## Templates to copy

- New statistical test → follow `statistics/association.py` (validate types → compute with scipy → frozen result dataclass with warnings).
- New supervised workflow → follow `regression/linear.py` (validate via `regression/_problem.py` → `resolve_config` + `build_preprocessor(..., config=config)` in a `Pipeline` → train/test split → metrics via `evaluation/` → `training_cross_validation` on the training rows → `describe_preprocessor` → result dataclass with `preprocessing` and `cross_validation` fields).
- New classifier → follow `classification/logistic.py`; validation (target types, 2..50 classes, `positive_class` for two classes only, numeric targets with >10 values rejected as continuous, missing/imbalance warnings) is shared in `classification/_problem.py::prepare_classification_problem`, with `class_probabilities`/`metric_scores`/`task_for` helpers. Split is stratified. Metrics via `evaluation.classification_metrics` (binary labels ordered `[negative, positive]`; 3+ labels give macro averages and one-vs-rest AUC; ROC AUC is NaN, serialized as `null`, when undefined).
- Tree-style models: `resolve_config(preprocessing, scale=False)` so thresholds stay in real units (linear/logistic/k-means/PCA default to standard scaling, so linear coefficients are per-SD). `classification/tree.py` and `regression/tree.py` expose the fitted model as plain data (flat `nodes`, `rules`) with a CV-on-train-only depth curve. Forests carry encoded-feature importances and per-tree sizes; the classifier also carries held-out ROC coordinates (binary). Unsupervised workflows fit preprocessing on all rows (no held-out set exists).

## Testing

- Every behavior change needs pytest coverage; shared fixtures live in `tests/conftest.py` (`mixed_df`, `linear_df`, `logistic_df`).
- Every new result type needs a `json.dumps(result.to_dict())` round-trip test.
- Coverage map: `test_io.py` (formats, leading zeros, markers, dates), `test_pipeline.py` (config, summaries, quality, CV, leakage per fold), `test_multiclass.py`, `test_regression_models.py` (ridge/lasso, diagnostics, tree/forest regression), `test_unsupervised.py` (k-means, PCA), `test_statistics.py`, `test_app_import.py` (desktop import/run/recovery/navigation, every recipe's demo, tasks, params, preprocessing; requires PySide6), `test_preprocessing.py` (nullable training-only imputation).
- Forest coverage: `tests/test_forest.py` checks leakage (including inside every CV fold), ROC/probability/class ordering, serialization, reproducibility, warnings, and input validation.
- Run `pytest` before declaring anything done.

## Definition of done for a new recipe

1. Core function + frozen result + `json.dumps` test (reuse shared helpers). 2. Exported deliberately from `mlcookbook/__init__.py`. 3. `pytest` green. 4. GUI recipe flipped to Ready with a screen and factory; render it and look at it. 5. README/AGENTS/ADAM updated. 6. Rebuild the exe if the user launches from the shortcut. 7. Don't commit unless asked.

## Windows gotcha

This machine is Windows. **Files must be UTF-8** — the original README was UTF-16 and broke the hatchling build. If a build fails with `UnicodeDecodeError`, check file encodings first.

## Docs to keep in sync

If you change the public API, architecture, or add a module: update `README.md`. If you complete or obsolete a roadmap item (only when asked): update `ROADMAP.md`.
