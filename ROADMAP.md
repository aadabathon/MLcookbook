# Roadmap

Originally features left for hands-on implementation; most were built at Adam's request on 2026-10-12 (marked [x]). Each should follow the existing patterns: schema-validated inputs, frozen result dataclasses with warnings, leakage-safe pipelines, sklearn/scipy for the math.

## Statistics

- [x] **Chi-square test of independence + Cramér's V** (`statistics/categorical.py`), with a low-expected-count warning and Fisher's exact test for 2x2 tables.
- [x] **One-way ANOVA + Kruskal–Wallis** (`statistics/groups.py`), with eta squared, Levene and per-group Shapiro-Wilk checks, and a recommended test.
- [x] **Pairwise association scan** (`statistics/scan.py`): Spearman / chi-square / Kruskal-Wallis chosen by `VarType`, effect sizes, Benjamini-Hochberg adjusted p-values.
- [x] **Confidence intervals** on correlation coefficients (Fisher z-transform).
- [x] **Normality checks** surfaced as warnings on Pearson.
- [ ] **Mutual information** for mixed-type pairs (`sklearn.feature_selection.mutual_info_*`).

## Machine learning

- [x] **Logistic regression classification workflow** with `classification_metrics`.
- [x] **Cross-validation** utility (`evaluation/cross_validation.py`); every supervised workflow cross-validates its training rows.
- [x] **Regularized regression** (ridge/lasso) as `penalty=`/`alpha=` on `linear_regression`.
- [x] **Random forest classification** (`classification/forest.py`).
- [x] **Tree-based regression**: `decision_tree_regression`, `random_forest_regression`.
- [x] **Multiclass classification** for all three classifiers (macro metrics, per-class metrics, multinomial logistic).
- [x] **K-means clustering workflow** (`clustering/kmeans.py`) with silhouette and an elbow/inertia scan.
- [x] **PCA workflow** (`dimensionality/pca.py`) with explained variance and loadings.
- [x] **Residual diagnostics** for regression (`regression/diagnostics.py`).
- [ ] **Hyperparameter search** (e.g. choose alpha or max_depth by cross-validation, reported as a curve).
- [x] **Fair model comparison, first scope**: ridge and random forest regression on the same materialized split and training folds, one CV comparison table/curve; select by mean CV RMSE and evaluate only the selected model on the test rows. Exact row/fold membership is retained. Built at Adam's request on 2026-10-17; other algorithms and classification comparison remain future extensions.
- [x] **SVM and neural-network recipes**: `svm`/`svm_regression` (kernels, support vectors, standardized SVR target) and `neural_network`/`neural_network_regression` (scikit-learn MLP with loss and early-stopping validation curves).

## Infrastructure

- [x] **Local CSV/Parquet input** for every ready recipe: preview, schema-role overrides, target/features/positive-class selection, background loading/training.
- [x] **Configurable, recorded preprocessing** (`PreprocessingConfig`, `PreprocessingSummary`): imputation, scaling, encoding, rare-level pooling, missing indicators, datetime features.
- [x] **Import quality**: leading zeros kept, extra missing markers, ISO-8601 date detection, ID-like/constant/mostly-missing column flags.
- [ ] **Visualization module**: functions that take result objects (never raw data + recomputation) and return figures (needs a plotting dependency decision).
- [ ] **`Dataset.dropna()` / row-filtering helpers** returning new `Dataset` instances.
- [ ] **FastAPI service** in a separate `api/` or sibling package: endpoints that wrap analyses and return `result.to_dict()`.
- [x] **CI** (GitHub Actions): ruff, library tests on Ubuntu 3.12/3.13, full app suite on Windows.
- [ ] **mypy** configuration and type-checking in CI.
- [ ] **Property-based tests** with `hypothesis` for schema inference edge cases.
- [x] **Saved experiments**: explicit `.mlcb` save/open for every recipe, with a Parquet dataset snapshot, import options, overrides, current settings, last report/request, run seed/time, versions, and a snapshot fingerprint. Reopening restores without training, including unrun edits. Built at Adam's request on 2026-10-17.
- [ ] **Result export**: standalone metrics/predictions files and fitted-model persistence. Comparison's complete result is already retained in experiment JSON; the other recipes currently save their display reports.

## Directions under discussion

Recorded after the CSV/Parquet milestone on 2026-09-27. These are proposals for Adam to consider, not scheduled work or authorization to implement anything. The learning curriculum above remains intact.

### Candidate data inputs

Relative effort includes the import UI, validation, tests, and packaging, not just calling a reader.

| Candidate | What it would enable | Relative effort and decisions |
|---|---|---|
| TSV (`.tsv`) | Tab-delimited research and application exports | Small: tab parsing already exists for `.csv`; extend extension dispatch/file filters and tests, with tab as the TSV default. |
| Compressed CSV (`.csv.gz`) | Read compressed exports without manual extraction | Small to moderate: make both header validation and data parsing compression-aware. This does not make training out-of-core. |
| Excel (`.xlsx`) | Analyze worksheets directly | Moderate: choose a sheet and header row; define how to handle blank regions and formula cells. An Excel engine such as `openpyxl` would need separate dependency approval. |
| JSON Lines (`.jsonl`/`.ndjson`), then record-array JSON | Analyze event records and application/API exports | Moderate: start with flat records; define missing-field and mixed-type behavior. Nested data needs an explicit flattening choice. |
| SQLite (`.sqlite`/`.db`) | Analyze local database tables without exporting CSV first | Moderate: start with read-only table selection and preview; joins, custom SQL, and remote databases can come later. |
| Feather (`.feather`) | Exchange typed tables with Python/R workflows | Small to moderate: the existing Arrow dependency helps; prioritize when there is a concrete dataset needing it. |

Proposed order: TSV/compressed CSV as convenience additions, Excel as the next major import capability, then JSONL. SQLite and Feather can follow actual dataset needs. Keep each supported input feeding the existing `Dataset` and schema checks so every algorithm benefits.

Technical references: pandas documents [Excel engines and sheet selection](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html), [JSON and line-oriented records](https://pandas.pydata.org/docs/reference/api/pandas.read_json.html), [SQL query input including sqlite3 connections](https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html), and [Feather input](https://pandas.pydata.org/docs/reference/api/pandas.read_feather.html).

### Candidate workflow improvements

A possible product direction is a local analysis workbench where users can inspect data, run suitable methods, understand the results, and reproduce the run.

1. **Import review and data quality:** preserve leading-zero identifiers through parser-level type choices; configurable missing-value markers; explicit numeric/date conversion; flag constant/all-missing columns and likely IDs. Keep cleaning choices visible.
2. **Reproducible projects and useful outputs:** save import options, schema overrides, selected columns, algorithm settings, seed, package versions, and a source-data fingerprint; export metrics and per-row predictions. Explain whether reopening references or includes the original data.
3. **Fair model comparison:** reuse the same holdout rows across compatible algorithms, introduce training-only cross-validation and appropriate metrics, and distinguish classification from regression comparisons. Grouped or time-based splits would be separate explicit choices when rows are not independent.
4. **Broader analysis:** connect the existing correlation core to the GUI, then consider the statistical tests, regularization, regression forests, multiclass classification, PCA, and clustering as individually chosen learning projects. Methods should explain their input requirements and interpretation.
5. **Larger datasets:** measure actual memory/runtime limits before adding sampling, column projection, or batch processing. A reader that streams does not automatically make the estimator train incrementally.

Saved configurations and reports deserve consideration alongside new formats: importing more files is most useful when users can retain and compare the analyses they run.

### Later model-inference workflows

GGUF stores model tensors and metadata for inference ([Hugging Face GGUF documentation](https://huggingface.co/docs/hub/gguf)). It belongs to a future model-loading workflow alongside a dataset input, with its own runtime and resource requirements.

Possible first scope: select a compatible local model, inspect its metadata, run a bounded text task on selected rows, and retain outputs with the model and prompt settings used. Examples include labeling text or extracting structured fields for review before downstream analysis. Labeling, generation, embeddings, and fine-tuning are distinct capabilities; support should be chosen explicitly rather than implied by accepting a GGUF file.

A model file alone does not supply a labeled evaluation dataset. Existing statistical/ML results should remain computed by their analytical core; any future generated explanation or label should be identifiable as such. Runtime selection, hardware support, model acquisition, licensing, and dependency approvals need their own design discussion before implementation.
