# Roadmap

Features intentionally left for hands-on implementation, roughly in suggested order. Each should follow the existing patterns: schema-validated inputs, frozen result dataclasses with warnings, leakage-safe pipelines, sklearn/scipy for the math.

## Statistics

- [ ] **Chi-square test of independence + Cramér's V** for categorical×categorical pairs (`statistics/association.py`). Warn on low expected cell counts.
- [ ] **One-way ANOVA** (and Kruskal–Wallis as the nonparametric fallback) for numeric×categorical. Report effect size (eta squared).
- [ ] **Correlation matrix / pairwise association scan**: given a `Dataset`, pick the right test per column pair based on `VarType` — this is the payoff of the schema design.
- [ ] **Confidence intervals** on correlation coefficients (Fisher z-transform).
- [ ] **Normality/assumption checks** (Shapiro–Wilk, skewness) surfaced as warnings on Pearson.
- [ ] **Mutual information** for mixed-type pairs (`sklearn.feature_selection.mutual_info_*`).

## Machine learning

- [ ] **Logistic regression classification workflow** (`classification/`) mirroring `regression/linear.py`, with `classification_metrics` (accuracy, precision/recall/F1, ROC AUC, confusion matrix) in `evaluation/`.
- [ ] **Cross-validation** utility (`evaluation/`) wrapping `sklearn.model_selection.cross_validate` and returning a structured result (per-fold + aggregate metrics).
- [ ] **Regularized regression** (Ridge/Lasso) — mostly a parameter to the existing workflow; a good exercise in extending without duplicating.
- [ ] **Tree-based models** (RandomForest for regression and classification) with feature importances in the result.
- [ ] **K-means clustering workflow** (`clustering/`) with silhouette score and an elbow/inertia scan.
- [ ] **PCA workflow** (`dimensionality/`) returning explained variance ratios and loadings.
- [ ] **Residual diagnostics** for regression: residual normality, heteroscedasticity flags, added to `LinearRegressionResult.warnings` or a dedicated diagnostics result.

## Infrastructure

- [ ] **Visualization module**: functions that take result objects (never raw data + recomputation) and return matplotlib figures.
- [ ] **`Dataset.dropna()` / row-filtering helpers** returning new `Dataset` instances.
- [ ] **FastAPI service** in a separate `api/` or sibling package: endpoints that wrap analyses and return `result.to_dict()`.
- [ ] **CI** (GitHub Actions: pytest on 3.12+), plus `ruff` and `mypy` configs.
- [ ] **Property-based tests** with `hypothesis` for schema inference edge cases.
