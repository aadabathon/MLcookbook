"""Tour of the current mlcookbook public API.

Run from the repo root after `pip install -e .`:

    python examples/basic_usage.py
"""

import json

import numpy as np
import pandas as pd

from mlcookbook import (
    Dataset,
    PreprocessingConfig,
    anova,
    association_scan,
    correlation,
    decision_tree,
    decision_tree_regression,
    kmeans,
    linear_regression,
    logistic_regression,
    pca,
    random_forest,
)

# --- Build a small synthetic dataset -----------------------------------------
rng = np.random.default_rng(7)
n = 300
sqm = rng.uniform(30, 200, n)
rooms = rng.integers(1, 6, n).astype(float)
city = rng.choice(["Oslo", "Bergen", "Stavanger"], n)
city_premium = pd.Series(city).map({"Oslo": 40_000, "Bergen": 10_000, "Stavanger": 0})
price = 55_000 * sqm + 120_000 * rooms + city_premium * sqm / 100 + rng.normal(0, 150_000, n)

df = pd.DataFrame(
    {
        "sqm": sqm,
        "rooms": rooms,
        "city": city,
        "listed": pd.date_range("2025-01-01", periods=n, freq="6h"),
        "price": price,
    }
)
df.loc[df.sample(15, random_state=7).index, "sqm"] = np.nan

# --- 1. Wrap it in a Dataset: schema inference + metadata --------------------
data = Dataset(df)

print("=== Schema ===")
for col in data.schema().columns:
    print(f"  {col.name:8s} {col.var_type.value:12s} (dtype={col.dtype})")

print("\n=== Describe ===")
print(json.dumps(data.describe().to_dict(), indent=2))

# --- 2. A statistical association --------------------------------------------
corr = correlation(data, "sqm", "price", method="pearson")
print("\n=== Correlation(sqm, price) ===")
print(json.dumps(corr.to_dict(), indent=2))

# --- 3. A supervised workflow ------------------------------------------------
result = linear_regression(data, target="price")
print("\n=== Linear regression: price ~ features ===")
print(f"  features:     {result.features}")
print(f"  test R^2:     {result.test_metrics.r2:.4f}")
print(f"  test RMSE:    {result.test_metrics.rmse:,.0f}")
print(f"  coefficients: {json.dumps({k: round(v) for k, v in result.coefficients.items()})}")
print(f"  warnings:     {result.warnings}")

# --- 4. Classifiers (binary here; 3+ classes work the same way) ---------------------------------------------------
df["expensive"] = df["price"] > df["price"].median()
clf_data = Dataset(df.drop(columns=["price"]))

logit = logistic_regression(clf_data, target="expensive")
print("\n=== Logistic regression: expensive ~ features ===")
print(f"  test accuracy: {logit.test_metrics.accuracy:.3f}   ROC AUC: {logit.test_metrics.roc_auc:.3f}")
print(f"  odds ratios:   {json.dumps({k: round(v, 2) for k, v in logit.odds_ratios.items()})}")

tree = decision_tree(clf_data, target="expensive", max_depth=3)
print("\n=== Decision tree (depth 3): expensive ~ features ===")
print(f"  test accuracy: {tree.test_metrics.accuracy:.3f}   leaves: {tree.n_leaves}")
for rule in tree.rules:
    print(f"  IF {' AND '.join(rule.conditions)} THEN {rule.prediction}  (n={rule.n_samples})")

forest = random_forest(clf_data, target="expensive", n_estimators=100)
print("\n=== Random forest: expensive ~ features ===")
print(f"  test accuracy: {forest.test_metrics.accuracy:.3f}   ROC AUC: {forest.test_metrics.roc_auc:.3f}")
print(f"  trees: {forest.n_estimators}   depth range: {min(forest.tree_depths)}-{max(forest.tree_depths)}")
print(f"  importances: {json.dumps({k: round(v, 3) for k, v in forest.feature_importances.items()})}")
print(f"  warnings: {forest.warnings}")

# --- 5. Preprocessing choices and cross-validation ---------------------------
config = PreprocessingConfig(scaling="robust", missing_indicators=True)
ridge = linear_regression(data, target="price", penalty="ridge", alpha=1.0, preprocessing=config)
cv = ridge.cross_validation
print("\n=== Ridge regression, robust scaling, missing-value flags ===")
print(f"  CV R^2: {cv.mean['r2']:.3f} +/- {cv.std['r2']:.3f} over {cv.n_folds} folds of the training rows")
for column in ridge.preprocessing.columns:
    print(f"  {column.column}: {' -> '.join(column.steps)}")

regression_tree = decision_tree_regression(data, target="price", max_depth=2)
print(f"\n=== Regression tree (depth 2): test R^2 {regression_tree.test_metrics.r2:.3f} ===")
for rule in regression_tree.rules:
    print(f"  IF {' AND '.join(rule.conditions)} THEN {rule.prediction:,.0f}")

# --- 6. Statistics and unsupervised learning ---------------------------------
groups = anova(data, "price", "city")
print(f"\n=== ANOVA price ~ city: eta^2 {groups.eta_squared:.3f}, p {groups.p_value:.3g}, "
      f"trust {groups.recommended_test} ===")
scan = association_scan(data)
print("=== Strongest associations ===")
for pair in scan.pairs[:3]:
    print(f"  {pair.x} ~ {pair.y}: {pair.test}, {pair.effect_measure} {pair.effect_size:.2f}")

clusters = kmeans(data, features=["sqm", "rooms", "price"], n_clusters=3)
print(f"\n=== K-means (k=3): silhouette {clusters.silhouette:.2f}, best k by silhouette "
      f"{clusters.best_k_by_silhouette} ===")
components = pca(data, features=["sqm", "rooms", "price"])
print(f"=== PCA: PC1 explains {components.components[0].explained_variance_ratio:.0%} of the variance ===")

# Every result serializes to JSON — this is what a future FastAPI layer returns.
for r in (result, logit, tree, forest, ridge, regression_tree, groups, scan, clusters, components):
    json.dumps(r.to_dict())
print("\nAll results are JSON-serializable. Done.")
