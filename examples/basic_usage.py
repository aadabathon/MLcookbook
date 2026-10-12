"""Tour of the current mlcookbook public API.

Run from the repo root after `pip install -e .`:

    python examples/basic_usage.py
"""

import json

import numpy as np
import pandas as pd

from mlcookbook import Dataset, correlation, decision_tree, linear_regression, logistic_regression, random_forest

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

# --- 4. Binary classifiers ---------------------------------------------------
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

# Every result serializes to JSON — this is what a future FastAPI layer returns.
for r in (result, logit, tree, forest):
    json.dumps(r.to_dict())
print("\nAll results are JSON-serializable. Done.")
