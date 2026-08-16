"""Build sklearn preprocessors from a Dataset's schema.

The returned ``ColumnTransformer`` is unfitted: it must be fitted inside a
``Pipeline`` on training data only, which is what keeps preprocessing free of
train/test leakage.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.schema import VarType


def build_preprocessor(dataset: Dataset, columns: list[str]) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer covering *columns*.

    Numeric columns: median imputation + standard scaling.
    Categorical/boolean columns: mode imputation + one-hot encoding
    (unknown categories at predict time are ignored, not errors).
    """
    schema = dataset.schema()
    numeric = [c for c in columns if schema.var_type(c) is VarType.NUMERIC]
    categorical = [
        c for c in columns if schema.var_type(c) in (VarType.CATEGORICAL, VarType.BOOLEAN)
    ]
    unsupported = sorted(set(columns) - set(numeric) - set(categorical))
    if unsupported:
        raise ValueError(
            f"Columns not supported by build_preprocessor (datetime/text/unsupported): {unsupported}"
        )

    transformers = []
    if numeric:
        numeric_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    if not transformers:
        raise ValueError("No usable columns to build a preprocessor from.")
    return ColumnTransformer(transformers, verbose_feature_names_out=False)
