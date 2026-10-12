"""Preprocessing built on sklearn Pipeline/ColumnTransformer (leakage-safe).

``build_preprocessor`` returns an unfitted ColumnTransformer configured by a
``PreprocessingConfig``; ``describe_preprocessor`` reports what a fitted one
learned; ``resolve_features`` is the shared feature selection/validation.
"""

from mlcookbook.preprocessing.builders import build_preprocessor
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.features import FEATURE_TYPES, resolve_features
from mlcookbook.preprocessing.summary import ColumnPreprocessing, PreprocessingSummary, describe_preprocessor

__all__ = [
    "FEATURE_TYPES",
    "ColumnPreprocessing",
    "PreprocessingConfig",
    "PreprocessingSummary",
    "build_preprocessor",
    "describe_preprocessor",
    "resolve_features",
]
