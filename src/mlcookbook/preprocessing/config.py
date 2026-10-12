"""User-facing preprocessing choices.

A ``PreprocessingConfig`` says *how* each column type is prepared; it never
holds fitted values. Fitting happens inside a Pipeline on training rows only,
and ``describe_preprocessor`` reports what was actually learned.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlcookbook.core.results import Result

NUMERIC_IMPUTE = ("median", "mean")
SCALING = ("standard", "robust", "minmax", "none")
CATEGORICAL_IMPUTE = ("most_frequent", "missing_category")
CATEGORICAL_ENCODING = ("onehot", "ordinal")


@dataclass(frozen=True)
class PreprocessingConfig(Result):
    """How to impute, scale, and encode features.

    * ``numeric_impute``: fill missing numeric (and date-derived) values with the
      training ``"median"`` or ``"mean"``.
    * ``scaling``: ``"standard"`` (z-score), ``"robust"`` (median/IQR, resists
      outliers), ``"minmax"`` (to [0, 1]) or ``"none"``. ``None`` means the
      workflow's default: standard for linear models, k-means and PCA; none for trees.
    * ``categorical_impute``: ``"most_frequent"`` fills with the training mode;
      ``"missing_category"`` keeps missingness visible as its own level.
    * ``categorical_encoding``: ``"onehot"`` (one column per level) or
      ``"ordinal"`` (one integer column; fine for trees, imposes an arbitrary
      order on linear models).
    * ``min_category_frequency``: levels rarer than this count (int) or share
      (float in (0, 1)) of training rows are pooled into one "infrequent" level.
    * ``missing_indicators``: add a 0/1 "was missing" column for each numeric
      feature that had missing values in training.
    """

    numeric_impute: str = "median"
    scaling: str | None = None
    categorical_impute: str = "most_frequent"
    categorical_encoding: str = "onehot"
    min_category_frequency: int | float | None = None
    missing_indicators: bool = False

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("numeric_impute", self.numeric_impute, NUMERIC_IMPUTE),
            ("categorical_impute", self.categorical_impute, CATEGORICAL_IMPUTE),
            ("categorical_encoding", self.categorical_encoding, CATEGORICAL_ENCODING),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {list(allowed)}; got {value!r}.")
        if self.scaling is not None and self.scaling not in SCALING:
            raise ValueError(f"scaling must be one of {list(SCALING)} or None; got {self.scaling!r}.")
        freq = self.min_category_frequency
        if freq is not None:
            if isinstance(freq, bool) or not isinstance(freq, (int, float)):
                raise ValueError(f"min_category_frequency must be a count or a share; got {freq!r}.")
            if isinstance(freq, float) and not 0 < freq < 1:
                raise ValueError(f"A min_category_frequency share must be in (0, 1); got {freq}.")
            if isinstance(freq, int) and freq < 1:
                raise ValueError(f"A min_category_frequency count must be at least 1; got {freq}.")
