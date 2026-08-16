"""Model evaluation metrics as structured results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn import metrics as sk_metrics

from mlcookbook.core.results import Result


@dataclass(frozen=True)
class RegressionMetrics(Result):
    r2: float
    rmse: float
    mae: float
    n_obs: int


def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    """Compute standard regression metrics for one set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return RegressionMetrics(
        r2=float(sk_metrics.r2_score(y_true, y_pred)),
        rmse=float(sk_metrics.root_mean_squared_error(y_true, y_pred)),
        mae=float(sk_metrics.mean_absolute_error(y_true, y_pred)),
        n_obs=len(y_true),
    )
