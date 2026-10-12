"""Residual diagnostics for regression fits.

Two classic questions about the errors of a least-squares fit, answered with
scipy tests on the *training* residuals (the ones the fit was chosen to make
small):

* Are they roughly normal? Shapiro-Wilk (up to 5000 rows) or D'Agostino-
  Pearson. Least squares stays unbiased either way, but classical p-values
  and intervals assume normal errors.
* Is their spread constant (homoscedastic)? Spearman correlation between the
  absolute residual and the fitted value: a positive, significant correlation
  means errors grow with the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from mlcookbook.core.results import Result

ALPHA = 0.05
SHAPIRO_MAX_ROWS = 5000


@dataclass(frozen=True)
class ResidualDiagnostics(Result):
    n_obs: int
    mean: float
    std: float
    skewness: float
    excess_kurtosis: float  # 0 for a normal distribution
    normality_test: str  # "shapiro", "dagostino", or "none" (too few rows)
    normality_p_value: float  # NaN (-> null) when not tested
    spread_correlation: float  # Spearman rho of |residual| vs fitted value
    spread_p_value: float


def residual_diagnostics(y_true, y_fitted, warnings: list[str]) -> ResidualDiagnostics:
    """Diagnose residuals ``y_true - y_fitted``; append plain-language warnings to *warnings*."""
    y_true = np.asarray(y_true, dtype=float)
    y_fitted = np.asarray(y_fitted, dtype=float)
    residuals = y_true - y_fitted
    n = len(residuals)
    nan = float("nan")
    varied = n >= 3 and np.ptp(residuals) > 0
    if not varied:
        test, normality_p = "none", nan
    elif n <= SHAPIRO_MAX_ROWS:
        test, normality_p = "shapiro", float(stats.shapiro(residuals).pvalue)
    else:
        test, normality_p = "dagostino", float(stats.normaltest(residuals).pvalue)
    skewness = float(stats.skew(residuals)) if varied else nan
    kurtosis = float(stats.kurtosis(residuals)) if varied else nan

    if varied and np.ptp(y_fitted) > 0:
        spread = stats.spearmanr(np.abs(residuals), y_fitted)
        rho, spread_p = float(spread.statistic), float(spread.pvalue)
    else:
        rho, spread_p = nan, nan

    if normality_p < ALPHA:
        name = "Shapiro-Wilk" if test == "shapiro" else "D'Agostino-Pearson"
        warnings.append(
            f"Training residuals depart from normality ({name} p={normality_p:.3g}, skew {skewness:.2f}). "
            "The fit is still unbiased, but classical p-values and intervals assume normal errors; "
            "with many rows this test flags even small departures."
        )
    if spread_p < ALPHA:
        direction = "grows" if rho > 0 else "shrinks"
        warnings.append(
            f"Residual spread {direction} with the fitted value (Spearman rho={rho:.2f}, p={spread_p:.3g}): "
            "the errors are heteroscedastic. A log transform of a positive target often helps."
        )
    return ResidualDiagnostics(
        n_obs=n,
        mean=float(residuals.mean()) if n else nan,
        std=float(residuals.std(ddof=1)) if n > 1 else nan,
        skewness=skewness,
        excess_kurtosis=kurtosis,
        normality_test=test,
        normality_p_value=normality_p,
        spread_correlation=rho,
        spread_p_value=spread_p,
    )
