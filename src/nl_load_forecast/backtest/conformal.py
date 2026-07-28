"""Conformalized Quantile Regression (CQR) for calibrated prediction intervals.

Why this exists
---------------
Plain quantile regression — one LightGBM per quantile with the pinball objective — is *not*
guaranteed to be calibrated. In this project the nominal 80% interval (q10..q90) empirically
covered only ~49% of actuals: the per-quantile models are underdispersed, and the NL load
*level* drifts between the training window and the forecast day, so intervals fit on history
are systematically too narrow for the future.

CQR (Romano, Patterson & Candès, "Conformalized Quantile Regression", NeurIPS 2019) is a
distribution-free wrapper that fixes this without retraining. On a held-out *calibration* set
it measures how far actuals fell outside the model's predicted interval, then shifts the
bounds outward (or inward) by a single margin `Q`. On exchangeable data this yields a
finite-sample coverage guarantee of at least the nominal level; on our mildly non-stationary
series it moves empirical coverage close to nominal while keeping the interval as sharp as the
data allow.

This module is deliberately model-agnostic and side-effect free so it can be unit-tested on
synthetic data (see tests/test_conformal.py).
"""
from __future__ import annotations

import numpy as np


def conformity_scores(y_true, lower, upper) -> np.ndarray:
    """CQR nonconformity score  E_i = max(lower_i - y_i, y_i - upper_i).

    Positive when the actual fell *outside* the predicted interval [lower, upper]
    (below the lower bound or above the upper bound); negative when comfortably inside.
    """
    y = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return np.maximum(lower - y, y - upper)


def conformal_margin(scores: np.ndarray, alpha: float) -> float:
    """Interval-width adjustment ``Q`` from calibration scores.

    ``Q`` is the ``(1 - alpha)`` empirical quantile of the scores with the standard
    finite-sample correction ``(1 + 1/n)``. Subtract ``Q`` from the lower bound and add it to
    the upper bound to obtain a calibrated interval. ``alpha`` is the target *miscoverage*
    (e.g. 0.2 for an 80% interval). ``Q`` can be negative, which *tightens* an interval that
    was too wide.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        return 0.0
    # Conservative finite-sample level; clipped to 1.0 so tiny calibration sets don't overflow.
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
    return float(np.quantile(scores, level, method="higher"))


def apply_margin(lower, upper, q: float, median=None) -> tuple[np.ndarray, np.ndarray]:
    """Widen (or tighten) an interval by margin ``q``: ``[lower - q, upper + q]``.

    If ``median`` is given, the result is clipped so the bounds never cross it — a guard
    against a negative ``q`` inverting a very wide interval.
    """
    lower = np.asarray(lower, dtype=float) - q
    upper = np.asarray(upper, dtype=float) + q
    if median is not None:
        median = np.asarray(median, dtype=float)
        lower = np.minimum(lower, median)
        upper = np.maximum(upper, median)
    return lower, upper


def calibrate_interval(
    lower_test,
    upper_test,
    lower_cal,
    upper_cal,
    y_cal,
    alpha: float,
    median_test=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return CQR-adjusted ``lower, upper`` bounds for the test set, plus the margin ``Q``.

    ``*_cal`` are the model's predicted bounds and actuals on the calibration set; ``*_test``
    are the raw predicted bounds to be corrected. Convenience wrapper that scores the
    calibration set, derives the margin and applies it in one call — used where the same model
    produces both the calibration and test predictions.
    """
    scores = conformity_scores(y_cal, lower_cal, upper_cal)
    q = conformal_margin(scores, alpha)
    lower, upper = apply_margin(lower_test, upper_test, q, median_test)
    return lower, upper, q
