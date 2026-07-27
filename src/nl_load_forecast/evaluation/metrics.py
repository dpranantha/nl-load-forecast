"""Probabilistic forecast metrics.

The headline for a quantile forecast is not RMSE — it's whether the intervals are
*calibrated* (a P90 exceeded ~10% of the time) and *sharp* (narrow while staying calibrated).
"""
from __future__ import annotations

import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball (quantile) loss for a single quantile level.

    L_q = mean( max(q * (y - yhat), (q - 1) * (y - yhat)) )
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def mean_pinball_loss(y_true: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    """Average pinball loss across a set of quantile predictions."""
    return float(np.mean([pinball_loss(y_true, p, q) for q, p in preds.items()]))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage: fraction of observations within [lower, upper]."""
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    return float(np.mean(inside))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of the prediction interval (sharpness; lower is better if calibrated)."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def crps_from_quantiles(y_true: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    """Approximate CRPS (Continuous Rank Prediction Score) from a discrete set of quantiles.

    CRPS = 2 * integral_0^1 pinball_q dq, approximated by averaging the pinball loss over
    the provided (evenly spaced, ideally) quantile levels and multiplying by 2.
    """
    if not preds:
        raise ValueError("preds must contain at least one quantile")
    return 2.0 * mean_pinball_loss(y_true, preds)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))
