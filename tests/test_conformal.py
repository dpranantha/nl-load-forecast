"""Unit tests for Conformalized Quantile Regression (CQR) helpers.

These run on synthetic data (no network, no LightGBM) and pin the two properties that make
CQR worth having: it *increases* coverage when the raw interval is too narrow, and it lands
close to the nominal level.
"""
from __future__ import annotations

import numpy as np

from nl_load_forecast.backtest.conformal import (
    calibrate_interval,
    conformal_margin,
    conformity_scores,
)


def test_conformity_scores_sign():
    # y below lower -> positive; inside -> negative; above upper -> positive.
    y = np.array([0.0, 5.0, 12.0])
    lower = np.array([2.0, 2.0, 2.0])
    upper = np.array([10.0, 10.0, 10.0])
    scores = conformity_scores(y, lower, upper)
    assert scores[0] == 2.0     # y=0 below lower: 2 - 0
    assert scores[2] == 2.0     # y=12 above upper: 12 - 10
    # y=5 comfortably inside: max(lower-y, y-upper) = max(2-5, 5-10) = max(-3, -5) = -3.
    assert scores[1] == -3.0


def test_conformal_margin_widens_when_interval_too_narrow():
    # Actuals routinely fall outside a tight interval -> positive margin (widen).
    y = np.array([0.0, 20.0] * 50)
    lower = np.full(100, 9.0)
    upper = np.full(100, 11.0)
    scores = conformity_scores(y, lower, upper)
    q = conformal_margin(scores, alpha=0.2)
    assert q > 0


def test_conformal_margin_can_tighten_when_interval_too_wide():
    # Actuals always well inside a huge interval -> negative margin (tighten).
    y = np.full(100, 10.0)
    lower = np.full(100, -100.0)
    upper = np.full(100, 100.0)
    q = conformal_margin(conformity_scores(y, lower, upper), alpha=0.2)
    assert q < 0


def test_calibrate_interval_achieves_nominal_coverage():
    """End-to-end: deliberately-too-narrow Gaussian intervals -> CQR restores ~80% coverage."""
    rng = np.random.default_rng(0)
    # True data ~ N(0, 1). A well-specified 80% interval would be about [-1.28, 1.28].
    y_cal = rng.normal(size=5000)
    y_test = rng.normal(size=5000)
    # Model is overconfident: it predicts a far-too-narrow interval [-0.3, 0.3].
    lo_cal, hi_cal = np.full(5000, -0.3), np.full(5000, 0.3)
    lo_test, hi_test = np.full(5000, -0.3), np.full(5000, 0.3)

    raw_cov = np.mean((y_test >= lo_test) & (y_test <= hi_test))
    lower, upper, q = calibrate_interval(
        lo_test, hi_test, lo_cal, hi_cal, y_cal, alpha=0.2,
    )
    cqr_cov = np.mean((y_test >= lower) & (y_test <= upper))

    assert raw_cov < 0.5, "sanity: the raw interval should be badly under-covered"
    assert q > 0, "should widen a too-narrow interval"
    assert abs(cqr_cov - 0.80) < 0.03, f"CQR coverage {cqr_cov:.3f} should be ~0.80"


def test_calibrate_interval_does_not_cross_median():
    # A pathologically wide-then-tightened interval must still bracket the median.
    y_cal = np.full(200, 10.0)
    lo_cal, hi_cal = np.full(200, -50.0), np.full(200, 70.0)
    median = np.full(5, 10.0)
    lower, upper, q = calibrate_interval(
        np.full(5, -50.0), np.full(5, 70.0), lo_cal, hi_cal, y_cal,
        alpha=0.2, median_test=median,
    )
    assert np.all(lower <= median)
    assert np.all(upper >= median)
