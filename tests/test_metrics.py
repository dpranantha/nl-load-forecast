import numpy as np

from nl_load_forecast.evaluation import metrics


def test_pinball_zero_when_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.pinball_loss(y, y, 0.5) == 0.0


def test_pinball_median_is_half_mae():
    # For q=0.5 the pinball loss equals 0.5 * MAE.
    y = np.array([10.0, 20.0, 30.0])
    yhat = np.array([12.0, 18.0, 33.0])
    expected = 0.5 * metrics.mae(y, yhat)
    assert np.isclose(metrics.pinball_loss(y, yhat, 0.5), expected)


def test_pinball_penalises_under_and_over_asymmetrically():
    y = np.array([100.0])
    under = metrics.pinball_loss(y, np.array([90.0]), 0.9)   # forecast too low
    over = metrics.pinball_loss(y, np.array([110.0]), 0.9)   # forecast too high
    # At q=0.9 under-prediction should be penalised more heavily than over-prediction.
    assert under > over


def test_coverage_full_and_empty():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.coverage(y, np.array([0, 0, 0]), np.array([5, 5, 5])) == 1.0
    assert metrics.coverage(y, np.array([9, 9, 9]), np.array([10, 10, 10])) == 0.0


def test_crps_is_twice_mean_pinball():
    y = np.array([5.0, 6.0, 7.0])
    preds = {0.25: np.array([4.0, 6.0, 6.0]), 0.75: np.array([6.0, 7.0, 8.0])}
    assert np.isclose(
        metrics.crps_from_quantiles(y, preds),
        2.0 * metrics.mean_pinball_loss(y, preds),
    )
