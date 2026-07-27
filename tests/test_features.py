import numpy as np
import pandas as pd

from nl_load_forecast.config import FeaturesConfig
from nl_load_forecast.features.build import build_features, split_xy, _calendar_features


def _fake_data(n=500):
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="Europe/Amsterdam")
    load = pd.DataFrame({"load_mw": 10000 + 1000 * np.sin(np.arange(n) / 24)}, index=idx)
    weather = pd.DataFrame(
        {
            "temperature_2m": 5 + 5 * np.cos(np.arange(n) / 24),
            "wind_speed_10m": np.abs(np.sin(np.arange(n) / 12)) * 10,
            "shortwave_radiation": np.clip(np.sin(np.arange(n) / 24), 0, None) * 400,
        },
        index=idx,
    )
    return load, weather


def _cfg():
    return FeaturesConfig(
        target="load_mw",
        weather_vars=["temperature_2m", "wind_speed_10m", "shortwave_radiation"],
        lags_hours=[24, 168],
        rolling_windows_hours=[24],
        add_calendar=True,
    )


def test_build_features_has_no_nans_and_expected_columns():
    load, weather = _fake_data()
    frame = build_features(load, weather, _cfg())
    assert not frame.isna().any().any()
    assert "load_mw_lag_24h" in frame.columns
    assert "load_mw_rollmean_24h" in frame.columns
    assert "is_holiday" in frame.columns


def test_lag_feature_matches_shifted_target():
    load, weather = _fake_data()
    frame = build_features(load, weather, _cfg())
    # lag_24h at time t should equal the target 24h earlier.
    t = frame.index[100]
    assert np.isclose(frame.loc[t, "load_mw_lag_24h"], load["load_mw"].shift(24).loc[t])


def test_split_xy_separates_target():
    load, weather = _fake_data()
    frame = build_features(load, weather, _cfg())
    x, y = split_xy(frame, "load_mw")
    assert "load_mw" not in x.columns
    assert y.name == "load_mw"


def test_rolling_features_are_horizon_safe():
    """Rolling/momentum features must only use target values known at forecast time.

    For a day-ahead (24h) forecast made from a single origin, the rolling mean at time t may
    only reference load up to t-24h. This is the regression test for the old shift(1) leak.
    """
    load, weather = _fake_data()
    h = 24
    frame = build_features(load, weather, _cfg(), horizon_hours=h)
    t = frame.index[300]

    safe = load["load_mw"].shift(h).rolling(24).mean().loc[t]
    leaky = load["load_mw"].shift(1).rolling(24).mean().loc[t]
    assert np.isclose(frame.loc[t, "load_mw_rollmean_24h"], safe)
    # The horizon-safe value must differ from the old leaky shift(1) version.
    assert not np.isclose(frame.loc[t, "load_mw_rollmean_24h"], leaky)


def test_momentum_features_present_and_correct():
    load, weather = _fake_data()
    h = 24
    frame = build_features(load, weather, _cfg(), horizon_hours=h)
    assert "load_mw_diff_24h" in frame.columns
    assert "load_mw_rollstd_24h" in frame.columns
    t = frame.index[300]
    expected_diff = load["load_mw"].shift(h).loc[t] - load["load_mw"].shift(2 * h).loc[t]
    assert np.isclose(frame.loc[t, "load_mw_diff_24h"], expected_diff)


def test_weather_derived_features_present():
    load, weather = _fake_data()
    frame = build_features(load, weather, _cfg())
    for col in ("hdh", "cdh", "wind_chill"):
        assert col in frame.columns
    # HDH and CDH are mutually exclusive non-negative degree-hours.
    assert (frame["hdh"] >= 0).all()
    assert (frame["cdh"] >= 0).all()
    assert (frame["hdh"] * frame["cdh"] == 0).all()
    # wind_speed_cubed was intentionally dropped (no-op for a tree model).
    assert "wind_speed_cubed" not in frame.columns


def test_feature_flags_toggle_off():
    load, weather = _fake_data()
    cfg = _cfg()
    cfg.add_momentum = False
    cfg.add_weather_derived = False
    frame = build_features(load, weather, cfg)
    assert not any(c.startswith(("load_mw_diff", "load_mw_rollstd")) for c in frame.columns)
    for col in ("hdh", "cdh", "wind_chill"):
        assert col not in frame.columns

def test_calendar_features_holidays_and_weekends():
    """Verify that known Dutch holidays and weekends are correctly flagged."""
    # Test dates across 2023:
    # 1. 2023-01-01: New Year's Day (Sunday) -> Holiday=1, Weekend=1
    # 2. 2023-04-27: King's Day (Koningsdag) (Thursday) -> Holiday=1, Weekend=0
    # 3. 2023-12-25: Christmas Day (Eerste Kerstdag) (Monday) -> Holiday=1, Weekend=0
    # 4. 2023-05-17: Regular Wednesday -> Holiday=0, Weekend=0
    dates = [
        "2023-01-01 12:00:00",
        "2023-04-27 09:00:00",
        "2023-12-25 18:00:00",
        "2023-05-17 14:00:00",
    ]
    idx = pd.DatetimeIndex(dates, tz="Europe/Amsterdam")

    df = _calendar_features(idx)

    # 1. Verify Holiday Flagging
    assert df.loc["2023-01-01 12:00:00", "is_holiday"] == 1  # New Year's Day
    assert df.loc["2023-04-27 09:00:00", "is_holiday"] == 1  # Koningsdag
    assert df.loc["2023-12-25 18:00:00", "is_holiday"] == 1  # Christmas Day
    assert df.loc["2023-05-17 14:00:00", "is_holiday"] == 0  # Regular Wednesday

    # 2. Verify Weekend Flagging
    assert df.loc["2023-01-01 12:00:00", "is_weekend"] == 1  # Sunday
    assert df.loc["2023-04-27 09:00:00", "is_weekend"] == 0  # Thursday
    assert df.loc["2023-12-25 18:00:00", "is_weekend"] == 0  # Monday


def test_calendar_features_structure_and_cyclical_encoding():
    """Verify shape, hour bounds, and cyclical sin/cos encodings."""
    idx = pd.date_range("2023-01-01", "2023-01-02", freq="h", tz="Europe/Amsterdam")
    df = _calendar_features(idx)

    # Check columns structure
    expected_cols = [
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "is_holiday",
        "hour_sin",
        "hour_cos",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 25

    # Check cyclical boundary continuity: sin(0) == 0, cos(0) == 1
    hour_0 = df[df["hour"] == 0].iloc[0]
    np.testing.assert_allclose(hour_0["hour_sin"], 0.0, atol=1e-5)
    np.testing.assert_allclose(hour_0["hour_cos"], 1.0, atol=1e-5)

    # Check hour 6: sin(6*2pi/24) == sin(pi/2) == 1.0
    hour_6 = df[df["hour"] == 6].iloc[0]
    np.testing.assert_allclose(hour_6["hour_sin"], 1.0, atol=1e-5)
