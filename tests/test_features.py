import numpy as np
import pandas as pd

from nl_load_forecast.config import FeaturesConfig
from nl_load_forecast.features.build import build_features, split_xy


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
