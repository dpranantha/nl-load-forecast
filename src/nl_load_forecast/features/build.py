"""Feature engineering: join load + weather, add calendar, lag and rolling features.

Leakage discipline: every feature derived from the target must only reference values that
are known at forecast time. Because the day-ahead backtest forecasts a whole ``horizon_hours``
block from a single origin cutoff, *any* target-derived feature is shifted by ``horizon_hours``
(not 1h): for the last hour of the block, load from earlier in the same block is still in the
future and must not leak in. The rolling backtest additionally enforces the train/predict
cutoff. See ``tests/test_features.py::test_rolling_features_are_horizon_safe``.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from ..config import FeaturesConfig


def _calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    nl_holidays = holidays.Netherlands()
    cal = pd.DataFrame(index=idx)
    cal["hour"] = idx.hour
    cal["dayofweek"] = idx.dayofweek
    cal["month"] = idx.month
    cal["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    cal["is_holiday"] = pd.Series(idx.date, index=idx).map(lambda d: d in nl_holidays).astype(int)
    # Cyclical encodings so the model sees hour 23 and hour 0 as adjacent.
    cal["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    cal["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    return cal


def _weather_derived_features(df: pd.DataFrame) -> None:
    """Add non-linear / interaction weather features in place, guarding on availability.

    Note: a pure monotone transform of a single variable (e.g. wind_speed**3) is a no-op for a
    tree model, since splits are order-based — so those are intentionally omitted. HDH/CDH encode
    the well-known non-linear temperature-load V-shape; wind chill is a genuine temp x wind
    interaction that shallow trees would otherwise need deep splits to recover.
    """
    if "temperature_2m" in df.columns:
        temp = df["temperature_2m"]
        # Heating / cooling degree hours (thresholds are standard NL comfort bands).
        df["hdh"] = np.maximum(0.0, 15.0 - temp)
        df["cdh"] = np.maximum(0.0, temp - 22.0)
        if "wind_speed_10m" in df.columns:
            wind_kmh = df["wind_speed_10m"] * 3.6
            # Environment Canada wind-chill index; only defined for temp<=10C & wind>4.8km/h.
            df["wind_chill"] = np.where(
                (temp <= 10.0) & (wind_kmh > 4.8),
                13.12 + 0.6215 * temp - 11.37 * wind_kmh**0.16 + 0.3965 * temp * wind_kmh**0.16,
                temp,
            )


def build_features(
    load: pd.DataFrame,
    weather: pd.DataFrame,
    cfg: FeaturesConfig,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Return a modelling frame: target column + all features, NaNs from lags dropped.

    ``horizon_hours`` is the day-ahead forecast horizon; all target-derived rolling/momentum
    features are shifted by it so nothing from inside the forecast block leaks into a prediction.
    """
    df = load.join(weather, how="inner").sort_index()

    target = cfg.target
    if target not in df.columns:
        raise KeyError(f"target {target!r} not in joined frame columns {list(df.columns)}")

    h = horizon_hours

    # Autoregressive lags (same hour yesterday / last week, etc.).
    for lag in cfg.lags_hours:
        df[f"{target}_lag_{lag}h"] = df[target].shift(lag)

    # Rolling means over values known at forecast time (shifted by the full horizon).
    for win in cfg.rolling_windows_hours:
        df[f"{target}_rollmean_{win}h"] = df[target].shift(h).rolling(win).mean()

    if cfg.add_momentum:
        # Day-over-day momentum: change between the two most recent knowable same-hour values.
        df[f"{target}_diff_{h}h"] = df[target].shift(h) - df[target].shift(2 * h)
        # Recent volatility — informs how wide the P10-P90 interval should be.
        for win in cfg.rolling_windows_hours:
            df[f"{target}_rollstd_{win}h"] = df[target].shift(h).rolling(win).std()

    if cfg.add_weather_derived:
        _weather_derived_features(df)

    if cfg.add_calendar:
        df = df.join(_calendar_features(df.index))

    return df.dropna()


def split_xy(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    y = frame[target]
    x = frame.drop(columns=[target])
    return x, y
