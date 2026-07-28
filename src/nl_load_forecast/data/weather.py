"""Fetch historical weather from Open-Meteo as an NWP proxy.

Open-Meteo's archive API is free and needs no key. For a true day-ahead setup we would
use the *forecast* endpoint at prediction time; the archive (reanalysis) is the right choice
for building and backtesting a model on history.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast" 


def _cache_path(cache_dir: str, lat: float, lon: float, start: str, end: str) -> Path:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(cache_dir) / f"weather_{lat}_{lon}_{start}_{end}.parquet"


def fetch_weather(
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    variables: list[str],
    timezone: str,
    cache_dir: str = "data",
    url: str = ARCHIVE_URL,
) -> pd.DataFrame:
    """Return an hourly, tz-aware DataFrame of the requested weather variables."""
    cache = _cache_path(cache_dir, latitude, longitude, start, end)
    if cache.exists():
        return pd.read_parquet(cache)

    # Request in UTC (which has no DST gaps/overlaps) and convert to the target zone below.
    # Fetching directly in a DST-observing zone makes Open-Meteo emit local wall-clock stamps
    # that include the spring-forward gap hour (e.g. 2023-03-26 02:00 Europe/Amsterdam), which
    # tz_localize cannot resolve (NonExistentTimeError).
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df.pop("time")).dt.tz_localize("UTC").dt.tz_convert(timezone)
    df = df.set_index("timestamp").sort_index()

    df.to_parquet(cache)
    return df
