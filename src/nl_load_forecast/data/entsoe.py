"""Fetch actual NL electricity load from the ENTSO-E Transparency Platform.

Requires a free API token in the ENTSOE_API_TOKEN environment variable.
Results are cached to parquet so we don't hammer the API on every run.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def _cache_path(cache_dir: str, country: str, start: str, end: str) -> Path:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(cache_dir) / f"load_{country}_{start}_{end}.parquet"


def fetch_load(
    country_code: str,
    start: str,
    end: str,
    timezone: str,
    cache_dir: str = "data",
) -> pd.DataFrame:
    """Return an hourly DataFrame indexed by tz-aware timestamp with a `load_mw` column.

    Uses ENTSO-E ``query_load``. Falls back to the parquet cache if present.
    """
    cache = _cache_path(cache_dir, country_code, start, end)
    if cache.exists():
        return pd.read_parquet(cache)

    load_dotenv()
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "ENTSOE_API_TOKEN not set. Copy .env.example to .env and add the free token."
        )

    # Imported lazily so tests / feature work don't require the dependency or network.
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=token)
    start_ts = pd.Timestamp(start, tz=timezone)
    end_ts = pd.Timestamp(end, tz=timezone)

    series = client.query_load(country_code, start=start_ts, end=end_ts)
    # entsoe-py returns a DataFrame with an "Actual Load" column (MW), 15-min or hourly.
    df = series.rename(columns={"Actual Load": "load_mw"})[["load_mw"]]
    df = df.resample("1h").mean()  # normalise to hourly
    df.index.name = "timestamp"

    df.to_parquet(cache)
    return df
