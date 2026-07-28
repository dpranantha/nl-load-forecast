from __future__ import annotations

import os

import pandas as pd
import pytest
from dotenv import load_dotenv

from nl_load_forecast.data.entsoe import _cache_path, fetch_load

# ENTSO-E needs an API token; skip cleanly when it isn't configured so this file
# is safe to run locally without credentials (CI deselects it via -m "not integration").
load_dotenv()
_TOKEN = os.environ.get("ENTSOE_API_TOKEN")
pytestmark = pytest.mark.skipif(
    not _TOKEN, reason="ENTSOE_API_TOKEN not set; skipping live ENTSO-E integration test"
)


@pytest.mark.integration
def test_live_fetch_and_cache(tmp_path):
    """Hits the live ENTSO-E query_load endpoint, checks structure, and tests caching."""
    country_code = "NL"
    start = "2023-05-01"
    end = "2023-05-03"  # 2-day window; end is exclusive in entsoe-py
    timezone = "Europe/Amsterdam"
    cache_dir = str(tmp_path / "live_cache_entsoe")

    # 1. Fetch live data
    df_live = fetch_load(
        country_code=country_code,
        start=start,
        end=end,
        timezone=timezone,
        cache_dir=cache_dir,
    )

    # 2. Console output (mirrors the weather integration test for easy eyeballing)
    print(f"\n{'=' * 50}")
    print("LIVE ENTSO-E DATAFRAME [query_load]:")
    print(f"{'=' * 50}")
    print(df_live.head())
    print(f"{'-' * 50}")
    print(f"Total Rows: {len(df_live)}")
    print(f"Timezone: {df_live.index.tz}")
    print(f"{'=' * 50}\n")

    # 3. Structure assertions
    assert not df_live.empty, "DataFrame should not be empty"
    assert list(df_live.columns) == ["load_mw"], "expected a single 'load_mw' column"
    assert df_live.index.name == "timestamp"
    assert str(df_live.index.tz) == timezone
    assert df_live.index.is_monotonic_increasing
    # Normalised to hourly: a 2-day window should be ~48 rows, and spacing should be 1h.
    assert 44 <= len(df_live) <= 50, f"unexpected row count for a 2-day window: {len(df_live)}"
    assert df_live.index.to_series().diff().median() == pd.Timedelta(hours=1)
    assert df_live["load_mw"].notna().any(), "load column is entirely NaN"

    # 4. Verify the cache file was written
    expected_cache_file = _cache_path(cache_dir, country_code, start, end)
    assert expected_cache_file.exists(), "Cache file was not created"

    # 5. Verify the cached read matches the live read
    df_cached = fetch_load(
        country_code=country_code,
        start=start,
        end=end,
        timezone=timezone,
        cache_dir=cache_dir,
    )
    # check_freq=False: resample() stamps a freq on the live index, but parquet doesn't
    # persist that metadata, so the cached read has freq=None. The data itself is identical.
    pd.testing.assert_frame_equal(df_live, df_cached, check_freq=False)
