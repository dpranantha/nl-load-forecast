from __future__ import annotations

import pandas as pd
import pytest

from nl_load_forecast.data.weather import (
    ARCHIVE_URL,
    FORECAST_URL,
    _cache_path,
    fetch_weather,
)

@pytest.mark.integration
@pytest.mark.parametrize(
    "url_type, api_url, start_date, end_date",
    [
        ("archive", ARCHIVE_URL, "2023-05-01", "2023-05-02"),
        ("forecast", FORECAST_URL, "2026-08-01", "2026-08-02"),
    ],
)
def test_live_fetch_and_cache(
    url_type: str,
    api_url: str,
    start_date: str,
    end_date: str,
    tmp_path,
):
    """Hits both Archive and Forecast endpoints, checks structure, and tests caching."""
    latitude = 52.10
    longitude = 5.18
    variables = ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]
    timezone = "Europe/Amsterdam"
    cache_dir = str(tmp_path / f"live_cache_{url_type}")

    # 1. Fetch live data using specific URL endpoint
    df_live = fetch_weather(
        latitude=latitude,
        longitude=longitude,
        start=start_date,
        end=end_date,
        variables=variables,
        timezone=timezone,
        cache_dir=cache_dir,
        url=api_url,
    )

    # 2. Console Output
    print(f"\n{'=' * 50}")
    print(f"LIVE OPEN-METEO DATAFRAME [{url_type.upper()} ENDPOINT]:")
    print(f"{'=' * 50}")
    print(df_live.head())
    print(f"{'-' * 50}")
    print(f"Total Rows: {len(df_live)}")
    print(f"Timezone: {df_live.index.tz}")
    print(f"{'=' * 50}\n")

    # 3. Assertions
    assert not df_live.empty, "DataFrame should not be empty"
    assert len(df_live) == 48, "2 days of hourly data should yield 48 rows"
    assert df_live.index.name == "timestamp"
    assert str(df_live.index.tz) == timezone
    assert list(df_live.columns) == variables

    # 4. Verify cache file was written
    expected_cache_file = _cache_path(
        cache_dir, latitude, longitude, start_date, end_date
    )
    assert expected_cache_file.exists(), "Cache file was not created"

    # 5. Verify local cache read
    df_cached = fetch_weather(
        latitude=latitude,
        longitude=longitude,
        start=start_date,
        end=end_date,
        variables=variables,
        timezone=timezone,
        cache_dir=cache_dir,
        url=api_url,
    )

    pd.testing.assert_frame_equal(df_live, df_cached)