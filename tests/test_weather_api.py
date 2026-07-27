from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from requests import HTTPError

from nl_load_forecast.data.weather import _cache_path, fetch_weather


@pytest.fixture
def mock_openmeteo_response():
    """Mock JSON response payload returned by Open-Meteo archive API."""
    return {
        "hourly": {
            "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
            "temperature_2m": [5.2, 4.8],
            "wind_speed_10m": [12.1, 11.5],
            "shortwave_radiation": [0.0, 0.0],
        }
    }


def test_cache_path_formatting(tmp_path):
    path = _cache_path(str(tmp_path), 52.10, 5.18, "2023-01-01", "2023-01-02")
    assert path == tmp_path / "weather_52.1_5.18_2023-01-01_2023-01-02.parquet"


@patch("nl_load_forecast.data.weather.requests.get")
def test_fetch_weather_api_call_and_structure(
    mock_get, mock_openmeteo_response, tmp_path
):
    """Test API response parsing, datetime localization, and index setting."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_openmeteo_response
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    cache_dir = str(tmp_path / "data_cache")

    df = fetch_weather(
        latitude=52.10,
        longitude=5.18,
        start="2023-01-01",
        end="2023-01-02",
        variables=["temperature_2m", "wind_speed_10m", "shortwave_radiation"],
        timezone="Europe/Amsterdam",
        cache_dir=cache_dir,
    )

    # Print directly to stdout
    # print("\n--- Processed Weather DataFrame ---")
    # print(df)
    # print("------------------------------------\n")

    # 1. Assert API was called once with comma-joined variables
    mock_get.assert_called_once()
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["hourly"] == "temperature_2m,wind_speed_10m,shortwave_radiation"

    # 2. Assert DataFrame shape and index characteristics
    assert len(df) == 2
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "Europe/Amsterdam"
    assert list(df.columns) == [
        "temperature_2m",
        "wind_speed_10m",
        "shortwave_radiation",
    ]

    # 3. Assert parquet cache file was actually written to disk
    cache_file = _cache_path(cache_dir, 52.10, 5.18, "2023-01-01", "2023-01-02")
    assert cache_file.exists()


@patch("nl_load_forecast.data.weather.requests.get")
def test_fetch_weather_uses_cache_on_second_call(
    mock_get, mock_openmeteo_response, tmp_path
):
    """Verify that subsequent calls load directly from disk without hitting the network."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_openmeteo_response
    mock_get.return_value = mock_resp

    cache_dir = str(tmp_path / "cache")
    args = (52.10, 5.18, "2023-01-01", "2023-01-02", ["temperature_2m"], "Europe/Amsterdam")

    # First call -> fetches from API and caches
    df_first = fetch_weather(*args, cache_dir=cache_dir)
    assert mock_get.call_count == 1

    # Second call -> should read parquet file without triggering API requests.get
    df_second = fetch_weather(*args, cache_dir=cache_dir)
    assert mock_get.call_count == 1  # count remains 1

    pd.testing.assert_frame_equal(df_first, df_second)


@patch("nl_load_forecast.data.weather.requests.get")
def test_fetch_weather_raises_on_http_error(mock_get, tmp_path):
    """Ensure HTTP errors (4xx/5xx) bubble up properly."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = HTTPError("404 Client Error")
    mock_get.return_value = mock_resp

    with pytest.raises(HTTPError):
        fetch_weather(
            latitude=52.10,
            longitude=5.18,
            start="2023-01-01",
            end="2023-01-02",
            variables=["temperature_2m"],
            timezone="Europe/Amsterdam",
            cache_dir=str(tmp_path),
        )