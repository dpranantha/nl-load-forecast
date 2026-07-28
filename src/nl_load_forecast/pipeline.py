"""End-to-end pipeline: data -> features -> rolling backtest -> metrics -> MLflow.

Run via ``scripts/run_backtest.py``. Logs params, metrics and a calibration artifact to
MLflow, and registers the final model trained on the full window.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import pandas as pd

from .backtest.rolling import rolling_backtest
from .config import Config
from .data.entsoe import fetch_load
from .data.weather import fetch_weather
from .evaluation import metrics
from .features.build import build_features, split_xy
from .models.quantile_lgbm import MultiQuantileLGBM

matplotlib.use("Agg")  # headless — set at import time, before any figure is created


def _on_databricks() -> bool:
    """True on a Databricks cluster — the same signal used to defer to managed MLflow tracking."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def _load_frame_local(cfg: Config) -> pd.DataFrame:
    """Fetch load + weather from the public APIs and build the modelling frame (local default)."""
    load = fetch_load(
        cfg.data.country_code, cfg.data.start, cfg.data.end,
        cfg.data.timezone, cfg.data.cache_dir,
    )
    weather = fetch_weather(
        cfg.data.weather_latitude, cfg.data.weather_longitude,
        cfg.data.start, cfg.data.end, cfg.features.weather_vars,
        cfg.data.timezone, cfg.data.cache_dir,
    )
    return build_features(load, weather, cfg.features, horizon_hours=cfg.backtest.horizon_hours)


def _load_frame_databricks(cfg: Config) -> pd.DataFrame:
    """Read raw load+weather from the Databricks Feature Store, then run the *same*
    ``build_features`` as local so metrics are identical regardless of where features come from.

    Imports are lazy on purpose: ``databricks-feature-engineering`` only exists on a cluster, so
    importing it at module top would break local runs and CI. This path is therefore not exercised
    by CI — see the "Known limitations" section of the README.
    """
    from databricks.feature_engineering import FeatureEngineeringClient

    fe = FeatureEngineeringClient()
    # The notebook registers a table keyed by ``timestamp`` holding the raw joined load+weather.
    pdf = (
        fe.read_table(name=cfg.data.feature_table)
        .toPandas()
        .set_index("timestamp")
        .sort_index()
    )
    load = pdf[[cfg.features.target]]
    weather = pdf[cfg.features.weather_vars]
    return build_features(load, weather, cfg.features, horizon_hours=cfg.backtest.horizon_hours)


def _load_frame(cfg: Config) -> pd.DataFrame:
    """Load the modelling frame, from the Feature Store on Databricks (if configured) or the
    public APIs locally. Feature engineering is shared, so the two paths yield the same frame."""
    if _on_databricks() and cfg.data.feature_table:
        return _load_frame_databricks(cfg)
    return _load_frame_local(cfg)


def _quantile_cols(quantiles: list[float]) -> dict[float, str]:
    return {q: f"q{int(q * 100)}" for q in quantiles}


def _evaluate(bt: pd.DataFrame, quantiles: list[float]) -> dict[str, float]:
    cols = _quantile_cols(quantiles)
    y = bt["y_true"].to_numpy()
    preds = {q: bt[c].to_numpy() for q, c in cols.items()}

    result = {
        "mean_pinball": metrics.mean_pinball_loss(y, preds),
        "crps": metrics.crps_from_quantiles(y, preds),
    }
    if 0.5 in cols:
        result["p50_mae"] = metrics.mae(y, bt[cols[0.5]].to_numpy())
    lo, hi = min(quantiles), max(quantiles)
    result["interval_coverage"] = metrics.coverage(y, bt[cols[lo]], bt[cols[hi]])
    result["interval_target"] = hi - lo
    result["interval_width"] = metrics.interval_width(bt[cols[lo]], bt[cols[hi]])
    return result


def _calibration_plot(bt: pd.DataFrame, quantiles: list[float], path: Path) -> None:
    cols = _quantile_cols(quantiles)
    y = bt["y_true"].to_numpy()
    observed = [(y <= bt[c].to_numpy()).mean() for c in cols.values()]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="ideal")
    ax.plot(list(quantiles), observed, "o-", label="observed")
    ax.set_xlabel("nominal quantile")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(config_path: str) -> dict[str, float]:
    cfg = Config.load(config_path)

    frame = _load_frame(cfg)

    bt = rolling_backtest(frame, cfg.features.target, cfg.model, cfg.backtest)
    scores = _evaluate(bt, cfg.model.quantiles)

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    calib_path = reports / "calibration.png"
    _calibration_plot(bt, cfg.model.quantiles, calib_path)

    # Print metrics before MLflow logging so results always surface, even if tracking fails.
    print("Backtest metrics:")
    for k, v in scores.items():
        print(f"  {k:20s} {v:.4f}")

    # Locally, use the configured backend (SQLite by default). On Databricks, MLflow tracking
    # is managed by the platform, so leave it alone rather than redirecting to a driver-local DB.
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ and cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run():
        mlflow.log_params({
            "quantiles": cfg.model.quantiles,
            "horizon_hours": cfg.backtest.horizon_hours,
            "n_folds": cfg.backtest.n_folds,
            "conformalize": cfg.backtest.conformalize,
            "calibration_days": cfg.backtest.calibration_days,
            **{f"lgbm_{k}": v for k, v in cfg.model.lgbm_params.items()},
        })
        mlflow.log_metrics(scores)
        mlflow.log_artifact(str(calib_path))

        # Fit a final model on the whole window and register it.
        x_all, y_all = split_xy(frame, cfg.features.target)
        final = MultiQuantileLGBM(cfg.model.quantiles, cfg.model.lgbm_params).fit(x_all, y_all)
        final.feature_importances().to_csv(reports / "feature_importances.csv")
        mlflow.log_artifact(str(reports / "feature_importances.csv"))
        # Native LightGBM flavor: the sklearn flavor serialises via skops, which rejects
        # LightGBM boosters as "untrusted types". log the median (P50) model for reference.
        mlflow.lightgbm.log_model(
            final.models_[0.5],
            name="p50_model",
            registered_model_name=cfg.mlflow.registered_model_name,
        )

    return scores
