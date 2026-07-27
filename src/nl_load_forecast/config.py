"""Typed configuration loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    country_code: str
    start: str
    end: str
    timezone: str
    weather_latitude: float
    weather_longitude: float
    cache_dir: str


@dataclass
class FeaturesConfig:
    target: str
    weather_vars: list[str]
    lags_hours: list[int]
    rolling_windows_hours: list[int]
    add_calendar: bool
    add_weather_derived: bool = True
    add_momentum: bool = True


@dataclass
class ModelConfig:
    quantiles: list[float]
    lgbm_params: dict


@dataclass
class BacktestConfig:
    horizon_hours: int
    n_folds: int
    initial_train_days: int
    step_days: int


@dataclass
class MlflowConfig:
    experiment_name: str
    registered_model_name: str


@dataclass
class Config:
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    backtest: BacktestConfig
    mlflow: MlflowConfig
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            data=DataConfig(**raw["data"]),
            features=FeaturesConfig(**raw["features"]),
            model=ModelConfig(**raw["model"]),
            backtest=BacktestConfig(**raw["backtest"]),
            mlflow=MlflowConfig(**raw["mlflow"]),
            raw=raw,
        )
