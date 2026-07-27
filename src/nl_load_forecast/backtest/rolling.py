"""Rolling-origin (walk-forward) backtest.

This is the honesty engine of the project: the model is only ever trained on data strictly
before each forecast origin, then evaluated on the next `horizon_hours`. The origin advances
by `step_days` each fold. No shuffling, no future leakage.
"""
from __future__ import annotations

import pandas as pd

from ..config import BacktestConfig, ModelConfig
from ..features import build
from ..models.quantile_lgbm import MultiQuantileLGBM


def rolling_backtest(
    frame: pd.DataFrame,
    target: str,
    model_cfg: ModelConfig,
    bt_cfg: BacktestConfig,
) -> pd.DataFrame:
    """Run the walk-forward backtest and return predictions with truth per timestamp.

    Returned columns: y_true, q10, q50, q90 (whatever quantiles are configured),
    plus `fold`.
    """
    frame = frame.sort_index()
    origin = frame.index.min() + pd.Timedelta(days=bt_cfg.initial_train_days)
    horizon = pd.Timedelta(hours=bt_cfg.horizon_hours)
    step = pd.Timedelta(days=bt_cfg.step_days)

    results = []
    for fold in range(bt_cfg.n_folds):
        train = frame.loc[frame.index < origin]
        test = frame.loc[(frame.index >= origin) & (frame.index < origin + horizon)]
        if len(test) == 0 or len(train) < 24:
            break

        x_train, y_train = build.split_xy(train, target)
        x_test, y_test = build.split_xy(test, target)

        model = MultiQuantileLGBM(model_cfg.quantiles, model_cfg.lgbm_params).fit(x_train, y_train)
        preds = model.predict(x_test)

        out = preds.copy()
        out["y_true"] = y_test.to_numpy()
        out["fold"] = fold
        results.append(out)

        origin = origin + step

    if not results:
        raise RuntimeError("Backtest produced no folds — check window sizes vs. data span.")
    return pd.concat(results).sort_index()
