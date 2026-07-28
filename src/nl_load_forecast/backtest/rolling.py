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
from . import conformal


def _qcol(q: float) -> str:
    return f"q{int(q * 100)}"


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

    quantiles = sorted(model_cfg.quantiles)
    lo_col, hi_col = _qcol(quantiles[0]), _qcol(quantiles[-1])
    median_col = _qcol(0.5) if 0.5 in quantiles else None
    # Target miscoverage of the outer interval, e.g. 1 - (0.9 - 0.1) = 0.2 for an 80% band.
    alpha = 1.0 - (quantiles[-1] - quantiles[0])
    calib_window = pd.Timedelta(days=bt_cfg.calibration_days)

    results = []
    for fold in range(bt_cfg.n_folds):
        train = frame.loc[frame.index < origin]
        test = frame.loc[(frame.index >= origin) & (frame.index < origin + horizon)]
        if len(test) == 0 or len(train) < 24:
            break

        x_test, y_test = build.split_xy(test, target)

        # CQR margin: fit a *calibration* model on the earlier part of the window, then measure
        # how far actuals fell outside its predicted interval over the most recent
        # `calibration_days`. We deliberately do NOT deploy this reduced model — holding the
        # freshest days out of training badly hurts point accuracy on a drifting load series.
        # Instead we only borrow the scalar margin `q` from it.
        margin = None
        if bt_cfg.conformalize and bt_cfg.calibration_days > 0 and lo_col != hi_col:
            fit = train.loc[train.index < origin - calib_window]
            calib = train.loc[train.index >= origin - calib_window]
            if len(fit) >= 24 and len(calib) >= 24:
                x_fit, y_fit = build.split_xy(fit, target)
                x_cal, y_cal = build.split_xy(calib, target)
                cal_model = MultiQuantileLGBM(
                    model_cfg.quantiles, model_cfg.lgbm_params
                ).fit(x_fit, y_fit)
                cal_preds = cal_model.predict(x_cal)
                scores = conformal.conformity_scores(y_cal, cal_preds[lo_col], cal_preds[hi_col])
                margin = conformal.conformal_margin(scores, alpha)

        # Deployed model: fit on the FULL training window (through the origin, no recency gap).
        x_train, y_train = build.split_xy(train, target)
        model = MultiQuantileLGBM(model_cfg.quantiles, model_cfg.lgbm_params).fit(x_train, y_train)
        preds = model.predict(x_test)

        if margin is not None:
            preds[lo_col], preds[hi_col] = conformal.apply_margin(
                preds[lo_col], preds[hi_col], margin,
                median=preds[median_col] if median_col else None,
            )

        out = preds.copy()
        out["y_true"] = y_test.to_numpy()
        out["fold"] = fold
        results.append(out)

        origin = origin + step

    if not results:
        raise RuntimeError("Backtest produced no folds — check window sizes vs. data span.")
    return pd.concat(results).sort_index()
