"""Multi-quantile LightGBM: one gradient-boosted model per quantile.

Each quantile is fit with LightGBM's pinball ("quantile") objective. Predictions are
sorted across quantiles per row to prevent quantile crossing (a P90 below a P50).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


class MultiQuantileLGBM:
    def __init__(self, quantiles: list[float], lgbm_params: dict | None = None):
        if not all(0.0 < q < 1.0 for q in quantiles):
            raise ValueError("quantiles must be strictly between 0 and 1")
        self.quantiles = sorted(quantiles)
        self.lgbm_params = lgbm_params or {}
        self.models_: dict[float, LGBMRegressor] = {}

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "MultiQuantileLGBM":
        for q in self.quantiles:
            model = LGBMRegressor(objective="quantile", alpha=q, **self.lgbm_params)
            model.fit(x, y)
            self.models_[q] = model
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with one column per quantile, monotone across quantiles."""
        preds = {f"q{int(q * 100)}": self.models_[q].predict(x) for q in self.quantiles}
        out = pd.DataFrame(preds, index=x.index)
        # Enforce monotonicity row-wise (guards against quantile crossing).
        sorted_vals = np.sort(out.to_numpy(), axis=1)
        return pd.DataFrame(sorted_vals, columns=out.columns, index=out.index)

    def feature_importances(self) -> pd.Series:
        """Mean gain-based importance across the per-quantile models."""
        booster = self.models_[self.quantiles[0]]
        names = booster.feature_name_
        stacked = np.vstack([m.feature_importances_ for m in self.models_.values()])
        return pd.Series(stacked.mean(axis=0), index=names).sort_values(ascending=False)
