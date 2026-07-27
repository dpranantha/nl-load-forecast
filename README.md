# NL Day-Ahead Load Forecasting — Probabilistic, Weather-Driven

Day-ahead electricity **load** forecasting for the Netherlands, built as a *probabilistic*
(quantile) problem rather than a single point prediction — because in a balancing context
the cost of being wrong is asymmetric and a point forecast hides exactly the information we
need to act on.

**Stack:** Databricks · Spark · MLflow (tracking + registry) · LightGBM (quantile) · Feature Store
**Data:** ENTSO-E (actual NL load & generation) · Open-Meteo (weather / NWP) · NL holidays

---

## Why probabilistic, and why this matters for balancing

A party responsible for balancing a portfolio (a BRP) nominates a position ahead of delivery.
If the realised load differs from the forecast, the imbalance is settled against the imbalance
price — which is **volatile and asymmetric**: being short in a tight market can cost far more per
MWh than being long in a loose one.

That changes what "a good forecast" means:

- A **point forecast** (P50) minimises average error but tells us nothing about *risk*.
- A **quantile forecast** (P10 / P50 / P90) gives us a calibrated view of the downside — so the
  decision can be sized against the cost of imbalance, not just the expected value.

So this project optimises and reports **pinball loss**, **interval coverage/calibration**, and
**CRPS** — not RMSE alone. Calibration is the headline metric: a P90 that is only exceeded 70% of
the time is worse than useless, it is *misleading*.

---

## Results

> Populate this section from `scripts/run_backtest.py` output. Numbers below are placeholders —
> **do not publish until the backtest has actually produced them.** Each row maps to a key printed
> by the run (and logged to MLflow): paste the value straight in.

**Rolling-origin backtest** (walk-forward, 24h day-ahead horizon, N folds over the last _____):

| Metric | Printed key | Value | Reads as |
|---|---|---|---|
| Pinball loss (mean over P10/P50/P90) | `mean_pinball` | `___` | lower = sharper + better calibrated |
| P50 MAE | `p50_mae` | `___ MW` | point accuracy baseline |
| 80% interval coverage (P10–P90) | `interval_coverage` | `___` | target ≈ 0.80 → calibrated |
| Mean 80% interval width | `interval_width` | `___ MW` | sharpness (narrower is better *if* covered) |
| CRPS | `crps` | `___` | overall probabilistic score |

Calibration plot and a sample day-ahead fan chart live in `reports/` after a run.

**Baselines it must beat:** (1) seasonal naïve (load 168h ago), (2) P50-only GBM. If the quantile
model doesn't beat seasonal naïve on pinball loss, something is wrong — that's the honesty check.

---

## Feature engineering & leakage discipline

Features fall into three groups, all built in `src/nl_load_forecast/features/build.py`:

- **Weather (raw + derived):** temperature, wind, irradiance, plus **heating/cooling degree-hours**
  (the non-linear temperature–load V-shape) and a **wind-chill** term (a genuine temp×wind
  *interaction*). A pure monotone transform of one variable — e.g. `wind_speed³` — is deliberately
  *omitted*: for a tree model splits are order-based, so it adds no information.
- **Autoregressive lags:** same hour yesterday / 2 days / last week.
- **Momentum & volatility:** day-over-day change and a rolling std (the latter directly informs how
  wide the P10–P90 interval should be).

**The one rule that matters:** because the day-ahead backtest forecasts a whole `horizon_hours`
block from a single origin, *every* target-derived rolling/momentum feature is shifted by the
**full horizon (24h), not 1h**. For the last hour of the block, load from earlier in the *same*
block is still in the future — shifting by 1h would leak it in. This is enforced in `build.py` and
guarded by `tests/test_features.py::test_rolling_features_are_horizon_safe`.

---

## Architecture

```
ENTSO-E API ─┐
             ├─► Spark join (Databricks) ─► Feature Store ─► Quantile LGBM ─► Rolling backtest
Open-Meteo ──┘        (load + weather              (weather +   (P10/P50/P90    (pinball, coverage,
                       aligned on ts)               calendar)    per horizon)     CRPS) ─► MLflow
```

- **Weather is the real input signal**, not load history alone — day-ahead skill comes largely
  from the NWP forecast (temperature drives heating/cooling; wind & irradiance drive net load via
  behind-the-meter solar/wind). Lags carry the autoregressive component.
- Every backtest run is logged to **MLflow** (params, metrics, calibration artifact); the best run
  is promoted to the **model registry**.

---

## Quickstart

```bash
# 1. Python env — pick ONE:
make setup                      # stock: python3 -m venv + pip
make setup-uv                   # faster: uv (https://docs.astral.sh/uv/) — same .venv result

# 2. Credentials (ENTSO-E token is free: https://transparency.entsoe.eu/ → account → API)
cp .env.example .env            # then fill ENTSOE_API_TOKEN

# 3. Pull data + build features + run the backtest locally
make backtest                   # -> reports/, and an MLflow run under ./mlruns

# 4. Inspect runs
mlflow ui                       # http://localhost:5000
```

> **Using uv?** It's an optional, much faster drop-in for step 1. Install it once with
> `brew install uv` (macOS) or `curl -LsSf https://astral.sh/uv/install.sh | sh`, then run
> `make setup-uv` instead of `make setup`. Both create the same `.venv`, so every later step
> (`make test`, `make backtest`) is identical.
>
> **macOS gotcha:** LightGBM needs the OpenMP runtime. If `make backtest` fails with
> `Library not loaded: @rpath/libomp.dylib`, run `brew install libomp` once. (`make test`
> passes without it — the maths/feature tests don't load LightGBM.)

On **Databricks** (Community Edition is free): import `notebooks/01_databricks_walkthrough.py`
as a notebook — it demonstrates the Spark join, Feature Store registration, and MLflow logging
end-to-end.

---

## Project structure

```
nl-load-forecast/
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .env.example
├── conf/
│   └── config.yaml               # data sources, feature + backtest params
├── src/nl_load_forecast/
│   ├── config.py                 # typed config loader
│   ├── data/
│   │   ├── entsoe.py             # actual NL load + generation
│   │   └── weather.py            # Open-Meteo historical weather (NWP proxy)
│   ├── features/
│   │   └── build.py             # join, calendar, lags, rolling features
│   ├── models/
│   │   └── quantile_lgbm.py     # one LGBM per quantile, monotone-sorted
│   ├── evaluation/
│   │   └── metrics.py           # pinball, coverage, interval width, CRPS
│   ├── backtest/
│   │   └── rolling.py           # walk-forward rolling-origin backtest
│   └── pipeline.py              # end-to-end, logs to MLflow
├── notebooks/
│   └── 01_databricks_walkthrough.py
├── scripts/
│   └── run_backtest.py
└── tests/
    ├── test_metrics.py           # pinball/coverage/CRPS correctness
    └── test_features.py
```

---

## What this demonstrates (skills → evidence)

| Claim | Where it's evidenced |
|---|---|
| Probabilistic / quantile forecasting | `models/quantile_lgbm.py`, `evaluation/metrics.py` |
| Time-series rigour (no leakage) | `backtest/rolling.py` walk-forward + horizon-shifted features in `features/build.py`, guarded by `test_rolling_features_are_horizon_safe` |
| Weather/NWP-driven modelling | `data/weather.py`; derived degree-hours + wind-chill interaction in `features/build.py`; feature importances in the run |
| Feature-selection judgment | `wind_speed³` deliberately omitted — a no-op for tree splits (see `_weather_derived_features`) |
| Databricks + Spark | `notebooks/01_databricks_walkthrough.py` |
| MLflow tracking + registry | `pipeline.py` |
| Feature Store | notebook (Databricks Feature Engineering) |
| Calibration as the metric that matters | reports + Results table |

---

## Roadmap

- [ ] MVP: ENTSO-E + Open-Meteo → one quantile model → rolling backtest → MLflow (the weekend build)
- [ ] Multi-horizon (1–24h) with horizon-specific calibration
- [ ] Databricks Feature Store registration + point-in-time lookup
- [ ] Drift check on incoming weather-feature distributions
- [ ] (Stretch) Serve the registered model behind FastAPI on AWS free-tier; artifacts in S3

---

## Honesty note

This is a personal learning project on public data, not production energy trading infrastructure.
Every result in this README is reproducible from `scripts/run_backtest.py`; if a number or a feature
isn't in the repo yet, it isn't claimed here.
