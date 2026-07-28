# NL Day-Ahead Load Forecasting — Probabilistic, Weather-Driven

[![CI](https://github.com/dpranantha/nl-load-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/dpranantha/nl-load-forecast/actions/workflows/ci.yml)

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

**Rolling-origin backtest** — walk-forward, 24h day-ahead horizon, 12 folds over 2023–2024
(NL load ≈ 10–18 GW, so the P50 MAE below is ≈ 2–3% of load). Reproduce with `make backtest`;
every value is printed by the run and logged to MLflow.

| Metric | Printed key | Raw quantiles | **+ CQR calibration** | Reads as |
|---|---|---|---|---|
| Pinball loss (mean P10/P50/P90) | `mean_pinball` | 101.5 | **93.7** | lower = sharper + better calibrated |
| P50 MAE | `p50_mae` | 292.2 MW | **292.2 MW** | point accuracy (unchanged by CQR) |
| 80% interval coverage (P10–P90) | `interval_coverage` | 0.49 ❌ | **0.87** ✓ | target ≈ 0.80 → calibrated |
| Mean 80% interval width | `interval_width` | 554 MW | 1133 MW | sharpness (the honest width for real coverage) |
| CRPS | `crps` | 203.0 | **187.4** | overall probabilistic score |

The raw quantile model is badly **over-confident** — its nominal 80% interval covers only 49% of
actuals. Conformalized Quantile Regression (see below) restores coverage to a slightly-conservative
0.87 *without touching point accuracy*, and improves both pinball loss and CRPS. The calibration
plot (`reports/calibration.png`) shows the P10/P90 endpoints snapping onto the diagonal after CQR;
the P50 point stays near 0.41, i.e. the median is still biased slightly high — a known, documented
limitation (CQR calibrates the *interval*, not the point — see Roadmap).

**Baselines it must beat:** (1) seasonal naïve (load 168h ago), (2) P50-only GBM. If the quantile
model doesn't beat seasonal naïve on pinball loss, something is wrong — that's the honesty check.

### Calibration: Conformalized Quantile Regression (CQR)

Plain quantile regression is not guaranteed to be calibrated, and here it wasn't: the models are
under-dispersed and the NL load *level* drifts between the training window and the forecast day, so
intervals fit on history come out too narrow for the future. **CQR** ([Romano, Patterson & Candès,
NeurIPS 2019](https://arxiv.org/abs/1905.03222)) is a distribution-free wrapper that fixes this
without retraining: on a held-out calibration slice it measures how far actuals fell outside the
predicted interval, then shifts the P10/P90 bounds outward by a single margin. On exchangeable data
this gives a finite-sample coverage guarantee of *at least* the nominal level.

One deployment-minded detail (`src/nl_load_forecast/backtest/rolling.py`): naïve split-conformal
holds the most recent days out of training, but on a drifting load series those are the *most
valuable* days — doing so wrecked P50 MAE (292 → 454 MW in testing). So the backtest estimates the
conformal margin from a recent split model, but **fits the deployed model on the full window through
the origin** (no recency gap). Point accuracy is preserved; only the interval is widened. Toggle it
with `backtest.conformalize` in `conf/config.yaml` to reproduce the raw column above.

The CQR helpers are pure and unit-tested on synthetic data (`tests/test_conformal.py`): the tests
pin that a too-narrow interval is widened back to ~80% coverage and that a too-wide one is tightened.

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
make backtest                   # -> reports/, metrics printed, MLflow run + registered model

# 4. Inspect runs (SQLite backend — see note below)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

> **MLflow backend:** runs are logged to a local **SQLite** store (`sqlite:///mlflow.db`, set in
> `conf/config.yaml`). Recent MLflow put the bare-filesystem store into maintenance mode *and* it
> can't host the model registry, so `registered_model_name` needs a real backend — SQLite is the
> zero-setup one. On Databricks the pipeline auto-detects the runtime and uses the platform's
> managed tracking instead.

### Viewing runs in the MLflow UI

Every `make backtest` logs one run — params, metrics, and the calibration plot — to `mlflow.db`.
Browse them locally:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

Open http://localhost:5000 → experiment **`nl-load-forecast`** → the latest run. There you'll find:

- **Metrics:** `interval_coverage`, `mean_pinball`, `crps`, `p50_mae`, `interval_width`.
- **Params:** quantiles, LGBM hyper-params, `conformalize`, `calibration_days`.
- **Artifacts:** `calibration.png` (P10/P50/P90 reliability).
- **Models:** the P50 model registered as `nl_load_quantile_lgbm` (see the *Models* tab).

> **Requires Python 3.12.** The MLflow UI server does not run on Python 3.14 — its FastAPI server
> imports `importlib.abc.Traversable`, which was removed in 3.14. `make setup` pins the venv to
> 3.12 (see the [Quickstart](#quickstart) note), so the command above works out of the box; if you
> built the env another way, run it from a 3.12 interpreter. Tracking/logging during `make backtest`
> is unaffected by this — only the UI server is.

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
│   │   ├── rolling.py           # walk-forward rolling-origin backtest
│   │   └── conformal.py         # CQR interval calibration (pure, unit-tested)
│   └── pipeline.py              # end-to-end, logs to MLflow
├── notebooks/
│   └── 01_databricks_walkthrough.py
├── scripts/
│   └── run_backtest.py
└── tests/
    ├── test_metrics.py           # pinball/coverage/CRPS correctness
    ├── test_features.py          # build_features, calendar, lags, leakage guard
    ├── test_conformal.py         # CQR: widen-when-narrow, tighten-when-wide, ~nominal coverage
    ├── test_weather_api.py       # Open-Meteo unit tests (mocked HTTP, caching)
    └── integration/
        ├── test_weather_api_integration.py  # live Open-Meteo archive + forecast endpoints
        └── test_entsoe_api_integration.py   # live ENTSO-E query_load (skips without token)
```

---

## Testing

**23 unit tests** (pure maths + synthetic data, no tokens or network):

```bash
make test                       # or: pytest -q
```

Tests cover metrics (pinball, coverage, CRPS), feature engineering (lags, rolling, momentum,
weather-derived, calendar/holiday), the horizon-safe leakage guard, the CQR calibration helpers
(widen-when-narrow, tighten-when-wide, ~nominal coverage on synthetic data), and the Open-Meteo
weather client (response parsing, parquet caching, HTTP error propagation — all mocked).

**3 integration tests** hit live external APIs — Open-Meteo (archive + forecast) and ENTSO-E
`query_load` — and are excluded from CI by default (the ENTSO-E one skips cleanly without a token):

```bash
pytest -m integration -v        # needs network (+ ENTSOE_API_TOKEN for the load test); not in CI
```

CI (`-m "not integration"`) runs lint + the 23 unit tests on every push/PR — no secrets or
native libs required.

---

## What this demonstrates (skills → evidence)

| Claim | Where it's evidenced |
|---|---|
| Probabilistic / quantile forecasting | `models/quantile_lgbm.py`, `evaluation/metrics.py` |
| Interval **calibration** (conformal prediction) | `backtest/conformal.py` (CQR), wired into `backtest/rolling.py`, unit-tested in `test_conformal.py`; before/after in Results |
| Time-series rigour (no leakage) | `backtest/rolling.py` walk-forward + horizon-shifted features in `features/build.py`, guarded by `test_rolling_features_are_horizon_safe` |
| Weather/NWP-driven modelling | `data/weather.py` (archive + forecast endpoints, parquet cache); derived degree-hours + wind-chill interaction in `features/build.py`; unit tests (`test_weather_api.py`) + live integration tests (`integration/test_weather_api_integration.py`) |
| Feature-selection judgment | `wind_speed³` deliberately omitted — a no-op for tree splits (see `_weather_derived_features`) |
| Databricks + Spark | `notebooks/01_databricks_walkthrough.py` |
| MLflow tracking + registry | `pipeline.py` |
| Calendar-aware features (NL holidays, cyclical encoding) | `_calendar_features` in `features/build.py`; tested against known Dutch holidays (Koningsdag, Kerstdag) in `test_features.py` |
| Feature Store | notebook (Databricks Feature Engineering) |
| Calibration as the metric that matters | Results table (raw 0.49 → CQR 0.87 coverage) + `reports/calibration.png` |
| Testing discipline (unit + integration, CI) | 18 unit tests (mocked, synthetic), 2 live integration tests, CI on every push with `pytest -m "not integration"` |

---

## Roadmap

- [x] MVP: ENTSO-E + Open-Meteo → quantile model → rolling backtest → MLflow (tracking + registry)
- [x] Interval calibration via CQR (coverage 0.49 → 0.87 with no loss of point accuracy)
- [ ] **Asymmetric CQR + median de-bias** — the current symmetric margin slightly over-covers
      (0.87 vs 0.80) because the raw miscoverage is asymmetric, and CQR leaves the P50 bias
      (empirical 0.41) untouched. Per-side margins + a calibration-set median shift would fix both.
- [ ] Multi-horizon (1–24h) with horizon-specific calibration
- [ ] Databricks Feature Store registration + point-in-time lookup
- [ ] Drift check on incoming weather-feature distributions
- [ ] (Stretch) Serve the registered model behind FastAPI on AWS free-tier; artifacts in S3

---

## Honesty note

This is a personal learning project on public data, not production energy trading infrastructure.
Every result in this README is reproducible from `scripts/run_backtest.py`; if a number or a feature
isn't in the repo yet, it isn't claimed here.
