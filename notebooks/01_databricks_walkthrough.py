# Databricks notebook source
# MAGIC %md
# MAGIC # NL Day-Ahead Load Forecasting — Databricks walkthrough
# MAGIC
# MAGIC Demonstrates the parts that are Databricks-specific: a **Spark** join of load + weather,
# MAGIC registration into the **Feature Store**, and **MLflow** tracking. The modelling logic is
# MAGIC reused from the `nl_load_forecast` package so nothing is duplicated.
# MAGIC
# MAGIC Runs on **Databricks Community Edition** (free). Set `ENTSOE_API_TOKEN` as a cluster env var
# MAGIC or a Databricks secret.

# COMMAND ----------
# MAGIC %pip install entsoe-py lightgbm holidays python-dotenv mlflow

# COMMAND ----------
from nl_load_forecast.config import Config
from nl_load_forecast.data.entsoe import fetch_load
from nl_load_forecast.data.weather import fetch_weather

cfg = Config.load("../conf/config.yaml")

load_pdf = fetch_load(cfg.data.country_code, cfg.data.start, cfg.data.end,
                      cfg.data.timezone, cfg.data.cache_dir)
weather_pdf = fetch_weather(cfg.data.weather_latitude, cfg.data.weather_longitude,
                            cfg.data.start, cfg.data.end, cfg.features.weather_vars,
                            cfg.data.timezone, cfg.data.cache_dir)

# COMMAND ----------
# MAGIC %md ## Spark join (load + weather aligned on timestamp)

# COMMAND ----------
load_sdf = spark.createDataFrame(load_pdf.reset_index())
weather_sdf = spark.createDataFrame(weather_pdf.reset_index())

joined = (
    load_sdf.join(weather_sdf, on="timestamp", how="inner")
    .orderBy("timestamp")
)
joined.createOrReplaceTempView("nl_load_weather")
display(joined)

# COMMAND ----------
# MAGIC %md ## Register features in the Feature Store
# MAGIC Point-in-time correct features keyed by timestamp — the same table the online model would read.

# COMMAND ----------
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
fe.create_table(
    name="main.default.nl_load_features",
    primary_keys=["timestamp"],
    df=joined,
    description="Hourly NL load + weather features for day-ahead forecasting",
)

# COMMAND ----------
# MAGIC %md ## (Advanced) Point-in-time lookup with `create_training_set`
# MAGIC The pipeline's `data.feature_table` flag does a simple **snapshot read** of the table above.
# MAGIC The fuller, leakage-safe online/offline pattern instead builds a **training set** from
# MAGIC `FeatureLookup`s joined to a *label* DataFrame on the timestamp key. This records feature
# MAGIC lineage, so the **same** lookups can be replayed at serving time and the online model reads
# MAGIC exactly the features it was trained on.
# MAGIC
# MAGIC This cell is illustrative — it's a roadmap item, not yet wired into `run()`.

# COMMAND ----------
from databricks.feature_engineering import FeatureLookup

# Label DataFrame: the join key + the target only. Features are pulled in via the lookup.
labels_sdf = load_sdf.select("timestamp", cfg.features.target)

lookups = [FeatureLookup(table_name="main.default.nl_load_features",
                         lookup_key="timestamp")]

training_set = fe.create_training_set(
    df=labels_sdf,
    feature_lookups=lookups,
    label=cfg.features.target,
    exclude_columns=["timestamp"],
)
train_pdf = training_set.load_df().toPandas()
display(train_pdf)

# COMMAND ----------
# MAGIC %md ## Backtest + MLflow logging
# MAGIC Reuses the packaged pipeline so the metrics match the local run exactly.
# MAGIC
# MAGIC The pipeline detects the Databricks runtime and logs to the **managed MLflow** tracking
# MAGIC server here (locally it falls back to the SQLite backend from `conf/config.yaml`).
# MAGIC
# MAGIC It also reads features from the **Feature Store** table registered above instead of
# MAGIC re-fetching from the public APIs — but only when `data.feature_table` is set in the config
# MAGIC (and we're on a cluster). Point it at the table before running; otherwise `run()` falls back
# MAGIC to fetching from ENTSO-E/Open-Meteo.

# COMMAND ----------
# To read features from the table registered above, set `data.feature_table` in conf/config.yaml
# to "main.default.nl_load_features" (or edit the loaded Config here before calling run()).
from nl_load_forecast.pipeline import run

scores = run("../conf/config.yaml")
scores
