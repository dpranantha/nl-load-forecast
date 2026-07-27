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
# MAGIC %pip install entsoe-py lightgbm holidays python-dotenv

# COMMAND ----------
import pandas as pd
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
# MAGIC %md ## Backtest + MLflow logging
# MAGIC Reuses the packaged pipeline so the metrics match the local run exactly.

# COMMAND ----------
from nl_load_forecast.pipeline import run

scores = run("../conf/config.yaml")
scores
