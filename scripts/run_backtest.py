#!/usr/bin/env python
"""CLI entry point for the rolling-origin backtest."""
from __future__ import annotations

import argparse

from nl_load_forecast.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="NL load forecasting backtest")
    parser.add_argument("--config", default="conf/config.yaml", help="path to config YAML")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
