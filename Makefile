.PHONY: setup setup-uv test lint backtest clean

VENV := .venv
PY := $(VENV)/bin/python
# Pin the interpreter: the project targets 3.12 (matches CI). Newer runtimes like 3.14
# aren't ready yet — e.g. MLflow's server imports importlib.abc.Traversable, removed in 3.14.
# Override on the CLI if you have a different supported minor, e.g. `make setup PYTHON=python3.13`.
PYTHON ?= python3.12

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	@echo "Done. Activate with: source $(VENV)/bin/activate"

# Alternative setup using uv (https://docs.astral.sh/uv/) — much faster.
# `uv venv` creates the same .venv, so test/lint/backtest work unchanged afterwards.
setup-uv:
	uv venv --clear --python 3.12 $(VENV)
	uv pip install --python $(PY) -e ".[dev]"
	@echo "Done. Activate with: source $(VENV)/bin/activate"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

backtest:
	$(PY) scripts/run_backtest.py --config conf/config.yaml

clean:
	rm -rf mlruns mlflow.db reports/*.png __pycache__ .pytest_cache
