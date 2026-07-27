.PHONY: setup setup-uv test lint backtest clean

VENV := .venv
PY := $(VENV)/bin/python

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	@echo "Done. Activate with: source $(VENV)/bin/activate"

# Alternative setup using uv (https://docs.astral.sh/uv/) — much faster.
# `uv venv` creates the same .venv, so test/lint/backtest work unchanged afterwards.
setup-uv:
	uv venv --clear $(VENV)
	uv pip install --python $(PY) -e ".[dev]"
	@echo "Done. Activate with: source $(VENV)/bin/activate"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

backtest:
	$(PY) scripts/run_backtest.py --config conf/config.yaml

clean:
	rm -rf mlruns reports/*.png __pycache__ .pytest_cache
