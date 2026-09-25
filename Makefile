.PHONY: install run smoke test lint security check

install:
	python -m pip install -e ".[dev]"

run:
	MPLCONFIGDIR=.matplotlib python -m ecommerce_forecasting.cli run --output-dir local-runs/latest

smoke:
	MPLCONFIGDIR=.matplotlib python -m ecommerce_forecasting.cli smoke
	MPLCONFIGDIR=.matplotlib python -m ecommerce_forecasting.cli smoke-supplied

test:
	MPLCONFIGDIR=.matplotlib python -m pytest

lint:
	python -m ruff check .

security:
	python scripts/check_sensitive.py

check: lint test security smoke
