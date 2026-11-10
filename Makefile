VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: dev dev-all test lint format check complexity notebooks docs clean

$(PY):
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

dev: $(PY)  ## create .venv and install lipipy with dev tools
	$(PY) -m pip install -e ".[dev]"

dev-all: $(PY)  ## also install PyTorch and docs tooling
	$(PY) -m pip install -e ".[dev,dl,docs]"

test:  ## run the test suite with coverage
	$(PY) -m pytest --cov=lipipy --cov-report=term-missing

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

check: lint test  ## everything CI runs

complexity:  ## timing-based complexity checks (machine dependent)
	$(PY) -m pytest -m complexity

notebooks:  ## execute every notebook to make sure it runs
	$(PY) -m jupyter nbconvert --to notebook --execute --output-dir /tmp/lipipy-nb notebooks/*/*.ipynb

docs:
	$(VENV)/bin/mkdocs serve

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
