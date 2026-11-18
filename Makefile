# Run these targets with the conda environment "rae" activated.
.PHONY: install test lint format experiments rq1 rq2 clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests experiments

format:
	ruff format src tests experiments

experiments:
	python experiments/run_baseline.py

rq1:
	python experiments/run_rq1.py

rq2:
	python experiments/run_rq2.py

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', '.ruff_cache', 'build', 'dist', 'results')]"
