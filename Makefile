# Run these targets with the conda environment "rae" activated.
.PHONY: install test lint format experiments calibration baseline rq1 rq2 ablation rq3 sensitivity figures clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests experiments
	ruff format --check src tests experiments

format:
	ruff format src tests experiments

experiments: calibration baseline rq1 rq2 ablation rq3 sensitivity

calibration:
	python experiments/run_calibration.py

baseline:
	python experiments/run_baseline.py

rq1:
	python experiments/run_rq1.py

rq2:
	python experiments/run_rq2.py

ablation:
	python experiments/run_ablation.py

rq3:
	python experiments/run_rq3.py

sensitivity:
	python experiments/run_sensitivity.py

figures:
	python experiments/make_figures.py

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', '.ruff_cache', 'build', 'dist', 'results')]"
