# Relative Age Effect and Feedback Loops in Youth Talent Selection

Simulation study (course project, *Ethics in AI*, University of Bologna) of how annual
age-grouping biases multi-stage youth talent selection, how machine-learning selectors may
amplify the bias through a training-exposure feedback loop, and which mitigations reduce the
loss of talent at the lowest cost in efficiency.

Latent talent is known by construction, so the harm of biased selection (talented athletes
born late in the year who are excluded) can be measured directly. No personal data are used.
The simulator explores mechanisms; it does not predict real-world outcomes.

The design choices, model specification and results are tracked in [`report/report.md`](report/report.md).

## Quick start

Requires [conda](https://docs.conda.io/). From the repository root:

```bash
conda env create -f environment.yml
conda activate rae
make install        # installs the package in editable mode with dev tools
make test           # runs the test suite
make lint           # runs ruff
make experiments    # runs the baseline experiment (writes to results/)
```

Without `make` (e.g., on Windows), run the underlying commands directly:
`pip install -e ".[dev]"`, `pytest`, `ruff check src tests experiments`,
`python experiments/run_baseline.py`.

## Repository layout

```
src/rae/          simulator: config, athletes, selectors, pipeline, metrics
tests/            unit tests and model invariants
experiments/      experiment scripts (results go to results/, ignored by git)
report/           project report (Markdown), extended during development
```

## Workflow

- Model-first and test-driven: invariants are written before experiments.
- GitLab-flow-style branching: short-lived feature branches (`feat/...`), pull requests into `main`,
  and `main` always holds the latest working version.
- CI runs lint and tests on every push and pull request.
