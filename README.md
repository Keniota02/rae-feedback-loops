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
make experiments    # runs baseline, RQ1, RQ2, ablation and RQ3 (writes CSV files to results/)
make figures        # draws the report figures from results/ into report/figures/
```

Single experiments: `make baseline`, `make rq1`, `make rq2`, `make ablation`, `make rq3`. The scripts accept options, e.g.
`python experiments/run_rq1.py --seeds 20 --model gbm` (see `--help`).

Without `make` (e.g., on Windows), run the underlying commands directly:
`pip install -e ".[dev]"`, `pytest`, `ruff check src tests experiments`,
`python experiments/run_rq1.py`.

## Repository layout

```
src/rae/          simulator: config, athletes, selectors (oracle, scouts), pipeline,
                  learning (ML selectors), generations (retraining loop), mitigations,
                  metrics, stats
tests/            unit tests and model invariants
experiments/      experiment scripts and shared settings (results go to results/, ignored by git)
report/           project report (Markdown), extended during development, and its figures
```

## Workflow

- Model-first and test-driven: invariants are written before experiments.
- GitLab flow: every change is developed on a short-lived branch (e.g., `feat/scout-selector`,
  `fix/...`) and merged into `main` through a merge request once CI is green; `main` always holds
  the latest working version. Commit messages describe the change.
- CI (`.gitlab-ci.yml` on GitLab; `.github/workflows/ci.yml` mirrors it on GitHub) runs lint and
  tests on every push to `main` and on every merge request.
