"""Retraining across successive cohorts (generations).

Generation 0 is the historical data produced by a human-like selector (by default a memoryless
performance-based one; experiments pass a scout).
A model is trained on it and deployed on the next cohorts (generation 1); it is then retrained
on its own decisions and outcomes, and so on. This is the cross-cohort feedback loop of RQ2.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from rae.config import SimConfig
from rae.learning import ModelKind, Regime, train_selector
from rae.metrics import summarize
from rae.pipeline import simulate
from rae.selectors import PerformanceSelector, Selector


def run_generations(
    cfg: SimConfig,
    *,
    regime: Regime,
    use_birth_info: bool,
    model: ModelKind = "linear",
    historical: Selector | None = None,
    n_generations: int = 5,
    cohorts_per_generation: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the retraining loop and return one row of metrics per (generation, cohort).

    ``generation == 0`` holds the historical human-like cohorts. Cohort seeds are derived from
    ``seed`` so that the whole loop is reproducible.
    """

    def cohort_seed(generation: int, k: int) -> int:
        return seed * 1_000_000 + generation * 1_000 + k

    rows: list[dict] = []
    selector = historical if historical is not None else PerformanceSelector()
    for generation in range(n_generations + 1):
        cohorts = [
            simulate(replace(cfg, seed=cohort_seed(generation, k)), selector)
            for k in range(cohorts_per_generation)
        ]
        rows += [
            {"generation": generation, "cohort": k, **summarize(res)}
            for k, res in enumerate(cohorts)
        ]
        if generation < n_generations:
            selector = train_selector(
                cohorts,
                regime=regime,
                use_birth_info=use_birth_info,
                model=model,
                seed=seed,
            )
    return pd.DataFrame(rows)
