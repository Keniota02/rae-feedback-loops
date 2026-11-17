"""Multi-stage selection pipeline.

At every stage the whole pool of athletes still in the system is observed and the top
fraction (relative to the initial cohort) is selected. Selected athletes accumulate
training exposure, which raises their future observed performance (feedback loop).
Unselected athletes leave the system with probability ``dropout_unselected``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rae.athletes import Athletes, generate_athletes
from rae.config import SimConfig
from rae.selectors import Selector


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of one run. ``selected`` is the mask of athletes selected at the final stage."""

    athletes: Athletes
    selected: np.ndarray
    times_selected: np.ndarray
    config: SimConfig


def simulate(cfg: SimConfig, selector: Selector) -> SimulationResult:
    """Run one simulation.

    Athletes, noise and dropout draws come from separate streams and are drawn for the whole
    cohort, so two selectors run with the same seed face exactly the same athletes and the
    same random numbers (common random numbers for paired comparisons).
    """
    ss_athletes, ss_noise, ss_dropout = np.random.SeedSequence(cfg.seed).spawn(3)
    n, n_stages = cfg.n_athletes, cfg.n_stages

    athletes = generate_athletes(n, cfg.maturation_corr, np.random.default_rng(ss_athletes))
    noise = np.random.default_rng(ss_noise).standard_normal((n_stages, n))
    dropout_u = np.random.default_rng(ss_dropout).random((n_stages, n))

    in_pool = np.ones(n, dtype=bool)
    times_selected = np.zeros(n, dtype=int)
    selected = np.zeros(n, dtype=bool)

    for stage, fraction in enumerate(cfg.stage_fractions):
        pool = np.flatnonzero(in_pool)
        k = round(fraction * n)
        if k > pool.size:
            raise ValueError(f"stage {stage}: cannot select {k} from a pool of {pool.size}")

        exposure = cfg.exposure_gain * times_selected[pool]
        maturation_weight = cfg.maturation_weight * cfg.maturation_decay**stage
        observed = (
            cfg.talent_weight * athletes.talent[pool]
            + maturation_weight * athletes.maturation[pool]
            + cfg.exposure_weight * exposure
            + cfg.noise_sd * noise[stage, pool]
        )

        scores = selector.scores(stage=stage, pool=pool, athletes=athletes, observed=observed)
        top = np.argsort(-scores, kind="stable")[:k]

        selected = np.zeros(n, dtype=bool)
        selected[pool[top]] = True
        times_selected[selected] += 1

        unselected = pool[~selected[pool]]
        leaves = dropout_u[stage, unselected] < cfg.dropout_unselected
        in_pool[unselected[leaves]] = False

    return SimulationResult(
        athletes=athletes, selected=selected, times_selected=times_selected, config=cfg
    )
