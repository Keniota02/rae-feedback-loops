"""Multi-stage selection pipeline.

At every stage the whole pool of athletes still in the system is observed and the top
fraction (relative to the initial cohort) is selected. Selected athletes accumulate
training exposure, which raises their future observed performance (feedback loop).
Unselected athletes leave the system with probability ``dropout_unselected``; those who stay
can be selected again at a later stage, and it is between them and the selected athletes
that the exposure gap opens.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from rae.athletes import Athletes, generate_athletes
from rae.config import SimConfig
from rae.selectors import Selector, StageContext, choose


@dataclass(frozen=True)
class StageRecord:
    """Log of one stage. Arrays are aligned with ``pool`` (indices of athletes in the pool)."""

    stage: int
    pool: np.ndarray
    observed: np.ndarray  # observed performance at this stage
    times_selected_before: np.ndarray  # number of earlier stages at which the athlete was selected
    selected: np.ndarray  # bool: selected at this stage
    history: np.ndarray  # observed performance at stages 0..stage, shape (stage + 1, pool.size)


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of one run. ``selected`` is the mask of athletes selected at the final stage."""

    athletes: Athletes
    selected: np.ndarray
    times_selected: np.ndarray
    config: SimConfig
    stages: tuple[StageRecord, ...] = field(default=())


def simulate(cfg: SimConfig, selector: Selector) -> SimulationResult:
    """Run one simulation.

    Athletes, noise, dropout and selector draws come from separate streams and are drawn for
    the whole cohort, so two selectors run with the same seed face exactly the same athletes
    and the same random numbers (common random numbers for paired comparisons).
    """
    ss_athletes, ss_noise, ss_dropout, ss_selector = np.random.SeedSequence(cfg.seed).spawn(4)
    n, n_stages = cfg.n_athletes, cfg.n_stages

    athletes = generate_athletes(n, cfg.maturation_corr, np.random.default_rng(ss_athletes))
    if cfg.adult_maturation_weight:
        potential = athletes.talent + cfg.adult_maturation_weight * athletes.maturation
        athletes = replace(athletes, potential=potential)
    noise = np.random.default_rng(ss_noise).standard_normal((n_stages, n))
    dropout_u = np.random.default_rng(ss_dropout).random((n_stages, n))
    selector_rng = np.random.default_rng(ss_selector)

    in_pool = np.ones(n, dtype=bool)
    times_selected = np.zeros(n, dtype=int)
    selected = np.zeros(n, dtype=bool)
    observed_log = np.full((n_stages, n), np.nan)
    records: list[StageRecord] = []

    for stage, fraction in enumerate(cfg.stage_fractions):
        pool = np.flatnonzero(in_pool)
        k = round(fraction * n)
        if k > pool.size:
            raise ValueError(f"stage {stage}: cannot select {k} from a pool of {pool.size}")

        exposure = cfg.exposure_gain * times_selected[pool]
        maturation_weight = cfg.maturation_weight * cfg.maturation_decay**stage
        observed_log[stage, pool] = (
            cfg.talent_weight * athletes.talent[pool]
            + maturation_weight * athletes.maturation[pool]
            + cfg.exposure_weight * exposure
            + cfg.noise_sd * noise[stage, pool]
        )
        history = observed_log[: stage + 1, pool]

        before = times_selected[pool].copy()
        ctx = StageContext(
            stage=stage,
            n_stages=n_stages,
            pool=pool,
            athletes=athletes,
            history=history,
            times_selected=before,
            rng=selector_rng,
        )
        top = choose(selector, ctx, k)

        selected = np.zeros(n, dtype=bool)
        selected[pool[top]] = True
        times_selected[selected] += 1
        records.append(
            StageRecord(
                stage=stage,
                pool=pool,
                observed=history[-1],
                times_selected_before=before,
                selected=selected[pool],
                history=history,
            )
        )

        unselected = pool[~selected[pool]]
        leaves = dropout_u[stage, unselected] < cfg.dropout_unselected
        in_pool[unselected[leaves]] = False

    return SimulationResult(
        athletes=athletes,
        selected=selected,
        times_selected=times_selected,
        config=cfg,
        stages=tuple(records),
    )
