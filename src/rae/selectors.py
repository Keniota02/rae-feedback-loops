"""Selectors rank the athletes in the candidate pool at a given stage.

The pipeline hands every selector a :class:`StageContext`; new selectors (machine-learning
models, mitigations) only need to implement ``scores(ctx)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rae.athletes import Athletes


@dataclass(frozen=True)
class StageContext:
    """What a selector may look at when ranking the pool at one stage.

    Arrays are aligned with ``pool`` (indices of the athletes still in the system). The pool
    only shrinks from one stage to the next, so every athlete in it has been observed at every
    earlier stage and ``history`` has no gaps.
    """

    stage: int
    pool: np.ndarray
    athletes: Athletes
    history: np.ndarray  # observed performance, shape (stage + 1, pool.size); last row = now
    times_selected: np.ndarray  # number of earlier stages at which each athlete was selected
    rng: np.random.Generator  # selector-specific random stream (e.g., scout judgement noise)

    @property
    def observed(self) -> np.ndarray:
        """Observed performance at the current stage."""
        return self.history[-1]


class Selector(Protocol):
    """Returns one score per athlete in ``ctx.pool``; the highest scores are selected."""

    def scores(self, ctx: StageContext) -> np.ndarray: ...


class OracleSelector:
    """Ranks by latent talent. Not implementable in practice: it is the benchmark."""

    def scores(self, ctx: StageContext) -> np.ndarray:
        return ctx.athletes.talent[ctx.pool]


class PerformanceSelector:
    """Memoryless baseline: ranks by the performance observed at the current stage only.

    Observed performance is contaminated by maturation and by accumulated exposure,
    which is how the RAE enters the selection.
    """

    def scores(self, ctx: StageContext) -> np.ndarray:
        return ctx.observed


@dataclass(frozen=True)
class ScoutSelector:
    """Human-like scout: remembers past observations, judges with noise, may favour maturity.

    * ``memory``: a measurement taken ``j`` stages ago has weight ``memory**j``; 0 keeps only
      the current stage (equivalent to :class:`PerformanceSelector`), 1 averages all stages.
    * ``noise_sd``: sd of a fresh subjective judgement error at every stage.
    * ``maturation_bias``: extra weight on visible physical maturity, beyond its effect on
      performance (scouts favouring athletes who look physically developed).
    """

    memory: float = 1.0
    noise_sd: float = 0.0
    maturation_bias: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.memory <= 1.0:
            raise ValueError("memory must be in [0, 1]")
        if self.noise_sd < 0.0:
            raise ValueError("noise_sd must be non-negative")

    def scores(self, ctx: StageContext) -> np.ndarray:
        lags = np.arange(ctx.stage, -1, -1, dtype=float)  # history rows 0..stage
        weights = self.memory**lags  # 0.0**0 == 1: the current stage always counts
        score = weights @ ctx.history / weights.sum()
        if self.maturation_bias:
            score = score + self.maturation_bias * ctx.athletes.maturation[ctx.pool]
        if self.noise_sd:
            # Drawn for the whole cohort and then indexed, so the noise of an athlete does
            # not depend on who else is in the pool.
            score = score + self.noise_sd * ctx.rng.standard_normal(len(ctx.athletes))[ctx.pool]
        return score
