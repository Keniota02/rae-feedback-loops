"""Selectors rank the athletes in the candidate pool at a given stage.

New selectors (machine-learning models, mitigations) only need to implement ``scores``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from rae.athletes import Athletes


class Selector(Protocol):
    """Returns one score per athlete in ``pool``; the highest scores are selected."""

    def scores(
        self,
        *,
        stage: int,
        pool: np.ndarray,
        athletes: Athletes,
        observed: np.ndarray,
    ) -> np.ndarray:
        """``pool`` holds athlete indices; ``observed`` is aligned with ``pool``."""
        ...


class OracleSelector:
    """Ranks by latent talent. Not implementable in practice: it is the benchmark."""

    def scores(
        self,
        *,
        stage: int,
        pool: np.ndarray,
        athletes: Athletes,
        observed: np.ndarray,
    ) -> np.ndarray:
        return athletes.talent[pool]


class PerformanceSelector:
    """Human-like baseline: ranks by observed performance.

    Observed performance is contaminated by maturation and by accumulated exposure,
    which is how the RAE enters the selection.
    """

    def scores(
        self,
        *,
        stage: int,
        pool: np.ndarray,
        athletes: Athletes,
        observed: np.ndarray,
    ) -> np.ndarray:
        return observed
