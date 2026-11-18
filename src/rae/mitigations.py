"""Mitigations of the Relative Age Effect (RQ3).

Each mitigation wraps a base selector (a scout or an ML model) and changes either its scores or
the way the selected athletes are chosen from them:

* :class:`RelativeAgeAdjusted` removes the linear trend of the scores on birth position
  (a corrective adjustment, as with age-in-days corrections of test scores);
* :class:`BirthQuarterQuota` reserves slots to each birth quarter in proportion to its size;
* :class:`BioBanding` compares athletes only with peers of similar (estimated) maturation, by
  reserving slots to maturation bands;
* :class:`RotatingSlots` gives part of the slots of the development stages to athletes drawn at
  random just below the cut, so that training exposure is spread over more athletes.

A ``strength``/``share`` of 0 always gives back the base selector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rae.selectors import Selector, StageContext, choose, top_k


def allocate(capacity: np.ndarray, total: int) -> np.ndarray:
    """Split ``total`` slots among groups in proportion to their size (largest remainder).

    No group receives more slots than it has members.
    """
    capacity = np.asarray(capacity, dtype=int)
    if total > capacity.sum():
        raise ValueError(f"cannot allocate {total} slots to {capacity.sum()} members")
    exact = capacity / capacity.sum() * total
    slots = np.floor(exact).astype(int)
    for g in np.argsort(-(exact - slots), kind="stable"):
        if slots.sum() == total:
            break
        if slots[g] < capacity[g]:
            slots[g] += 1
    return slots


def select_by_groups(scores: np.ndarray, groups: np.ndarray, k: int, share: float) -> np.ndarray:
    """Reserve ``round(share * k)`` slots to groups in proportion to their size, filled by the
    best scores within each group; the remaining slots go to the best of the others."""
    n_reserved = round(share * k)
    labels, counts = np.unique(groups, return_counts=True)
    chosen = []
    for label, slots in zip(labels, allocate(counts, n_reserved), strict=True):
        members = np.flatnonzero(groups == label)
        chosen.append(members[top_k(scores[members], slots)])
    picked = np.concatenate(chosen)
    others = np.setdiff1d(np.arange(scores.size), picked)
    return np.concatenate([picked, others[top_k(scores[others], k - n_reserved)]])


def _check_share(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RelativeAgeAdjusted:
    """Subtract ``strength`` times the within-pool OLS trend of the scores on birth position.

    Birth date is known to the club, so the trend can be estimated from the pool itself;
    ``strength = 1`` removes the linear association between score and birth position.
    """

    base: Selector
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise ValueError("strength must be non-negative")

    def scores(self, ctx: StageContext) -> np.ndarray:
        scores = np.asarray(self.base.scores(ctx), dtype=float)
        u = ctx.athletes.birth_frac[ctx.pool]
        centred = u - u.mean()
        var = centred @ centred
        if var == 0.0 or self.strength == 0.0:
            return scores
        slope = centred @ (scores - scores.mean()) / var
        return scores - self.strength * slope * centred


@dataclass(frozen=True)
class BirthQuarterQuota:
    """Reserve a ``share`` of the slots to birth quarters, in proportion to their pool size."""

    base: Selector
    share: float = 1.0

    def __post_init__(self) -> None:
        _check_share(self.share, "share")

    def scores(self, ctx: StageContext) -> np.ndarray:
        return self.base.scores(ctx)

    def select(self, ctx: StageContext, k: int) -> np.ndarray:
        groups = ctx.athletes.quarter[ctx.pool]
        return select_by_groups(np.asarray(self.scores(ctx)), groups, k, self.share)


@dataclass(frozen=True)
class BioBanding:
    """Reserve a ``share`` of the slots to ``n_bands`` bands of estimated maturation.

    Maturation is not observed directly: it is estimated with an error of sd ``estimate_sd``
    (e.g., predicted adult height), drawn afresh at every stage. Within a band athletes compete
    with peers of similar maturity, so the maturation advantage largely cancels out.
    """

    base: Selector
    n_bands: int = 3
    estimate_sd: float = 0.5
    share: float = 1.0

    def __post_init__(self) -> None:
        _check_share(self.share, "share")
        if self.n_bands < 1:
            raise ValueError("n_bands must be at least 1")
        if self.estimate_sd < 0.0:
            raise ValueError("estimate_sd must be non-negative")

    def scores(self, ctx: StageContext) -> np.ndarray:
        return self.base.scores(ctx)

    def select(self, ctx: StageContext, k: int) -> np.ndarray:
        scores = np.asarray(self.scores(ctx))
        noise = ctx.rng.standard_normal(len(ctx.athletes))[ctx.pool]
        estimate = ctx.athletes.maturation[ctx.pool] + self.estimate_sd * noise
        rank = np.argsort(np.argsort(estimate, kind="stable"), kind="stable")
        bands = rank * self.n_bands // ctx.pool.size
        return select_by_groups(scores, bands, k, self.share)


@dataclass(frozen=True)
class RotatingSlots:
    """Give a ``share`` of the slots to athletes drawn at random just below the cut.

    The candidates are the next ``window * k`` athletes after the guaranteed slots. Rotation
    applies to the development stages; the final selection is left to the base selector unless
    ``at_final_stage`` is set.
    """

    base: Selector
    share: float = 0.2
    window: float = 1.0
    at_final_stage: bool = False

    def __post_init__(self) -> None:
        _check_share(self.share, "share")
        if self.window < 0.0:
            raise ValueError("window must be non-negative")

    def scores(self, ctx: StageContext) -> np.ndarray:
        return self.base.scores(ctx)

    def select(self, ctx: StageContext, k: int) -> np.ndarray:
        if ctx.is_final and not self.at_final_stage:
            return choose(self.base, ctx, k)
        order = top_k(np.asarray(self.scores(ctx)), ctx.pool.size)
        n_rotating = round(self.share * k)
        fixed = order[: k - n_rotating]
        candidates = order[k - n_rotating : k + round(self.window * k)]
        rotating = ctx.rng.choice(candidates, size=n_rotating, replace=False)
        return np.concatenate([fixed, rotating])
