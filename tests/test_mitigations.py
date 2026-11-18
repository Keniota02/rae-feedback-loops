from dataclasses import replace

import numpy as np
import pytest

from rae import Athletes, ScoutSelector, SimConfig, StageContext, simulate
from rae.metrics import overall_talent_loss, selection_rate_ratio
from rae.mitigations import (
    BioBanding,
    BirthQuarterQuota,
    RelativeAgeAdjusted,
    RotatingSlots,
    allocate,
    select_by_groups,
)
from rae.selectors import choose

BIG = SimConfig(n_athletes=50_000, seed=21)
SCOUT = ScoutSelector(memory=1.0, noise_sd=0.5)


class _Fixed:
    """Selector with given scores, for unit tests."""

    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=float)

    def scores(self, ctx):
        return self._scores


def _context(n: int = 12, stage: int = 0, n_stages: int = 3, seed: int = 0) -> StageContext:
    rng = np.random.default_rng(seed)
    athletes = Athletes(
        talent=rng.standard_normal(n),
        birth_frac=(np.arange(n) + 0.5) / n,  # evenly spread: n / 4 per quarter
        maturation=-np.arange(n, dtype=float),  # early-born athletes are more mature
    )
    return StageContext(
        stage=stage,
        n_stages=n_stages,
        pool=np.arange(n),
        athletes=athletes,
        history=np.zeros((stage + 1, n)),
        times_selected=np.zeros(n, dtype=int),
        rng=np.random.default_rng(seed),
    )


def test_allocate_is_proportional_and_respects_capacity():
    assert allocate(np.array([5, 5, 5, 5]), 8).tolist() == [2, 2, 2, 2]
    slots = allocate(np.array([10, 3, 1]), 7)
    assert slots.sum() == 7
    assert np.all(slots <= [10, 3, 1])
    assert slots.tolist() == [5, 2, 0]  # exact shares 5.0, 1.5, 0.5: one remainder seat
    with pytest.raises(ValueError):
        allocate(np.array([1, 1]), 3)


def test_select_by_groups_with_full_share_is_balanced():
    scores = np.arange(12, dtype=float)[::-1]  # the first athletes are the best
    groups = np.repeat([0, 1, 2, 3], 3)
    chosen = select_by_groups(scores, groups, k=4, share=1.0)
    assert sorted(groups[chosen].tolist()) == [0, 1, 2, 3]
    unconstrained = select_by_groups(scores, groups, k=4, share=0.0)
    assert sorted(unconstrained.tolist()) == [0, 1, 2, 3]


def test_relative_age_adjustment_removes_the_birth_trend():
    ctx = _context()
    u = ctx.athletes.birth_frac
    base = _Fixed(5.0 - 3.0 * u + np.sin(np.arange(12)))
    adjusted = RelativeAgeAdjusted(base).scores(ctx)
    slope = np.polyfit(u, adjusted, 1)[0]
    assert slope == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(RelativeAgeAdjusted(base, strength=0.0).scores(ctx), base.scores(ctx))


def test_quota_selects_every_quarter_in_proportion():
    ctx = _context()
    best_are_early = _Fixed(-ctx.athletes.birth_frac)
    chosen = choose(BirthQuarterQuota(best_are_early), ctx, 4)
    assert sorted(ctx.athletes.quarter[chosen].tolist()) == [0, 1, 2, 3]


def test_bio_banding_with_exact_maturation_selects_from_every_band():
    ctx = _context()
    best_are_mature = _Fixed(ctx.athletes.maturation)
    chosen = choose(BioBanding(best_are_mature, n_bands=3, estimate_sd=0.0), ctx, 3)
    # Bands of 4 athletes by maturation: positions 8-11, 4-7, 0-3.
    assert sorted((chosen // 4).tolist()) == [0, 1, 2]


def test_rotating_slots_rotate_only_below_the_cut_and_not_at_the_final_stage():
    ctx = _context(n=40)
    base = _Fixed(-np.arange(40.0))  # athlete 0 is the best
    chosen = choose(RotatingSlots(base, share=0.5, window=1.0), ctx, 10)
    assert set(range(5)) <= set(chosen.tolist())  # guaranteed slots
    assert all(5 <= c < 20 for c in chosen if c >= 5)  # rotating slots from the next band
    final = _context(n=40, stage=2)
    assert sorted(choose(RotatingSlots(base, share=0.5), final, 10).tolist()) == list(range(10))


def test_zero_strength_mitigations_reproduce_the_base_selector():
    base = simulate(BIG, SCOUT).selected
    for mitigated in (
        RelativeAgeAdjusted(SCOUT, strength=0.0),
        BirthQuarterQuota(SCOUT, share=0.0),
        RotatingSlots(SCOUT, share=0.0),
    ):
        assert np.array_equal(simulate(BIG, mitigated).selected, base)


@pytest.mark.parametrize(
    "mitigation",
    [
        RelativeAgeAdjusted(SCOUT),
        BirthQuarterQuota(SCOUT),
        BioBanding(SCOUT),
    ],
    ids=["relative-age-adjusted", "quota", "bio-banding"],
)
def test_mitigations_reduce_the_rae_of_the_scout(mitigation):
    base = selection_rate_ratio(simulate(BIG, SCOUT))
    assert selection_rate_ratio(simulate(BIG, mitigation)) < 0.6 * base


def test_rotating_slots_spread_exposure_over_more_athletes():
    cfg = replace(BIG, exposure_weight=1.0)
    base = simulate(cfg, SCOUT)
    rotating = simulate(cfg, RotatingSlots(SCOUT, share=0.3))
    assert (rotating.times_selected > 0).sum() > (base.times_selected > 0).sum()


def test_mitigations_are_reproducible():
    for mitigation in (BioBanding(SCOUT), RotatingSlots(SCOUT)):
        a, b = simulate(BIG, mitigation), simulate(BIG, mitigation)
        assert np.array_equal(a.selected, b.selected)
        assert overall_talent_loss(a) == overall_talent_loss(b)


def test_invalid_select_output_is_rejected():
    class Broken:
        def scores(self, ctx):
            return np.zeros(ctx.pool.size)

        def select(self, ctx, k):
            return np.zeros(k, dtype=int)  # duplicates

    with pytest.raises(ValueError):
        simulate(SimConfig(n_athletes=1_000), Broken())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RelativeAgeAdjusted(SCOUT, strength=-1.0),
        lambda: BirthQuarterQuota(SCOUT, share=1.5),
        lambda: BioBanding(SCOUT, n_bands=0),
        lambda: BioBanding(SCOUT, estimate_sd=-1.0),
        lambda: RotatingSlots(SCOUT, share=-0.1),
        lambda: RotatingSlots(SCOUT, window=-1.0),
    ],
)
def test_invalid_mitigations_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()
