"""Invariants of the model (written before the experiments, as in test-driven development)."""

from dataclasses import replace

import numpy as np
import pytest

from rae import OracleSelector, PerformanceSelector, ScoutSelector, SimConfig, simulate
from rae.metrics import (
    overall_talent_loss,
    regret,
    selection_rate_ratio,
    talent_loss_by_quarter,
)

BIG = SimConfig(n_athletes=100_000, seed=11)
NO_BIAS = replace(BIG, maturation_weight=0.0, exposure_weight=0.0)


def test_oracle_never_loses_high_talent_athletes():
    """If the final stage selects at least the high-talent share, the oracle loses nobody."""
    for cfg in (BIG, NO_BIAS, replace(BIG, exposure_weight=0.8, dropout_unselected=1.0)):
        res = simulate(cfg, OracleSelector())
        assert overall_talent_loss(res) == 0.0
        assert np.all(talent_loss_by_quarter(res) == 0.0)


def test_no_rae_without_maturation_and_exposure():
    """No maturation effect and no exposure effect: birth date must not matter."""
    res = simulate(NO_BIAS, PerformanceSelector())
    assert selection_rate_ratio(res) == pytest.approx(1.0, abs=0.08)


def test_oracle_shows_no_rae_even_with_biased_observations():
    res = simulate(BIG, OracleSelector())
    assert selection_rate_ratio(res) == pytest.approx(1.0, abs=0.08)


def test_rae_appears_with_maturation():
    res = simulate(BIG, PerformanceSelector())
    assert selection_rate_ratio(res) > 1.3


def test_stronger_maturation_gives_stronger_rae():
    weak = simulate(replace(BIG, maturation_weight=0.5), PerformanceSelector())
    strong = simulate(replace(BIG, maturation_weight=1.5), PerformanceSelector())
    assert selection_rate_ratio(strong) > selection_rate_ratio(weak)


def test_feedback_loop_amplifies_rae():
    """Same athletes, same noise: adding the exposure effect must increase the Q1/Q4 ratio."""
    without = simulate(BIG, PerformanceSelector())
    with_loop = simulate(replace(BIG, exposure_weight=0.5), PerformanceSelector())
    assert selection_rate_ratio(with_loop) > selection_rate_ratio(without)


def test_biased_selection_loses_more_talent_than_oracle():
    human = simulate(BIG, PerformanceSelector())
    assert overall_talent_loss(human) > 0.0


def test_no_rae_for_an_unbiased_scout_without_maturation_and_exposure():
    """Memory and judgement noise alone must not create a RAE."""
    res = simulate(NO_BIAS, ScoutSelector(memory=1.0, noise_sd=0.5))
    assert selection_rate_ratio(res) == pytest.approx(1.0, abs=0.08)


def test_scout_maturation_bias_increases_rae():
    plain = simulate(BIG, ScoutSelector(memory=1.0))
    biased = simulate(BIG, ScoutSelector(memory=1.0, maturation_bias=0.5))
    assert selection_rate_ratio(biased) > selection_rate_ratio(plain)


def test_feedback_loop_increases_talent_loss_for_a_scout_with_memory():
    """RQ2 mechanism. For a selector that already remembers past observations, exposure adds
    no information about talent, only a head start for those selected early: talent loss and
    regret must grow. (For a memoryless selector the count of past selections also acts as a
    memory of past measurements, so the effect on talent loss is confounded; see the report.)
    """
    scout = ScoutSelector(memory=1.0)
    without = simulate(BIG, scout)
    with_loop = simulate(replace(BIG, exposure_weight=1.0), scout)
    assert overall_talent_loss(with_loop) > overall_talent_loss(without)
    assert regret(with_loop) > regret(without)
    assert selection_rate_ratio(with_loop) > selection_rate_ratio(without)


def test_persistent_maturation_changes_the_target_not_the_athletes():
    """With kappa > 0 the target is T + kappa * M: the oracle still loses nobody, but now it
    selects early-born athletes more often, because part of their advantage is real."""
    persistent = replace(BIG, adult_maturation_weight=0.5)
    oracle = simulate(persistent, OracleSelector())
    assert overall_talent_loss(oracle) == 0.0
    assert selection_rate_ratio(oracle) > 1.1
    base = simulate(BIG, OracleSelector())
    assert np.array_equal(base.athletes.talent, oracle.athletes.talent)
    assert base.athletes.potential is None
    assert np.allclose(
        oracle.athletes.potential, oracle.athletes.talent + 0.5 * oracle.athletes.maturation
    )
