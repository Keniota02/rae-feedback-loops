from dataclasses import replace

import numpy as np

from rae import OracleSelector, PerformanceSelector, SimConfig, simulate


def test_same_seed_is_reproducible_and_different_seeds_differ():
    cfg = SimConfig(n_athletes=5_000, seed=7)
    a = simulate(cfg, PerformanceSelector())
    b = simulate(cfg, PerformanceSelector())
    c = simulate(replace(cfg, seed=8), PerformanceSelector())
    assert np.array_equal(a.selected, b.selected)
    assert not np.array_equal(a.selected, c.selected)


def test_final_stage_selects_the_configured_number_of_athletes():
    cfg = SimConfig(n_athletes=10_000, stage_fractions=(0.4, 0.25, 0.1))
    res = simulate(cfg, PerformanceSelector())
    assert res.selected.sum() == 1_000


def test_selectors_face_the_same_athletes_with_the_same_seed():
    cfg = SimConfig(n_athletes=5_000, seed=3)
    oracle = simulate(cfg, OracleSelector())
    human = simulate(cfg, PerformanceSelector())
    assert np.array_equal(oracle.athletes.talent, human.athletes.talent)
    assert np.array_equal(oracle.athletes.birth_frac, human.athletes.birth_frac)


def test_selected_athletes_were_selected_at_every_stage_they_accumulated_exposure():
    cfg = SimConfig(n_athletes=5_000, dropout_unselected=1.0)  # pure pyramid
    res = simulate(cfg, PerformanceSelector())
    assert np.all(res.times_selected[res.selected] == cfg.n_stages)
