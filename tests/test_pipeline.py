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


def test_stage_logs_are_consistent():
    cfg = SimConfig(n_athletes=5_000, stage_fractions=(0.4, 0.25, 0.1))
    res = simulate(cfg, PerformanceSelector())
    assert len(res.stages) == cfg.n_stages
    for record, fraction in zip(res.stages, cfg.stage_fractions, strict=True):
        assert record.selected.sum() == round(fraction * cfg.n_athletes)
        assert record.pool.shape == record.observed.shape == record.selected.shape
        assert record.pool.shape == record.times_selected_before.shape
    for current, following in zip(res.stages, res.stages[1:], strict=False):
        assert set(following.pool) <= set(current.pool)
    last = res.stages[-1]
    assert np.array_equal(last.pool[last.selected], np.flatnonzero(res.selected))


def test_history_is_complete_and_consistent_with_earlier_stages():
    cfg = SimConfig(n_athletes=5_000, exposure_weight=0.5)
    res = simulate(cfg, PerformanceSelector())
    for record in res.stages:
        assert record.history.shape == (record.stage + 1, record.pool.size)
        assert not np.isnan(record.history).any()
        assert np.array_equal(record.history[-1], record.observed)
        for earlier in res.stages[: record.stage]:
            position = np.searchsorted(earlier.pool, record.pool)
            assert np.array_equal(record.history[earlier.stage], earlier.observed[position])
