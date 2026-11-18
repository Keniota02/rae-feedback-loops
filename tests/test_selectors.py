from dataclasses import replace

import numpy as np
import pytest

from rae import Athletes, PerformanceSelector, ScoutSelector, SimConfig, StageContext, simulate

CFG = SimConfig(n_athletes=5_000, exposure_weight=0.5, seed=3)


def _context(history: np.ndarray, seed: int = 0) -> StageContext:
    n = history.shape[1]
    athletes = Athletes(
        talent=np.zeros(n), birth_frac=np.linspace(0, 0.99, n), maturation=np.arange(n, dtype=float)
    )
    return StageContext(
        stage=history.shape[0] - 1,
        pool=np.arange(n),
        athletes=athletes,
        history=history,
        times_selected=np.zeros(n, dtype=int),
        rng=np.random.default_rng(seed),
    )


def test_scout_without_memory_is_the_performance_selector():
    a = simulate(CFG, ScoutSelector(memory=0.0))
    b = simulate(CFG, PerformanceSelector())
    assert np.array_equal(a.selected, b.selected)


def test_scout_weights_past_stages_geometrically():
    history = np.array([[4.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    assert np.allclose(ScoutSelector(memory=1.0).scores(_context(history)), [2.0, 4.0 / 3.0])
    # weights 0.25, 0.5, 1 for lags 2, 1, 0
    expected = np.array([0.25 * 4 + 0 + 2.0, 0 + 0.5 * 2 + 2.0]) / 1.75
    assert np.allclose(ScoutSelector(memory=0.5).scores(_context(history)), expected)


def test_scout_maturation_bias_adds_visible_maturity():
    history = np.zeros((1, 3))
    scores = ScoutSelector(maturation_bias=2.0).scores(_context(history))
    assert np.allclose(scores, [0.0, 2.0, 4.0])


def test_scout_noise_is_reproducible_and_seed_dependent():
    scout = ScoutSelector(noise_sd=1.0)
    a = simulate(CFG, scout)
    b = simulate(CFG, scout)
    c = simulate(replace(CFG, seed=4), scout)
    assert np.array_equal(a.selected, b.selected)
    assert not np.array_equal(a.selected, c.selected)


def test_scout_noise_changes_the_selection():
    noisy = simulate(CFG, ScoutSelector(noise_sd=1.0))
    clean = simulate(CFG, ScoutSelector(noise_sd=0.0))
    assert not np.array_equal(noisy.selected, clean.selected)


@pytest.mark.parametrize("kwargs", [{"memory": -0.1}, {"memory": 1.1}, {"noise_sd": -1.0}])
def test_invalid_scout_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ScoutSelector(**kwargs)
