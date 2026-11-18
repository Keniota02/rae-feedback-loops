from dataclasses import replace

import numpy as np
import pytest

from rae import (
    OracleSelector,
    PerformanceSelector,
    ScoutSelector,
    SimConfig,
    SimulationResult,
    build_training_data,
    simulate,
    train_selector,
)
from rae.metrics import selection_rate_ratio

SMALL = SimConfig(n_athletes=4_000, exposure_weight=0.5)


def _history(cfg: SimConfig, n: int = 2) -> list[SimulationResult]:
    return [simulate(replace(cfg, seed=k), PerformanceSelector()) for k in range(n)]


def test_training_data_for_decisions_has_one_row_per_pool_member():
    history = _history(SMALL)
    expected_rows = sum(record.pool.size for res in history for record in res.stages)
    x, y = build_training_data(history, "decisions", use_birth_info=False)
    assert x.shape == (expected_rows, 4)
    assert set(np.unique(y)) <= {0, 1}
    x_birth, _ = build_training_data(history, "decisions", use_birth_info=True)
    assert x_birth.shape == (expected_rows, 5)


def test_training_data_for_later_performance_only_uses_athletes_still_in_the_system():
    history = _history(SMALL, n=1)
    stages = history[0].stages
    expected_rows = sum(
        np.intersect1d(a.pool, b.pool).size for a, b in zip(stages, stages[1:], strict=False)
    )
    x, y = build_training_data(history, "later_performance", use_birth_info=False)
    assert x.shape[0] == y.shape[0] == expected_rows
    assert expected_rows < sum(record.pool.size for record in stages[:-1])  # dropouts lose labels


def test_invalid_inputs_are_rejected():
    history = _history(SMALL, n=1)
    with pytest.raises(ValueError):
        build_training_data(history, "nonsense", use_birth_info=False)  # type: ignore[arg-type]
    without_logs = replace(history[0], stages=())
    with pytest.raises(ValueError):
        build_training_data([without_logs], "decisions", use_birth_info=False)


def test_decision_imitation_reproduces_the_historical_selector():
    cfg = replace(SimConfig(n_athletes=20_000), exposure_weight=0.0)
    selector = train_selector(_history(cfg, n=3), regime="decisions", use_birth_info=False)
    test = replace(cfg, seed=999)
    human = simulate(test, PerformanceSelector())
    imitation = simulate(test, selector)
    overlap = (human.selected & imitation.selected).sum() / human.selected.sum()
    assert overlap > 0.98


def test_decision_imitation_learns_the_memory_of_a_scout():
    """The history-mean feature lets a model imitate a scout who averages all stages."""
    cfg = replace(SimConfig(n_athletes=20_000), exposure_weight=0.0)
    scout = ScoutSelector(memory=1.0)
    history = [simulate(replace(cfg, seed=k), scout) for k in range(3)]
    selector = train_selector(history, regime="decisions", use_birth_info=False)
    test = replace(cfg, seed=999)
    human = simulate(test, scout)
    imitation = simulate(test, selector)
    overlap = (human.selected & imitation.selected).sum() / human.selected.sum()
    assert overlap > 0.95


@pytest.mark.parametrize("regime", ["decisions", "later_performance"])
@pytest.mark.parametrize("use_birth_info", [False, True])
def test_ml_selectors_do_not_invent_bias(regime, use_birth_info):
    """With no maturation and no exposure effect there is no RAE to learn."""
    cfg = SimConfig(n_athletes=50_000, maturation_weight=0.0, exposure_weight=0.0)
    selector = train_selector(_history(cfg, n=3), regime=regime, use_birth_info=use_birth_info)
    res = simulate(replace(cfg, seed=999), selector)
    assert selection_rate_ratio(res) == pytest.approx(1.0, abs=0.12)


def test_birth_information_increases_rae_when_trained_on_later_performance():
    cfg = SimConfig(n_athletes=20_000, exposure_weight=0.5)
    history = _history(cfg, n=3)
    test = replace(cfg, seed=999)
    blind = train_selector(history, regime="later_performance", use_birth_info=False)
    aware = train_selector(history, regime="later_performance", use_birth_info=True)
    assert selection_rate_ratio(simulate(test, aware)) > selection_rate_ratio(simulate(test, blind))


def test_oracle_still_beats_every_ml_selector_on_regret():
    from rae.metrics import regret

    cfg = SimConfig(n_athletes=20_000, exposure_weight=0.5)
    history = _history(cfg, n=2)
    test = replace(cfg, seed=999)
    assert regret(simulate(test, OracleSelector())) == 0.0
    for regime in ("decisions", "later_performance"):
        selector = train_selector(history, regime=regime, use_birth_info=True)
        assert regret(simulate(test, selector)) > 0.0


def test_feature_subsets_select_the_columns():
    history = _history(SMALL, n=1)
    x_all, _ = build_training_data(history, "decisions", use_birth_info=True)
    x_sub, _ = build_training_data(
        history, "decisions", use_birth_info=True, features=("observed", "stage")
    )
    assert x_sub.shape[1] == 3
    assert np.array_equal(x_sub, x_all[:, [0, 3, 4]])  # observed, stage, birth_frac


def test_invalid_feature_subsets_are_rejected():
    history = _history(SMALL, n=1)
    with pytest.raises(ValueError):
        build_training_data(history, "decisions", use_birth_info=False, features=("height",))
    with pytest.raises(ValueError):
        build_training_data(history, "decisions", use_birth_info=False, features=())


def test_selector_trained_on_a_subset_scores_with_the_same_subset():
    history = _history(SMALL, n=2)
    selector = train_selector(
        history, regime="later_performance", use_birth_info=False, features=("observed",)
    )
    assert selector.feature_names == ("observed",)
    res = simulate(replace(SMALL, seed=50), selector)
    assert res.selected.sum() == round(SMALL.stage_fractions[-1] * SMALL.n_athletes)
