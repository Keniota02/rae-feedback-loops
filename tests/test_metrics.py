import numpy as np
import pytest

from rae import Athletes, SimConfig, SimulationResult
from rae.metrics import (
    equal_opportunity_gap,
    high_talent_mask,
    mean_selected_talent,
    overall_talent_loss,
    regret,
    selection_rate_by_quarter,
    selection_rate_ratio,
    talent_loss_by_quarter,
)


def _result(selected: list[bool]) -> SimulationResult:
    """Eight athletes, two per birth quarter; half of them (every other one) have high talent."""
    athletes = Athletes(
        talent=np.array([3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0]),
        birth_frac=np.array([0.1, 0.1, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9]),
        maturation=np.zeros(8),
    )
    cfg = SimConfig(n_athletes=8, stage_fractions=(0.5,), top_talent_fraction=0.5)
    return SimulationResult(
        athletes=athletes,
        selected=np.array(selected),
        times_selected=np.zeros(8, dtype=int),
        config=cfg,
    )


def test_selection_rates_and_ratio():
    res = _result([True, True, True, False, False, False, False, True])
    assert np.allclose(selection_rate_by_quarter(res), [1.0, 0.5, 0.0, 0.5])
    assert selection_rate_ratio(res) == pytest.approx(2.0)


def test_talent_loss_and_equal_opportunity_gap():
    res = _result([True, False, True, False, False, False, True, False])
    assert high_talent_mask(res).sum() == 4
    assert np.allclose(talent_loss_by_quarter(res), [0.0, 0.0, 1.0, 0.0])
    assert overall_talent_loss(res) == pytest.approx(0.25)
    assert equal_opportunity_gap(res) == pytest.approx(1.0)


def test_efficiency_and_regret():
    res = _result([True, False, True, False, False, True, False, False])
    assert mean_selected_talent(res) == pytest.approx(2.0)  # talents 3, 3, 0
    assert regret(res) == pytest.approx(1.0)  # oracle picks 3, 3, 3
