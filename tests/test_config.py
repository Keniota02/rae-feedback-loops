import pytest

from rae import SimConfig


def test_default_config_is_valid():
    assert SimConfig().n_stages == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_athletes": 0},
        {"stage_fractions": ()},
        {"stage_fractions": (0.1, 0.2)},  # increasing
        {"stage_fractions": (0.0,)},
        {"stage_fractions": (1.5,)},
        {"maturation_corr": 1.0},
        {"dropout_unselected": 1.5},
        {"noise_sd": -0.1},
        {"top_talent_fraction": 0.0},
        {"n_athletes": 5, "stage_fractions": (0.01,)},  # selects zero athletes
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SimConfig(**kwargs)
