import numpy as np

from rae import generate_athletes


def test_talent_is_independent_of_birth_date():
    athletes = generate_athletes(200_000, 0.6, np.random.default_rng(1))
    corr = np.corrcoef(athletes.talent, athletes.birth_frac)[0, 1]
    assert abs(corr) < 0.01


def test_maturation_has_unit_variance_and_requested_correlation():
    rho = 0.6
    athletes = generate_athletes(200_000, rho, np.random.default_rng(2))
    assert abs(athletes.maturation.std() - 1.0) < 0.02
    corr_with_earliness = np.corrcoef(athletes.maturation, -athletes.birth_frac)[0, 1]
    assert abs(corr_with_earliness - rho) < 0.02


def test_quarters_are_balanced_and_in_range():
    athletes = generate_athletes(100_000, 0.5, np.random.default_rng(3))
    assert set(np.unique(athletes.quarter)) == {0, 1, 2, 3}
    shares = np.bincount(athletes.quarter) / len(athletes)
    assert np.allclose(shares, 0.25, atol=0.01)
