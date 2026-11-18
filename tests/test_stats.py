import math

import pytest

from rae.stats import mean_ci


def test_mean_ci_uses_the_student_t_quantile():
    mean, half = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert mean == pytest.approx(2.5)
    # sd = 1.2910, t_{0.975, 3} = 3.1824
    assert half == pytest.approx(3.1824 * 1.2910 / 2.0, rel=1e-3)


def test_mean_ci_with_a_single_value_has_no_width():
    mean, half = mean_ci([5.0])
    assert mean == 5.0
    assert math.isnan(half)
