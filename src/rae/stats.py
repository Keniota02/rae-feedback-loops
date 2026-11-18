"""Summary statistics for results over independent seeds."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of a Student-t confidence interval.

    Seeds are few (10-30), so the t quantile is used instead of the normal 1.96.
    Returns a half-width of NaN when fewer than two values are given.
    """
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    if x.size < 2:
        return mean, float("nan")
    t = stats.t.ppf(0.5 + confidence / 2.0, df=x.size - 1)
    return mean, float(t * x.std(ddof=1) / np.sqrt(x.size))
