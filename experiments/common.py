"""Settings shared by the experiment scripts.

The values below are placeholders until the parameters are justified (see the report,
"Open modelling decisions"); keeping them in one place makes the sensitivity study easier.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rae import ScoutSelector
from rae.stats import mean_ci

# Human-like selectors that produce the historical data.
# A scout remembers every stage (memory 1) and judges with noise; the biased scout also
# favours athletes who look physically mature.
SCOUT = ScoutSelector(memory=1.0, noise_sd=0.5)
BIASED_SCOUT = ScoutSelector(memory=1.0, noise_sd=0.5, maturation_bias=0.5)
HISTORICAL = {"scout": SCOUT, "biased scout": BIASED_SCOUT}

# Feedback loop off / on (weight c of accumulated training exposure).
EXPOSURES = (0.0, 1.0)

RESULTS_DIR = Path("results")


def save(df: pd.DataFrame, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    df.to_csv(path, index=False)
    return path


def fmt(values) -> str:
    """Mean +/- half-width of the 95% t confidence interval."""
    mean, half = mean_ci(values)
    return f"{mean:7.3f} +/-{half:6.3f}"
