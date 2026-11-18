from dataclasses import replace

import pandas as pd
import pytest

from rae import PerformanceSelector, SimConfig, run_generations, simulate
from rae.metrics import summarize

CFG = SimConfig(n_athletes=5_000, exposure_weight=0.5)


def test_shape_and_columns():
    df = run_generations(
        CFG, regime="decisions", use_birth_info=True, n_generations=3, cohorts_per_generation=2
    )
    assert len(df) == (3 + 1) * 2
    assert sorted(df["generation"].unique()) == [0, 1, 2, 3]
    assert "selection_rate_ratio_q1_q4" in df.columns


def test_is_reproducible():
    kwargs = {
        "regime": "later_performance",
        "use_birth_info": True,
        "n_generations": 2,
        "cohorts_per_generation": 1,
        "seed": 5,
    }
    pd.testing.assert_frame_equal(run_generations(CFG, **kwargs), run_generations(CFG, **kwargs))


def test_generation_zero_is_the_human_like_baseline():
    df = run_generations(
        CFG,
        regime="decisions",
        use_birth_info=False,
        n_generations=1,
        cohorts_per_generation=1,
        seed=2,
    )
    direct = summarize(simulate(replace(CFG, seed=2 * 1_000_000), PerformanceSelector()))
    row = df[(df["generation"] == 0) & (df["cohort"] == 0)].iloc[0]
    assert row["selection_rate_ratio_q1_q4"] == pytest.approx(direct["selection_rate_ratio_q1_q4"])


def test_retraining_loop_does_not_create_rae_from_nothing():
    """Without any real bias the Q1/Q4 ratio fluctuates around 1 and shows no drift.

    A single cohort has a standard deviation of about 0.05, so the check uses the mean over
    all generations and cohorts, plus a loose bound on every single value.
    """
    cfg = SimConfig(n_athletes=30_000, maturation_weight=0.0, exposure_weight=0.0)
    df = run_generations(
        cfg,
        regime="later_performance",
        use_birth_info=True,
        n_generations=3,
        cohorts_per_generation=2,
    )
    ratios = df["selection_rate_ratio_q1_q4"]
    assert ratios.mean() == pytest.approx(1.0, abs=0.06)
    assert ratios.between(0.75, 1.3).all()
