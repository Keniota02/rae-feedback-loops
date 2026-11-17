"""Baseline experiment: how much does a performance-based selector amplify the RAE?

Compares the oracle with a performance-based selector without and with the training-exposure
feedback loop, over several seeds, and prints means with 95% confidence intervals.
Results are written to ``results/baseline_runs.csv`` (ignored by git).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from rae import OracleSelector, PerformanceSelector, SimConfig, simulate
from rae.metrics import summarize

N_SEEDS = 30
BASE = SimConfig(n_athletes=20_000)

CONDITIONS = {
    "oracle": (OracleSelector(), BASE),
    "performance, no feedback loop": (PerformanceSelector(), BASE),
    "performance, with feedback loop": (
        PerformanceSelector(),
        replace(BASE, exposure_weight=0.5),
    ),
}


def run() -> pd.DataFrame:
    rows = []
    for name, (selector, cfg) in CONDITIONS.items():
        for seed in range(N_SEEDS):
            metrics = summarize(simulate(replace(cfg, seed=seed), selector))
            rows.append({"condition": name, "seed": seed, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    runs = run()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    runs.to_csv(out_dir / "baseline_runs.csv", index=False)

    metric_cols = [c for c in runs.columns if c not in ("condition", "seed")]
    grouped = runs.groupby("condition", sort=False)[metric_cols]
    mean = grouped.mean()
    half_ci = 1.96 * grouped.std() / np.sqrt(N_SEEDS)
    for condition in mean.index:
        print(f"\n{condition}")
        for col in metric_cols:
            value, err = mean.loc[condition, col], half_ci.loc[condition, col]
            print(f"  {col:32s} {value:8.3f} +/- {err:.3f}")


if __name__ == "__main__":
    main()
