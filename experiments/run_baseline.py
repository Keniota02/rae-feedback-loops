"""Baseline: how much bias does each human-like selector introduce, with and without the loop?

Compares the oracle, a memoryless performance-based selector and a scout with memory, with
the feedback loop off and on, over several seeds (paired: same athletes for every selector).
Results are written to ``results/baseline_runs.csv`` (ignored by git).

Usage: python experiments/run_baseline.py [--seeds 30] [--athletes 20000]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from common import EXPOSURES, SCOUT, fmt, save
from rae import OracleSelector, PerformanceSelector, SimConfig, simulate
from rae.metrics import summarize

SELECTORS = {
    "oracle": OracleSelector(),
    "memoryless performance": PerformanceSelector(),
    "scout with memory": SCOUT,
}


def run(n_seeds: int, n_athletes: int) -> pd.DataFrame:
    rows = []
    for exposure in EXPOSURES:
        cfg = SimConfig(n_athletes=n_athletes, exposure_weight=exposure)
        for name, selector in SELECTORS.items():
            for seed in range(n_seeds):
                metrics = summarize(simulate(replace(cfg, seed=seed), selector))
                row = {"exposure_weight": exposure, "selector": name, "seed": seed}
                rows.append({**row, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--athletes", type=int, default=20_000)
    args = parser.parse_args()

    runs = run(args.seeds, args.athletes)
    save(runs, "baseline_runs.csv")

    metric_cols = [c for c in runs.columns if c not in ("exposure_weight", "selector", "seed")]
    for (exposure, name), block in runs.groupby(["exposure_weight", "selector"], sort=False):
        print(f"\n{name}, exposure_weight = {exposure}")
        for col in metric_cols:
            print(f"  {col:32s} {fmt(block[col])}")


if __name__ == "__main__":
    main()
