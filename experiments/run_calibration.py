"""Calibration of the maturation weight b0 (stretch goal: calibrated RAE magnitude).

With rho fixed from growth data (0.3), b0 is the free parameter that sets the size of the RAE.
For a grid of b0 values, this script reports the scout's final-stage Q1/Q4 selection-rate ratio
with the feedback loop off and on; the default b0 is the value that gives a ratio of about 3.

Usage: python experiments/run_calibration.py [--seeds 20] [--athletes 20000] [--target 3]
Results are written to ``results/calibration.csv``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from common import EXPOSURES, SCOUT, fmt, save
from rae import SimConfig, simulate
from rae.metrics import selection_rate_ratio

GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def run(n_seeds: int, n_athletes: int) -> pd.DataFrame:
    rows = []
    for exposure, b0 in ((e, b) for e in EXPOSURES for b in GRID):
        cfg = SimConfig(n_athletes=n_athletes, exposure_weight=exposure, maturation_weight=b0)
        for seed in range(n_seeds):
            ratio = selection_rate_ratio(simulate(replace(cfg, seed=seed), SCOUT))
            rows.append({"exposure_weight": exposure, "b0": b0, "seed": seed, "ratio": ratio})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--athletes", type=int, default=20_000)
    parser.add_argument("--target", type=float, default=3.0)
    args = parser.parse_args()

    runs = run(args.seeds, args.athletes)
    save(runs, "calibration.csv")
    for exposure, block in runs.groupby("exposure_weight"):
        print(f"\nexposure_weight = {exposure}: scout's final Q1/Q4 ratio by b0")
        means = block.groupby("b0")["ratio"].mean()
        for b0, group in block.groupby("b0"):
            print(f"  b0 = {b0:3.1f}: {fmt(group['ratio'])}")
        b0_star = float(np.interp(args.target, means.values, means.index.values))
        print(f"  b0 giving a ratio of {args.target:g}: {b0_star:.2f} (linear interpolation)")


if __name__ == "__main__":
    main()
