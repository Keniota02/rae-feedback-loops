"""RQ1: how much does an ML selector trained on historical outcomes amplify the RAE?

For each seed, historical cohorts are produced by a human-like (performance-based) selector.
ML selectors are trained on their logs and deployed on a fresh cohort, together with the
oracle and the human-like selector. All selectors face the same test athletes (common random
numbers), so differences with the human-like baseline are paired.

Grid: labelling regime (past decisions vs later performance) x birth information
(withheld vs available) x feedback loop (exposure weight 0 vs 0.5).

Usage: python experiments/run_rq1.py [--seeds 20] [--model linear|gbm] [--athletes 20000]
Results are written to ``results/rq1_runs.csv`` (ignored by git).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from rae import OracleSelector, PerformanceSelector, SimConfig, simulate, train_selector
from rae.metrics import summarize

HUMAN = PerformanceSelector()
EXPOSURES = (0.0, 0.5)
REGIMES = ("decisions", "later_performance")
N_HISTORICAL_COHORTS = 3
METRICS = [
    "selection_rate_ratio_q1_q4",
    "overall_talent_loss",
    "equal_opportunity_gap",
    "regret",
]


def selector_name(regime: str, use_birth_info: bool) -> str:
    return f"ML {regime}, birth info {'yes' if use_birth_info else 'no'}"


def run(n_seeds: int, model: str, n_athletes: int) -> pd.DataFrame:
    rows = []
    for exposure in EXPOSURES:
        cfg = replace(SimConfig(n_athletes=n_athletes), exposure_weight=exposure)
        for seed in range(n_seeds):
            history = [
                simulate(replace(cfg, seed=seed * 1_000 + k), HUMAN)
                for k in range(N_HISTORICAL_COHORTS)
            ]
            test_cfg = replace(cfg, seed=seed * 1_000 + 999)
            selectors = {"oracle": OracleSelector(), "human (historical)": HUMAN}
            for regime, use_birth_info in product(REGIMES, (False, True)):
                selectors[selector_name(regime, use_birth_info)] = train_selector(
                    history, regime=regime, use_birth_info=use_birth_info, model=model, seed=seed
                )
            for name, selector in selectors.items():
                metrics = summarize(simulate(test_cfg, selector))
                rows.append(
                    {"exposure_weight": exposure, "seed": seed, "selector": name, **metrics}
                )
    return pd.DataFrame(rows)


def mean_ci(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def report(runs: pd.DataFrame) -> None:
    order = list(dict.fromkeys(runs["selector"]))
    for exposure, block in runs.groupby("exposure_weight"):
        print(
            f"\n=== exposure_weight = {exposure} (feedback loop {'on' if exposure else 'off'}) ==="
        )
        ratio = block.pivot(index="seed", columns="selector", values=METRICS[0])
        header = f"{'selector':46s}" + "".join(f"{m[:18]:>20s}" for m in METRICS)
        print(header + f"{'d ratio vs human':>20s}")
        for name in order:
            cells = []
            for metric in METRICS:
                mean, half = mean_ci(block.loc[block["selector"] == name, metric])
                cells.append(f"{mean:11.3f} +/-{half:5.3f}")
            delta_mean, delta_half = mean_ci(ratio[name] - ratio["human (historical)"])
            print(f"{name:46s}" + " ".join(cells) + f"  {delta_mean:+7.3f} +/-{delta_half:5.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--model", choices=["linear", "gbm"], default="linear")
    parser.add_argument("--athletes", type=int, default=20_000)
    args = parser.parse_args()

    runs = run(args.seeds, args.model, args.athletes)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    runs.to_csv(out_dir / "rq1_runs.csv", index=False)
    report(runs)


if __name__ == "__main__":
    main()
