"""RQ1: how much does an ML selector trained on historical outcomes amplify the RAE?

For each seed, historical cohorts are produced by a human-like selector (a noisy scout with
memory, or the same scout with a bias towards physically mature athletes). ML selectors are
trained on their logs and deployed on a fresh cohort, together with the oracle and the
historical selector itself. All selectors face the same test athletes (common random numbers),
so differences with the historical selector are paired.

Grid: historical selector x labelling regime (past decisions vs later performance) x birth
information (withheld vs available) x feedback loop (off vs on).

Usage: python experiments/run_rq1.py [--seeds 20] [--athletes 20000]
       [--model linear|linear_by_stage|gbm]
Results are written to ``results/rq1_runs_<model>.csv`` (ignored by git).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product

import pandas as pd

from common import EXPOSURES, HISTORICAL, fmt, save
from rae import OracleSelector, SimConfig, simulate, train_selector
from rae.metrics import summarize

REGIMES = ("decisions", "later_performance")
N_HISTORICAL_COHORTS = 3
METRICS = [
    "selection_rate_ratio_q1_q4",
    "overall_talent_loss",
    "equal_opportunity_gap",
    "regret",
]
BASELINE = "historical selector"


def selector_name(regime: str, use_birth_info: bool) -> str:
    return f"ML {regime}, birth info {'yes' if use_birth_info else 'no'}"


def run(n_seeds: int, model: str, n_athletes: int) -> pd.DataFrame:
    rows = []
    for exposure, (hist_name, human) in product(EXPOSURES, HISTORICAL.items()):
        cfg = SimConfig(n_athletes=n_athletes, exposure_weight=exposure)
        for seed in range(n_seeds):
            history = [
                simulate(replace(cfg, seed=seed * 1_000 + k), human)
                for k in range(N_HISTORICAL_COHORTS)
            ]
            test_cfg = replace(cfg, seed=seed * 1_000 + 999)
            selectors = {"oracle": OracleSelector(), BASELINE: human}
            for regime, use_birth_info in product(REGIMES, (False, True)):
                selectors[selector_name(regime, use_birth_info)] = train_selector(
                    history, regime=regime, use_birth_info=use_birth_info, model=model, seed=seed
                )
            for name, selector in selectors.items():
                rows.append(
                    {
                        "exposure_weight": exposure,
                        "historical": hist_name,
                        "seed": seed,
                        "selector": name,
                        **summarize(simulate(test_cfg, selector)),
                    }
                )
    return pd.DataFrame(rows)


def report(runs: pd.DataFrame) -> None:
    order = list(dict.fromkeys(runs["selector"]))
    for (exposure, hist_name), block in runs.groupby(["exposure_weight", "historical"]):
        loop = "on" if exposure else "off"
        print(f"\n=== historical: {hist_name}, exposure_weight = {exposure} (loop {loop}) ===")
        header = f"{'selector':40s}" + "".join(f"{m[:20]:>21s}" for m in METRICS)
        print(header + f"{'d ratio vs historical':>23s}")
        ratio = block.pivot(index="seed", columns="selector", values=METRICS[0])
        for name in order:
            cells = [fmt(block.loc[block["selector"] == name, m]) for m in METRICS]
            delta = fmt(ratio[name] - ratio[BASELINE])
            print(f"{name:40s}" + " ".join(cells) + f"  {delta}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--model", choices=["linear", "linear_by_stage", "gbm"], default="linear")
    parser.add_argument("--athletes", type=int, default=20_000)
    args = parser.parse_args()

    runs = run(args.seeds, args.model, args.athletes)
    save(runs, f"rq1_runs_{args.model}.csv")
    report(runs)


if __name__ == "__main__":
    main()
