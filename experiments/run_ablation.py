"""RQ1 follow-up: do features without birth date act as proxies for it?

Part A (leakage): on the training logs of the scout, how much of the birth position is
explained by the features a "birth-blind" model sees (correlation of each feature with the
birth position, and R^2 of a linear regression of the birth position on all of them).

Part B (ablation): ML selectors trained WITHOUT birth information on feature subsets, deployed on
a fresh cohort; paired change of the Q1/Q4 ratio with respect to the scout that produced the data.

Usage: python experiments/run_ablation.py [--seeds 20] [--model linear|gbm] [--athletes 20000]
Results are written to ``results/ablation_leakage.csv`` and ``results/ablation_<model>.csv``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from common import EXPOSURES, SCOUT, fmt, save
from rae import SimConfig, build_training_data, simulate, train_selector
from rae.learning import FEATURES
from rae.metrics import summarize

REGIMES = ("decisions", "later_performance")
N_HISTORICAL_COHORTS = 3
FEATURE_SETS = {
    "all": FEATURES,
    "no times_selected": ("observed", "history_mean", "stage"),
    "no history_mean": ("observed", "times_selected", "stage"),
    "observed + stage": ("observed", "stage"),
}
METRICS = ["selection_rate_ratio_q1_q4", "overall_talent_loss", "regret"]


def leakage(history) -> dict[str, float]:
    x, _ = build_training_data(history, "decisions", use_birth_info=True)
    features, birth = x[:, :-1], x[:, -1]
    row = {
        f"corr {name}": float(np.corrcoef(features[:, j], birth)[0, 1])
        for j, name in enumerate(FEATURES)
        if name != "stage"
    }
    row["r2 all features"] = float(LinearRegression().fit(features, birth).score(features, birth))
    return row


def run(n_seeds: int, model: str, n_athletes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    leak_rows, rows = [], []
    for exposure in EXPOSURES:
        cfg = SimConfig(n_athletes=n_athletes, exposure_weight=exposure)
        for seed in range(n_seeds):
            history = [
                simulate(replace(cfg, seed=seed * 1_000 + k), SCOUT)
                for k in range(N_HISTORICAL_COHORTS)
            ]
            leak_rows.append({"exposure_weight": exposure, "seed": seed, **leakage(history)})
            test_cfg = replace(cfg, seed=seed * 1_000 + 999)
            base = {"exposure_weight": exposure, "seed": seed}
            rows.append({**base, "selector": "scout", **summarize(simulate(test_cfg, SCOUT))})
            for regime, (set_name, features) in product(REGIMES, FEATURE_SETS.items()):
                selector = train_selector(
                    history,
                    regime=regime,
                    use_birth_info=False,
                    model=model,
                    seed=seed,
                    features=features,
                )
                metrics = summarize(simulate(test_cfg, selector))
                rows.append({**base, "selector": f"ML {regime}, {set_name}", **metrics})
    return pd.DataFrame(leak_rows), pd.DataFrame(rows)


def report(leak: pd.DataFrame, runs: pd.DataFrame) -> None:
    print("\nPart A: leakage of birth position into birth-blind features (training logs)")
    cols = [c for c in leak.columns if c not in ("exposure_weight", "seed")]
    for exposure, block in leak.groupby("exposure_weight"):
        print(f"  exposure_weight = {exposure}")
        for col in cols:
            print(f"    {col:24s} {fmt(block[col])}")

    print("\nPart B: birth-blind ML selectors on feature subsets")
    order = list(dict.fromkeys(runs["selector"]))
    for exposure, block in runs.groupby("exposure_weight"):
        print(f"\n=== exposure_weight = {exposure} ===")
        header = f"{'selector':40s}" + "".join(f"{m[:20]:>21s}" for m in METRICS)
        print(header + f"{'d ratio vs scout':>21s}")
        ratio = block.pivot(index="seed", columns="selector", values=METRICS[0])
        for name in order:
            cells = [fmt(block.loc[block["selector"] == name, m]) for m in METRICS]
            print(f"{name:40s}" + " ".join(cells) + f"  {fmt(ratio[name] - ratio['scout'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--model", choices=["linear", "gbm"], default="linear")
    parser.add_argument("--athletes", type=int, default=20_000)
    args = parser.parse_args()

    leak, runs = run(args.seeds, args.model, args.athletes)
    save(leak, "ablation_leakage.csv")
    save(runs, f"ablation_{args.model}.csv")
    report(leak, runs)


if __name__ == "__main__":
    main()
