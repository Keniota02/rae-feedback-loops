"""RQ2: does the training-exposure feedback loop increase talent loss across stages and cohorts?

Part A (across stages): Q1/Q4 selection-rate ratio at each stage for the human-like selector,
with the feedback loop off and on.

Part B (across cohorts): an ML selector is trained on past logs, deployed on the next cohorts,
and retrained on its own decisions and outcomes for several generations. Generation 0 is the
historical human-like data.

Usage: python experiments/run_rq2.py [--seeds 10] [--model linear|gbm] [--athletes 20000]
Results are written to ``results/rq2_stages.csv`` and ``results/rq2_generations.csv``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd

from rae import PerformanceSelector, SimConfig, run_generations, simulate
from rae.metrics import selection_rate_ratio_by_stage

EXPOSURES = (0.0, 0.5)
REGIMES = ("decisions", "later_performance")
N_GENERATIONS = 5


def part_a(n_seeds: int, n_athletes: int) -> pd.DataFrame:
    rows = []
    for exposure in EXPOSURES:
        cfg = replace(SimConfig(n_athletes=n_athletes), exposure_weight=exposure)
        for seed in range(n_seeds):
            ratios = selection_rate_ratio_by_stage(
                simulate(replace(cfg, seed=seed), PerformanceSelector())
            )
            rows += [
                {"exposure_weight": exposure, "seed": seed, "stage": s, "ratio_q1_q4": r}
                for s, r in enumerate(ratios)
            ]
    return pd.DataFrame(rows)


def part_b(n_seeds: int, model: str, n_athletes: int) -> pd.DataFrame:
    frames = []
    for exposure in EXPOSURES:
        cfg = replace(SimConfig(n_athletes=n_athletes), exposure_weight=exposure)
        for regime, use_birth_info in product(REGIMES, (False, True)):
            for seed in range(n_seeds):
                df = run_generations(
                    cfg,
                    regime=regime,
                    use_birth_info=use_birth_info,
                    model=model,
                    n_generations=N_GENERATIONS,
                    seed=seed,
                )
                df = df.assign(
                    exposure_weight=exposure,
                    regime=regime,
                    birth_info="yes" if use_birth_info else "no",
                    seed=seed,
                )
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def print_stage_table(stages: pd.DataFrame) -> None:
    print("\nPart A: Q1/Q4 selection-rate ratio by stage (human-like selector)")
    table = stages.groupby(["exposure_weight", "stage"])["ratio_q1_q4"].mean().unstack("stage")
    print(table.round(3).to_string())


def print_generation_table(gens: pd.DataFrame, metric: str, title: str) -> None:
    print(f"\nPart B: {title} by generation (0 = historical human-like cohorts)")
    keys = ["exposure_weight", "regime", "birth_info"]
    table = gens.groupby([*keys, "generation"])[metric].mean().unstack("generation")
    print(table.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--model", choices=["linear", "gbm"], default="linear")
    parser.add_argument("--athletes", type=int, default=20_000)
    args = parser.parse_args()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    stages = part_a(args.seeds, args.athletes)
    stages.to_csv(out_dir / "rq2_stages.csv", index=False)
    print_stage_table(stages)

    gens = part_b(args.seeds, args.model, args.athletes)
    gens.to_csv(out_dir / "rq2_generations.csv", index=False)
    print_generation_table(gens, "selection_rate_ratio_q1_q4", "Q1/Q4 selection-rate ratio")
    print_generation_table(gens, "overall_talent_loss", "overall talent loss")
    print_generation_table(gens, "regret", "regret")


if __name__ == "__main__":
    main()
