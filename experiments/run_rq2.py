"""RQ2: does the training-exposure feedback loop increase talent loss across stages and cohorts?

Part A (across stages): Q1/Q4 selection-rate ratio, talent loss and equal-opportunity gap at
each stage, for a memoryless selector and a scout with memory, as the exposure weight grows.
Runs with the same seed face the same athletes, so differences across exposure weights are
paired.

Part B (across cohorts): an ML selector is trained on past logs, deployed on the next cohorts,
and retrained on its own decisions and outcomes for several generations. Generation 0 is the
historical data of the scout. Cohorts are independent birth years: the only channel across
cohorts is the data the model is retrained on.

Usage: python experiments/run_rq2.py [--seeds 10] [--athletes 20000]
       [--model linear|linear_by_stage|gbm]
Results are written to ``results/rq2_stages.csv`` and ``results/rq2_generations_<model>.csv``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product

import pandas as pd

from common import EXPOSURES, SCOUT, save
from rae import PerformanceSelector, SimConfig, run_generations, simulate
from rae.metrics import (
    equal_opportunity_gap_by_stage,
    selection_rate_ratio_by_stage,
    talent_loss_by_stage,
)
from rae.stats import mean_ci

DOSE = (0.0, 0.5, 1.0, 2.0)  # exposure weights for the dose-response of Part A
SELECTORS = {"memoryless performance": PerformanceSelector(), "scout with memory": SCOUT}
REGIMES = ("decisions", "later_performance")
N_GENERATIONS = 5


def part_a(n_seeds: int, n_athletes: int) -> pd.DataFrame:
    rows = []
    for exposure, (name, selector) in product(DOSE, SELECTORS.items()):
        cfg = SimConfig(n_athletes=n_athletes, exposure_weight=exposure)
        for seed in range(n_seeds):
            res = simulate(replace(cfg, seed=seed), selector)
            columns = zip(
                selection_rate_ratio_by_stage(res),
                talent_loss_by_stage(res),
                equal_opportunity_gap_by_stage(res),
                strict=True,
            )
            rows += [
                {
                    "selector": name,
                    "exposure_weight": exposure,
                    "seed": seed,
                    "stage": stage,
                    "ratio_q1_q4": ratio,
                    "talent_loss": loss,
                    "equal_opportunity_gap": gap,
                }
                for stage, (ratio, loss, gap) in enumerate(columns)
            ]
    return pd.DataFrame(rows)


def part_b(n_seeds: int, model: str, n_athletes: int) -> pd.DataFrame:
    frames = []
    for exposure in EXPOSURES:
        cfg = SimConfig(n_athletes=n_athletes, exposure_weight=exposure)
        for regime, use_birth_info in product(REGIMES, (False, True)):
            for seed in range(n_seeds):
                df = run_generations(
                    cfg,
                    regime=regime,
                    use_birth_info=use_birth_info,
                    model=model,
                    historical=SCOUT,
                    n_generations=N_GENERATIONS,
                    seed=seed,
                )
                frames.append(
                    df.assign(
                        exposure_weight=exposure,
                        regime=regime,
                        birth_info="yes" if use_birth_info else "no",
                        seed=seed,
                    )
                )
    return pd.concat(frames, ignore_index=True)


def print_stage_tables(stages: pd.DataFrame) -> None:
    for metric in ("ratio_q1_q4", "talent_loss", "equal_opportunity_gap"):
        print(f"\nPart A: {metric} by stage (mean over seeds)")
        table = (
            stages.groupby(["selector", "exposure_weight", "stage"])[metric].mean().unstack("stage")
        )
        print(table.round(3).to_string())

    print("\nPart A: paired change in final-stage talent loss vs exposure_weight = 0")
    last = stages[stages["stage"] == stages["stage"].max()]
    for name, block in last.groupby("selector"):
        wide = block.pivot(index="seed", columns="exposure_weight", values="talent_loss")
        for exposure in DOSE[1:]:
            mean, half = mean_ci(wide[exposure] - wide[0.0])
            print(f"  {name:24s} c = {exposure:3.1f}: {mean:+.4f} +/- {half:.4f}")


def print_generation_table(gens: pd.DataFrame, metric: str, title: str) -> None:
    print(f"\nPart B: {title} by generation (0 = historical scout cohorts)")
    keys = ["exposure_weight", "regime", "birth_info"]
    table = gens.groupby([*keys, "generation"])[metric].mean().unstack("generation")
    print(table.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--model", choices=["linear", "linear_by_stage", "gbm"], default="linear")
    parser.add_argument("--athletes", type=int, default=20_000)
    args = parser.parse_args()

    stages = part_a(args.seeds, args.athletes)
    save(stages, "rq2_stages.csv")
    print_stage_tables(stages)

    gens = part_b(args.seeds, args.model, args.athletes)
    save(gens, f"rq2_generations_{args.model}.csv")
    print_generation_table(gens, "selection_rate_ratio_q1_q4", "Q1/Q4 selection-rate ratio")
    print_generation_table(gens, "overall_talent_loss", "overall talent loss")
    print_generation_table(gens, "regret", "regret")


if __name__ == "__main__":
    main()
