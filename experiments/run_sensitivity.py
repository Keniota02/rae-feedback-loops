"""Sensitivity study: how much do the conclusions depend on the assumed parameters?

Part A (one at a time): each parameter is varied around the calibrated defaults (feedback loop
on, c = 1) while the others stay fixed. For the scout and for the three mitigations that worked
in RQ3 (at full strength) it reports the Q1/Q4 ratio, the talent loss among Q4 athletes and the
regret, and the paired change of the mitigations with respect to the scout. The parameter
``adult_maturation_weight`` (kappa) lets part of the maturation advantage persist into adult
potential, which is the case in which mitigations can have a cost.

Part B (loop strength): whether rotating selection slots help when the feedback loop is stronger
(larger exposure weight) and re-entry is easier (lower dropout of unselected athletes).

Usage: python experiments/run_sensitivity.py [--seeds 10] [--athletes 20000]
Results are written to ``results/sensitivity_oat.csv`` and ``results/sensitivity_loop.csv``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product

import pandas as pd

from common import SCOUT, fmt, save
from rae import ScoutSelector, SimConfig, simulate
from rae.metrics import summarize
from rae.mitigations import BioBanding, BirthQuarterQuota, RelativeAgeAdjusted, RotatingSlots

BASE = SimConfig(exposure_weight=1.0)
SWEEPS = {
    "maturation_corr": (0.15, 0.3, 0.45, 0.6),
    "maturation_weight": (1.0, 2.0, 3.0),
    "maturation_decay": (0.5, 0.7, 0.9),
    "noise_sd": (0.5, 1.0, 1.5),
    "top_talent_fraction": (0.05, 0.10, 0.20),
    "exposure_weight": (0.0, 1.0, 2.0),
    "adult_maturation_weight": (0.0, 0.25, 0.5, 1.0),
    "scout_noise": (0.0, 0.5, 1.0),  # judgement noise of the scout, not a SimConfig field
}
MITIGATIONS = {
    "RA-adjusted": RelativeAgeAdjusted,
    "quota": BirthQuarterQuota,
    "bio-banding": BioBanding,
}
METRICS = ["selection_rate_ratio_q1_q4", "talent_loss_q4", "overall_talent_loss", "regret"]
LOOP_EXPOSURES = (0.0, 1.0, 2.0, 4.0)
LOOP_DROPOUTS = (0.0, 0.2)
ROTATION = (0.0, 0.2, 0.4)


def part_a(n_seeds: int, n_athletes: int) -> pd.DataFrame:
    rows = []
    for parameter, levels in SWEEPS.items():
        for level in levels:
            scout = SCOUT
            cfg = replace(BASE, n_athletes=n_athletes)
            if parameter == "scout_noise":
                scout = ScoutSelector(memory=SCOUT.memory, noise_sd=level)
            else:
                cfg = replace(cfg, **{parameter: level})
            selectors = {"scout": scout} | {n: m(scout) for n, m in MITIGATIONS.items()}
            for seed, (name, selector) in product(range(n_seeds), selectors.items()):
                metrics = summarize(simulate(replace(cfg, seed=seed), selector))
                row = {"parameter": parameter, "level": level, "seed": seed, "selector": name}
                rows.append({**row, **metrics})
    return pd.DataFrame(rows)


def part_b(n_seeds: int, n_athletes: int) -> pd.DataFrame:
    rows = []
    for exposure, dropout, share in product(LOOP_EXPOSURES, LOOP_DROPOUTS, ROTATION):
        cfg = replace(
            BASE, n_athletes=n_athletes, exposure_weight=exposure, dropout_unselected=dropout
        )
        selector = RotatingSlots(SCOUT, share=share) if share else SCOUT
        for seed in range(n_seeds):
            metrics = summarize(simulate(replace(cfg, seed=seed), selector))
            row = {"exposure_weight": exposure, "dropout": dropout, "rotation": share, "seed": seed}
            rows.append({**row, **metrics})
    return pd.DataFrame(rows)


def report_a(runs: pd.DataFrame) -> None:
    print("\nPart A: one-at-a-time sensitivity (loop on, c = 1 unless varied)")
    for parameter, block in runs.groupby("parameter", sort=False):
        print(f"\n--- {parameter} ---")
        print(
            f"{'level':>7s} {'scout Q1/Q4':>17s} {'scout Q4 loss':>17s} {'scout regret':>17s}"
            + "".join(f"{'d regret ' + m:>27s}" for m in MITIGATIONS)
            + "".join(f"{'d Q4 loss ' + m:>27s}" for m in MITIGATIONS)
        )
        for level, sub in block.groupby("level"):
            scout = sub[sub["selector"] == "scout"].set_index("seed")
            cells = [fmt(scout[m]) for m in ("selection_rate_ratio_q1_q4", "talent_loss_q4")]
            cells.append(fmt(scout["regret"]))
            for metric in ("regret", "talent_loss_q4"):
                for name in MITIGATIONS:
                    mitigated = sub[sub["selector"] == name].set_index("seed")
                    cells.append(f"{fmt(mitigated[metric] - scout[metric]):>27s}")
            print(f"{level:7.2f} " + " ".join(cells))


def report_b(runs: pd.DataFrame) -> None:
    print("\nPart B: rotating slots vs feedback-loop strength (paired change vs no rotation)")
    print(f"{'c':>4s} {'dropout':>8s} {'rotation':>9s} {'d overall loss':>18s} {'d regret':>18s}")
    for (exposure, dropout), block in runs.groupby(["exposure_weight", "dropout"]):
        none = block[block["rotation"] == 0.0].set_index("seed")
        for share in ROTATION[1:]:
            rot = block[block["rotation"] == share].set_index("seed")
            d_loss = fmt(rot["overall_talent_loss"] - none["overall_talent_loss"])
            d_regret = fmt(rot["regret"] - none["regret"])
            print(f"{exposure:4.1f} {dropout:8.1f} {share:9.1f} {d_loss:>18s} {d_regret:>18s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--athletes", type=int, default=20_000)
    args = parser.parse_args()

    oat = part_a(args.seeds, args.athletes)
    save(oat, "sensitivity_oat.csv")
    report_a(oat)
    loop = part_b(args.seeds, args.athletes)
    save(loop, "sensitivity_loop.csv")
    report_b(loop)


if __name__ == "__main__":
    main()
