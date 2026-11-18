"""RQ3: which mitigations reduce talent loss, and at what cost in selection efficiency?

Each mitigation is applied, at two or more strengths, on top of two base selectors: the scout
(human-like) and the ML selector that amplified the RAE most in RQ1 (trained on later
performance, with birth information, on the scout's historical data). All variants face the
same test cohort as the unmitigated base (paired comparison).

Usage: python experiments/run_rq3.py [--seeds 20] [--model linear|gbm] [--athletes 20000]
Results are written to ``results/rq3_runs_<model>.csv`` (ignored by git).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product

import pandas as pd

from common import EXPOSURES, SCOUT, fmt, save
from rae import OracleSelector, SimConfig, simulate, train_selector
from rae.metrics import summarize
from rae.mitigations import BioBanding, BirthQuarterQuota, RelativeAgeAdjusted, RotatingSlots

N_HISTORICAL_COHORTS = 3
NONE = "none"

# name -> (mitigation family, strength, factory taking the base selector)
MITIGATIONS = {
    NONE: ("none", 0.0, lambda base: base),
    "RA-adjusted 0.5": ("RA-adjusted", 0.5, lambda b: RelativeAgeAdjusted(b, strength=0.5)),
    "RA-adjusted 1.0": ("RA-adjusted", 1.0, lambda b: RelativeAgeAdjusted(b, strength=1.0)),
    "quota 0.5": ("quota", 0.5, lambda b: BirthQuarterQuota(b, share=0.5)),
    "quota 1.0": ("quota", 1.0, lambda b: BirthQuarterQuota(b, share=1.0)),
    "bio-banding 0.5": ("bio-banding", 0.5, lambda b: BioBanding(b, share=0.5)),
    "bio-banding 1.0": ("bio-banding", 1.0, lambda b: BioBanding(b, share=1.0)),
    "rotating 0.2": ("rotating", 0.2, lambda b: RotatingSlots(b, share=0.2)),
    "rotating 0.4": ("rotating", 0.4, lambda b: RotatingSlots(b, share=0.4)),
}
METRICS = [
    "selection_rate_ratio_q1_q4",
    "talent_loss_q4",
    "overall_talent_loss",
    "equal_opportunity_gap",
    "regret",
]


def run(n_seeds: int, model: str, n_athletes: int) -> pd.DataFrame:
    rows = []
    for exposure in EXPOSURES:
        cfg = SimConfig(n_athletes=n_athletes, exposure_weight=exposure)
        for seed in range(n_seeds):
            history = [
                simulate(replace(cfg, seed=seed * 1_000 + k), SCOUT)
                for k in range(N_HISTORICAL_COHORTS)
            ]
            test_cfg = replace(cfg, seed=seed * 1_000 + 999)
            bases = {
                "scout": SCOUT,
                "ML later perf. + birth": train_selector(
                    history, regime="later_performance", use_birth_info=True, model=model, seed=seed
                ),
            }
            info = {"exposure_weight": exposure, "seed": seed}
            oracle = summarize(simulate(test_cfg, OracleSelector()))
            rows.append({**info, "base": "oracle", "mitigation": NONE, **oracle})
            for (base_name, base), (name, (family, strength, wrap)) in product(
                bases.items(), MITIGATIONS.items()
            ):
                metrics = summarize(simulate(test_cfg, wrap(base)))
                rows.append(
                    {
                        **info,
                        "base": base_name,
                        "mitigation": name,
                        "family": family,
                        "strength": strength,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def report(runs: pd.DataFrame) -> None:
    for (exposure, base), block in runs[runs["base"] != "oracle"].groupby(
        ["exposure_weight", "base"], sort=False
    ):
        print(f"\n=== base: {base}, exposure_weight = {exposure} ===")
        header = f"{'mitigation':18s}" + "".join(f"{m[:20]:>21s}" for m in METRICS)
        print(header + f"{'d Q4 loss vs none':>21s}{'d regret vs none':>21s}")
        wide = {m: block.pivot(index="seed", columns="mitigation", values=m) for m in METRICS}
        for name in MITIGATIONS:
            cells = [fmt(block.loc[block["mitigation"] == name, m]) for m in METRICS]
            d_q4 = fmt(wide["talent_loss_q4"][name] - wide["talent_loss_q4"][NONE])
            d_regret = fmt(wide["regret"][name] - wide["regret"][NONE])
            print(f"{name:18s}" + " ".join(cells) + f"  {d_q4}  {d_regret}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--model", choices=["linear", "gbm"], default="linear")
    parser.add_argument("--athletes", type=int, default=20_000)
    args = parser.parse_args()

    runs = run(args.seeds, args.model, args.athletes)
    save(runs, f"rq3_runs_{args.model}.csv")
    report(runs)


if __name__ == "__main__":
    main()
