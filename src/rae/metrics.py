"""Fairness and efficiency metrics. Latent talent is known, so harm can be measured directly."""

from __future__ import annotations

import numpy as np

from rae.pipeline import SimulationResult


def selection_rate_by_quarter(res: SimulationResult) -> np.ndarray:
    """Share of athletes selected at the final stage, for each birth quarter (Q1..Q4)."""
    quarter = res.athletes.quarter
    return np.array([res.selected[quarter == q].mean() for q in range(4)])


def selection_rate_ratio(res: SimulationResult) -> float:
    """Selection rate of Q1 over Q4. Equals 1 when birth date plays no role."""
    rates = selection_rate_by_quarter(res)
    return float(rates[0] / rates[3]) if rates[3] > 0 else float("inf")


def _selected_by_stage(res: SimulationResult) -> list[np.ndarray]:
    """Mask over the whole cohort of the athletes selected at each stage."""
    masks = []
    for record in res.stages:
        chosen = np.zeros(len(res.athletes), dtype=bool)
        chosen[record.pool[record.selected]] = True
        masks.append(chosen)
    return masks


def selection_rate_ratio_by_stage(res: SimulationResult) -> np.ndarray:
    """Q1/Q4 selection-rate ratio at each stage (share of the initial cohort selected there)."""
    quarter = res.athletes.quarter
    ratios = []
    for chosen in _selected_by_stage(res):
        q1, q4 = chosen[quarter == 0].mean(), chosen[quarter == 3].mean()
        ratios.append(q1 / q4 if q4 > 0 else float("inf"))
    return np.array(ratios)


def high_talent_mask(res: SimulationResult) -> np.ndarray:
    """Athletes in the top ``top_talent_fraction`` of latent talent."""
    threshold = np.quantile(res.athletes.talent, 1.0 - res.config.top_talent_fraction)
    return res.athletes.talent >= threshold


def talent_loss_by_quarter(res: SimulationResult) -> np.ndarray:
    """Share of high-talent athletes NOT selected at the final stage, for each birth quarter."""
    high = high_talent_mask(res)
    quarter = res.athletes.quarter
    return np.array([1.0 - res.selected[high & (quarter == q)].mean() for q in range(4)])


def talent_loss_by_stage(res: SimulationResult) -> np.ndarray:
    """Share of high-talent athletes NOT selected at each stage.

    Stages select a shrinking share of the cohort, so part of the loss at early stages is
    unavoidable only when the stage selects fewer athletes than the high-talent share.
    """
    high = high_talent_mask(res)
    return np.array([1.0 - chosen[high].mean() for chosen in _selected_by_stage(res)])


def equal_opportunity_gap_by_stage(res: SimulationResult) -> np.ndarray:
    """Largest difference across birth quarters in the talent loss at each stage."""
    high = high_talent_mask(res)
    quarter = res.athletes.quarter
    gaps = []
    for chosen in _selected_by_stage(res):
        loss = np.array([1.0 - chosen[high & (quarter == q)].mean() for q in range(4)])
        gaps.append(loss.max() - loss.min())
    return np.array(gaps)


def overall_talent_loss(res: SimulationResult) -> float:
    """Share of high-talent athletes not selected at the final stage."""
    return float(1.0 - res.selected[high_talent_mask(res)].mean())


def equal_opportunity_gap(res: SimulationResult) -> float:
    """Largest difference in talent loss across birth quarters."""
    loss = talent_loss_by_quarter(res)
    return float(loss.max() - loss.min())


def mean_selected_talent(res: SimulationResult) -> float:
    """Efficiency: mean latent talent of the selected cohort."""
    return float(res.athletes.talent[res.selected].mean())


def regret(res: SimulationResult) -> float:
    """Gap between the mean talent of an oracle cohort of the same size and the actual one."""
    k = int(res.selected.sum())
    best = np.sort(res.athletes.talent)[::-1][:k].mean()
    return float(best - mean_selected_talent(res))


def summarize(res: SimulationResult) -> dict[str, float]:
    """All scalar metrics of a run."""
    return {
        "selection_rate_ratio_q1_q4": selection_rate_ratio(res),
        "overall_talent_loss": overall_talent_loss(res),
        "talent_loss_q1": float(talent_loss_by_quarter(res)[0]),
        "talent_loss_q4": float(talent_loss_by_quarter(res)[3]),
        "equal_opportunity_gap": equal_opportunity_gap(res),
        "mean_selected_talent": mean_selected_talent(res),
        "regret": regret(res),
    }
