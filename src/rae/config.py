"""Configuration of the simulated multi-stage selection pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """Parameters of one simulation run.

    Notation follows the model specification in ``report/report.md``.
    Use ``dataclasses.replace(cfg, ...)`` to derive variants of a configuration.
    """

    n_athletes: int = 20_000
    # Fraction of the *initial cohort* selected at each stage (non-increasing).
    stage_fractions: tuple[float, ...] = (0.30, 0.20, 0.10)
    talent_weight: float = 1.0  # a: weight of latent talent in observed performance
    # b0: weight of maturation at stage 0. Calibrated (experiments/run_calibration.py) so that
    # the scout's final-stage Q1/Q4 selection-rate ratio is about 3, between the pooled RAE
    # estimate across levels (OR 1.65) and academy-level estimates (3-5); see the report.
    maturation_weight: float = 2.0
    maturation_decay: float = 0.7  # b_s = b0 * decay**s: maturation matters less at later stages
    # rho: correlation between relative age and maturation. Maturity offset = chronological age
    # minus age at peak height velocity (APHV). Within a selection year the age has sd
    # 1/sqrt(12) = 0.29 years and APHV has sd of about 0.9 years, so rho = 0.29 / 0.95 = 0.3.
    maturation_corr: float = 0.3
    # kappa: share of the maturation advantage that persists into adult performance. The
    # selection target (what an ideal selector would pick on) is T + kappa * M; kappa = 0 (the
    # default) means the maturation advantage is transient. Used in the sensitivity study.
    adult_maturation_weight: float = 0.0
    exposure_weight: float = 0.0  # c: weight of accumulated training exposure (feedback loop)
    exposure_gain: float = 1.0  # exposure units gained per previous stage in which one was selected
    noise_sd: float = 1.0  # sd of the measurement noise on observed performance
    dropout_unselected: float = 0.2  # probability that an unselected athlete leaves the system
    top_talent_fraction: float = 0.10  # athletes counted as "high talent" in fairness metrics
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_athletes <= 0:
            raise ValueError("n_athletes must be positive")
        if not self.stage_fractions:
            raise ValueError("stage_fractions must not be empty")
        if any(not 0.0 < f <= 1.0 for f in self.stage_fractions):
            raise ValueError("every stage fraction must be in (0, 1]")
        if any(b > a for a, b in zip(self.stage_fractions, self.stage_fractions[1:], strict=False)):
            raise ValueError("stage_fractions must be non-increasing")
        if any(round(f * self.n_athletes) < 1 for f in self.stage_fractions):
            raise ValueError("every stage must select at least one athlete")
        if not 0.0 <= self.maturation_corr < 1.0:
            raise ValueError("maturation_corr must be in [0, 1)")
        if not 0.0 <= self.dropout_unselected <= 1.0:
            raise ValueError("dropout_unselected must be in [0, 1]")
        if self.adult_maturation_weight < 0.0:
            raise ValueError("adult_maturation_weight must be non-negative")
        if self.noise_sd < 0.0:
            raise ValueError("noise_sd must be non-negative")
        if not 0.0 < self.top_talent_fraction < 1.0:
            raise ValueError("top_talent_fraction must be in (0, 1)")

    @property
    def n_stages(self) -> int:
        return len(self.stage_fractions)
