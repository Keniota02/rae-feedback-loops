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
    maturation_weight: float = 1.0  # b0: weight of maturation at stage 0
    maturation_decay: float = 0.7  # b_s = b0 * decay**s: maturation matters less at later stages
    maturation_corr: float = 0.6  # rho: correlation between relative age and maturation
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
        if self.noise_sd < 0.0:
            raise ValueError("noise_sd must be non-negative")
        if not 0.0 < self.top_talent_fraction < 1.0:
            raise ValueError("top_talent_fraction must be in (0, 1)")

    @property
    def n_stages(self) -> int:
        return len(self.stage_fractions)
