"""Synthetic athletes: latent talent, birth date within the selection year, maturation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_UNIFORM_SD = np.sqrt(1.0 / 12.0)  # standard deviation of a Uniform(0, 1) variable


@dataclass(frozen=True)
class Athletes:
    """A cohort of athletes. All arrays have one entry per athlete."""

    talent: np.ndarray  # latent talent T, independent of birth date by construction
    birth_frac: np.ndarray  # position in the selection year, in [0, 1); 0 = right after cut-off
    maturation: np.ndarray  # biological maturation M, zero mean and unit variance

    def __len__(self) -> int:
        return int(self.talent.size)

    @property
    def quarter(self) -> np.ndarray:
        """Birth quarter after the cut-off: 0 (Q1, oldest in the group) to 3 (Q4, youngest)."""
        return np.minimum((self.birth_frac * 4).astype(int), 3)


def generate_athletes(n: int, maturation_corr: float, rng: np.random.Generator) -> Athletes:
    """Draw a cohort in which talent is independent of birth date.

    Maturation is ``M = rho * ra + sqrt(1 - rho**2) * eps`` where ``ra`` is the standardised
    relative age (positive for early-born athletes) and ``eps`` is standard normal noise.
    """
    talent = rng.standard_normal(n)
    birth_frac = rng.random(n)
    relative_age = (0.5 - birth_frac) / _UNIFORM_SD
    noise = rng.standard_normal(n)
    maturation = maturation_corr * relative_age + np.sqrt(1.0 - maturation_corr**2) * noise
    return Athletes(talent=talent, birth_frac=birth_frac, maturation=maturation)
