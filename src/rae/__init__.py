"""Simulation of the Relative Age Effect (RAE) and selection feedback loops."""

from rae.athletes import Athletes, generate_athletes
from rae.config import SimConfig
from rae.pipeline import SimulationResult, simulate
from rae.selectors import OracleSelector, PerformanceSelector, Selector

__version__ = "0.1.0"

__all__ = [
    "Athletes",
    "OracleSelector",
    "PerformanceSelector",
    "Selector",
    "SimConfig",
    "SimulationResult",
    "generate_athletes",
    "simulate",
]
