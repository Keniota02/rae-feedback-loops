"""Simulation of the Relative Age Effect (RAE) and selection feedback loops."""

from rae.athletes import Athletes, generate_athletes
from rae.config import SimConfig
from rae.generations import run_generations
from rae.learning import MLSelector, build_training_data, train_selector
from rae.pipeline import SimulationResult, StageRecord, simulate
from rae.selectors import OracleSelector, PerformanceSelector, Selector

__version__ = "0.1.0"

__all__ = [
    "Athletes",
    "MLSelector",
    "OracleSelector",
    "PerformanceSelector",
    "Selector",
    "SimConfig",
    "SimulationResult",
    "StageRecord",
    "build_training_data",
    "generate_athletes",
    "run_generations",
    "simulate",
    "train_selector",
]
