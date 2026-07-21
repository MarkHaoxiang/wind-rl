"""MAPPO training: PPO wiring and the concrete trainer."""

from __future__ import annotations

from wind_rl.rl.mappo import (
    PPOConfig,
    batch_normalise_reward,
    build_loss_module,
    build_optimiser,
)
from wind_rl.rl.trainer import LoggingConfig, MappoTrainer, TrainingConfig
from wind_rl.rl.wind_rose import (
    SMARTEOLE_DIRECTION_OFFSET,
    WindRose,
    WindRoseEvalConfig,
    prepare_wind_rose,
)

__all__ = [
    "SMARTEOLE_DIRECTION_OFFSET",
    "LoggingConfig",
    "MappoTrainer",
    "PPOConfig",
    "TrainingConfig",
    "WindRose",
    "WindRoseEvalConfig",
    "batch_normalise_reward",
    "build_loss_module",
    "build_optimiser",
    "prepare_wind_rose",
]
