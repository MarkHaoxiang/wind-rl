"""MAPPO training: PPO wiring and the concrete trainer."""

from __future__ import annotations

from wind_rl.rl.mappo import (
    PPOConfig,
    batch_normalise_reward,
    build_loss_module,
    build_optimiser,
)
from wind_rl.rl.trainer import LoggingConfig, MappoTrainer, TrainingConfig

__all__ = [
    "LoggingConfig",
    "MappoTrainer",
    "PPOConfig",
    "TrainingConfig",
    "batch_normalise_reward",
    "build_loss_module",
    "build_optimiser",
]
