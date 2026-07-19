"""MLP actor/critic models for the ``turbine`` agent group."""

from __future__ import annotations

from wind_rl.models.mlp import MlpModelConfig, build_mlp_actor_critic

__all__ = ["MlpModelConfig", "build_mlp_actor_critic"]
