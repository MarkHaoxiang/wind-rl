"""Actor/critic models for the ``turbine`` agent group."""

from __future__ import annotations

from typing import Annotated, assert_never

import torch
from pydantic import Field
from tensordict.nn import TensorDictSequential
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor

from wind_rl.models.gnn import GcnModelConfig, build_gcn_actor_critic
from wind_rl.models.mlp import MlpModelConfig, build_mlp_actor_critic
from wind_rl.scenario import ScenarioConfig

ModelConfig = Annotated[MlpModelConfig | GcnModelConfig, Field(discriminator="kind")]


def build_actor_critic(
    env: EnvBase,
    scenario: ScenarioConfig,
    cfg: MlpModelConfig | GcnModelConfig,
    device: str | torch.device,
) -> tuple[ProbabilisticActor, TensorDictSequential]:
    """Build the (policy, critic) pair for ``cfg``'s model kind."""
    match cfg:
        case GcnModelConfig():
            return build_gcn_actor_critic(env, scenario, cfg, device)
        case MlpModelConfig():
            return build_mlp_actor_critic(env, scenario, cfg, device)
        case _:
            assert_never(cfg)


__all__ = [
    "GcnModelConfig",
    "MlpModelConfig",
    "ModelConfig",
    "build_actor_critic",
    "build_gcn_actor_critic",
    "build_mlp_actor_critic",
]
