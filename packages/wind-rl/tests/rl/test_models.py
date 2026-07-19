from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("wfcrl")

from wind_rl.env.factory import make_env
from wind_rl.env.windfarm import GROUP_NAME
from wind_rl.models.mlp import MlpModelConfig, build_mlp_actor_critic
from wind_rl.scenario import ScenarioConfig

pytestmark = pytest.mark.sim

N_TURBINES = 3
_LAYOUT = np.array([[252.0, 1000.0], [756.0, 1000.0], [1260.0, 1000.0]])


@pytest.fixture
def scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="smoke3",
        n_turbines=N_TURBINES,
        max_steps=10,
        map_x_length=2000.0,
        map_y_length=2000.0,
        min_distance_between_turbines=400.0,
    )


def test_policy_actions_within_yaw_bounds(scenario: ScenarioConfig) -> None:
    env = make_env("train", scenario, layout=_LAYOUT)
    policy, _ = build_mlp_actor_critic(env, scenario, MlpModelConfig(), "cpu")

    yaw_spec = env.full_action_spec_unbatched[env.action_key]
    td = policy(env.reset())
    action = td[env.action_key]

    assert action.shape == torch.Size([N_TURBINES, 1])
    assert torch.all(action >= yaw_spec.space.low)
    assert torch.all(action <= yaw_spec.space.high)


def test_critic_state_value_shape(scenario: ScenarioConfig) -> None:
    env = make_env("train", scenario, layout=_LAYOUT)
    _, critic = build_mlp_actor_critic(env, scenario, MlpModelConfig(), "cpu")

    td = critic(env.reset())
    value = td[GROUP_NAME, "state_value"]
    assert value.shape == torch.Size([N_TURBINES, 1])
    assert torch.isfinite(value).all()
