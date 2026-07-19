"""Acceptance tests for the FLORIS co-design environment pipeline (T2).

FLORIS is a fast analytic model, so the simulator-touching tests run in the
default suite (a 3-turbine env builds and rolls out well under a second).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchrl.envs.utils import check_env_specs

pytest.importorskip("wfcrl")

from wind_rl.env import RewardNormalisation, make_env, render_layout
from wind_rl.scenario import ScenarioConfig

pytestmark = pytest.mark.sim

N_TURBINES = 3


@pytest.fixture
def scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="test3",
        n_turbines=N_TURBINES,
        max_steps=20,
        map_x_length=2000.0,
        map_y_length=2000.0,
        min_distance_between_turbines=400.0,
    )


def test_build_env_specs(scenario: ScenarioConfig) -> None:
    """A 3-turbine FLORIS env builds with correct per-agent yaw-control specs."""
    env = make_env("train", scenario)

    action_keys = list(env.full_action_spec.keys(True, True))
    assert ("turbine", "action", "yaw") in action_keys, action_keys

    yaw_spec = env.full_action_spec["turbine", "action", "yaw"]
    assert yaw_spec.shape == torch.Size([N_TURBINES, 1])

    obs_layout = env.observation_spec["turbine", "observation", "layout"]
    assert obs_layout.shape == torch.Size([N_TURBINES, 2])
    obs_yaw = env.observation_spec["turbine", "observation", "yaw"]
    assert obs_yaw.shape[0] == N_TURBINES

    state_layout = env.observation_spec["state", "layout"]
    assert state_layout.shape == torch.Size([N_TURBINES, 2])


def test_reset_rebuilds_mdp_with_layout(scenario: ScenarioConfig) -> None:
    """An explicit layout override lands in the state, and a second, different
    override genuinely rebuilds the MDP with the new coordinates."""
    env = make_env("train", scenario)

    layout_a = np.array([[100.0, 100.0], [900.0, 100.0], [1700.0, 100.0]])
    env.base_env.set_layout_override(layout_a)
    reset_a = env.reset()
    np.testing.assert_allclose(
        reset_a["state", "layout"].numpy(), layout_a.astype(np.float32)
    )

    layout_b = np.array([[200.0, 300.0], [1000.0, 400.0], [1800.0, 900.0]])
    env.base_env.set_layout_override(layout_b)
    reset_b = env.reset()
    np.testing.assert_allclose(
        reset_b["state", "layout"].numpy(), layout_b.astype(np.float32)
    )

    # The per-agent observation should also carry the new per-turbine coords.
    np.testing.assert_allclose(
        reset_b["turbine", "observation", "layout"].numpy(),
        layout_b.astype(np.float32),
    )


def test_env_specs_match_rollout_data(scenario: ScenarioConfig) -> None:
    """The env's declared specs (dtypes, shapes, keys) match actual rollout data."""
    env = make_env("train", scenario)
    check_env_specs(env)


def test_random_rollout_accumulates_reward(scenario: ScenarioConfig) -> None:
    """RewardSum output equals the running sum of per-step rewards."""
    env = make_env("train", scenario)
    env.set_seed(0)

    rollout = env.rollout(12)
    assert rollout.shape[0] == 12

    per_step_reward = rollout["next", "turbine", "reward"]
    episode_reward = rollout["next", "turbine", "episode_reward"]
    running_sum = torch.cumsum(per_step_reward, dim=0)
    torch.testing.assert_close(episode_reward, running_sum)


def test_reward_normalisation_transforms() -> None:
    """RewardNormalisation applies (reward - mean) / std; identity when unset."""
    norm = RewardNormalisation(mean=2.0, std=4.0)
    out = norm._apply_transform(torch.tensor([6.0, 2.0, -2.0]))
    torch.testing.assert_close(out, torch.tensor([1.0, 0.0, -1.0]))

    identity = RewardNormalisation()
    reward = torch.tensor([3.0, -1.0])
    torch.testing.assert_close(identity._apply_transform(reward), reward)


def test_default_layout_infeasible_scenario_raises() -> None:
    """A min distance the default grid cannot satisfy fails fast in make_env."""
    infeasible = ScenarioConfig(
        name="tight",
        n_turbines=N_TURBINES,
        max_steps=10,
        map_x_length=1000.0,
        map_y_length=1000.0,
        min_distance_between_turbines=600.0,
    )
    with pytest.raises(ValueError, match="infeasible"):
        make_env("train", infeasible)


def test_render_layout_returns_rgb(scenario: ScenarioConfig) -> None:
    """Layout render returns an (H, W, 3) uint8 array."""
    layout = np.array([[100.0, 100.0], [900.0, 100.0], [1700.0, 100.0]])
    image = render_layout(layout, scenario)
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
