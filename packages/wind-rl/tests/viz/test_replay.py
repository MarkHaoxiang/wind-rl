from __future__ import annotations

import base64
import math

import numpy as np
import pytest
import torch
from tensordict import TensorDictBase

from wind_rl.viz import (
    ReplayFlow,
    ReplayStatic,
    ReplayTrajectory,
    build_replay_html,
    load_template,
)
from wind_rl.viz.player import _PLACEHOLDER


def _sample_trajectory() -> ReplayTrajectory:
    size = 4
    frame = base64.b64encode(bytes(range(size * size))).decode("ascii")
    return ReplayTrajectory(
        static=ReplayStatic(
            map_x=2000.0,
            map_y=2000.0,
            n_turbines=2,
            min_distance=400.0,
            rotor_diameter=126.0,
            layout=[[500.0, 1000.0], [1500.0, 1000.0]],
        ),
        yaw=[[0.0, 5.0], [-3.0, 8.0]],
        power_mw=[[1.2, 0.4], [1.1, 0.6]],
        wind_speed=[8.0, 8.0],
        wind_dir=[270.0, 272.0],
        reward=[1.6, 1.7],
        cumulative_reward=[1.6, 3.3],
        flow=ReplayFlow(
            size=size, vmin=0.0, vmax=8.0, steps=[0, 1], frames=[frame, frame]
        ),
    )


def test_template_ships_placeholder_and_canvas() -> None:
    template = load_template()
    assert _PLACEHOLDER in template
    assert "<canvas" in template
    assert 'getContext("2d")' in template


def test_build_replay_html_is_standalone_and_embeds_trajectory() -> None:
    traj = _sample_trajectory()
    html = build_replay_html(traj)

    assert _PLACEHOLDER not in html
    assert "const TRAJECTORY =" in html
    assert traj.model_dump_json() in html  # sample payload has no '<' to escape
    assert 'getContext("2d")' in html
    assert "http://" not in html and "https://" not in html
    assert "src=" not in html and "<link" not in html


@pytest.mark.sim
def test_record_episode_round_trips_with_finite_shapes() -> None:
    from wind_rl.env.factory import make_env
    from wind_rl.scenario import ScenarioConfig
    from wind_rl.viz import RecordConfig, record_episode

    scenario = ScenarioConfig(
        name="smoke3",
        n_turbines=3,
        max_steps=6,
        map_x_length=2000.0,
        map_y_length=2000.0,
        min_distance_between_turbines=400.0,
        fixed_wind_direction=270.0,
        fixed_wind_speed=8.0,
    )
    layout = np.array([[252.0, 1000.0], [756.0, 1000.0], [1260.0, 1000.0]])
    env = make_env("eval", scenario, layout=layout, device="cpu")
    action_key = env.action_key

    def policy(td: TensorDictBase) -> TensorDictBase:
        td.set(action_key, torch.full((scenario.n_turbines, 1), 3.0))
        return td

    traj = record_episode(env, policy, config=RecordConfig(flow_size=32))
    env.close()

    steps = len(traj.yaw)
    assert 0 < steps <= scenario.max_steps
    assert traj.static.n_turbines == 3
    for row in (*traj.yaw, *traj.power_mw):
        assert len(row) == 3
        assert all(math.isfinite(v) for v in row)
    assert len(traj.wind_speed) == steps
    assert all(math.isfinite(v) for v in (*traj.reward, *traj.cumulative_reward))
    # cumulative reward is the running sum of per-step reward
    assert traj.cumulative_reward[-1] == pytest.approx(sum(traj.reward), rel=1e-4)

    assert traj.flow is not None
    assert len(traj.flow.frames) == len(traj.flow.steps)
    assert traj.flow.vmax > traj.flow.vmin
    decoded = base64.b64decode(traj.flow.frames[0])
    assert len(decoded) == traj.flow.size * traj.flow.size

    restored = ReplayTrajectory.model_validate_json(traj.model_dump_json())
    assert restored == traj
