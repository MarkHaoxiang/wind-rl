"""Capture an eval episode into a typed, JSON-serialisable replay trajectory.

The recorder steps the eval env manually (rather than :meth:`EnvBase.rollout`)
so that after each ``env.step`` it can read the *live* FLORIS interface for that
step -- yaw command, per-turbine power, free-stream wind -- and, optionally, a
downsampled hub-height flow-speed snapshot. The resulting :class:`ReplayTrajectory`
round-trips through JSON and is consumed by :func:`wind_rl.viz.player.build_replay_html`.

Coordinate/angle-frame conventions live in :mod:`wind_rl.env.flow`.

Flow snapshots are stored as base64 ``uint8`` grids (row-major, row index = y,
column index = x) linearly quantised over the global ``[vmin, vmax]`` speed range,
keeping a whole episode's JSON small (see :func:`record_episode` for the measured
budget).
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from wind_rl.env.flow import read_farm_state, sample_plane
from wind_rl.static import GROUP_NAME

if TYPE_CHECKING:
    from tensordict.nn import TensorDictModule
    from torchrl.envs import TransformedEnv

_REWARD_KEY = ("next", GROUP_NAME, "reward")
_EPISODE_REWARD_KEY = ("next", GROUP_NAME, "episode_reward")
_DONE_KEY = ("next", GROUP_NAME, "done")


class RecordConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capture_flow: bool = True
    #: Side length (px) of the square flow-speed snapshot grid.
    flow_size: int = 56
    #: Capture a flow snapshot every ``flow_every`` steps (1 = every step).
    flow_every: int = 1


class ReplayStatic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    map_x: float
    map_y: float
    n_turbines: int
    min_distance: float
    rotor_diameter: float
    layout: list[list[float]]


class ReplayFlow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: int
    vmin: float
    vmax: float
    steps: list[int]
    frames: list[str]


class ReplayTrajectory(BaseModel):
    """A full eval episode: static farm layout + per-step telemetry + flow frames."""

    model_config = ConfigDict(extra="forbid")

    static: ReplayStatic
    yaw: list[list[float]]
    power_mw: list[list[float]]
    wind_speed: list[float]
    wind_dir: list[float]
    reward: list[float]
    cumulative_reward: list[float]
    flow: ReplayFlow | None = None


def record_episode(
    env: TransformedEnv,
    policy: TensorDictModule,
    *,
    config: RecordConfig | None = None,
) -> ReplayTrajectory:
    """Roll out one deterministic episode on ``env``, recording it for replay.

    ``env`` must be the eval :class:`~torchrl.envs.TransformedEnv` built by
    :func:`wind_rl.env.factory.make_env`; its live FLORIS interface is read after
    every step, so the env is left at the episode's terminal state on return.
    """
    cfg = config or RecordConfig()
    designable = env.base_env.designable_env
    scenario = designable.scenario
    bounds = (scenario.map_x_length, scenario.map_y_length)
    static = read_farm_state(designable)

    yaw: list[list[float]] = []
    power_mw: list[list[float]] = []
    wind_speed: list[float] = []
    wind_dir: list[float] = []
    reward: list[float] = []
    cumulative: list[float] = []
    flow_steps: list[int] = []
    flow_grids: list[NDArray[np.float64]] = []

    with set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.reset()
        for step in range(scenario.max_steps):
            td = policy(td)
            td = env.step(td)

            state = read_farm_state(designable)
            yaw.append(state.yaw.tolist())
            power_mw.append(state.powers_mw.tolist())
            wind_speed.append(state.wind_speed)
            wind_dir.append(state.wind_dir)
            reward.append(float(td[_REWARD_KEY].mean()))
            cumulative.append(float(td[_EPISODE_REWARD_KEY].mean()))

            if cfg.capture_flow and step % cfg.flow_every == 0:
                _, _, u, v = sample_plane(
                    designable.floris,
                    hub_height=static.hub_height,
                    bounds=bounds,
                    yaw=state.yaw,
                    wind_speed=state.wind_speed,
                    wind_dir=state.wind_dir,
                    resolution=cfg.flow_size,
                    fill="ambient",
                )
                flow_steps.append(step)
                flow_grids.append(np.hypot(u, v))

            if bool(td[_DONE_KEY].any()):
                break
            td = step_mdp(td)

    flow = _pack_flow(flow_grids, flow_steps, cfg.flow_size) if flow_grids else None
    return ReplayTrajectory(
        static=ReplayStatic(
            map_x=float(scenario.map_x_length),
            map_y=float(scenario.map_y_length),
            n_turbines=int(scenario.n_turbines),
            min_distance=float(scenario.min_distance_between_turbines),
            rotor_diameter=static.rotor_diameter,
            layout=static.layout.astype(float).tolist(),
        ),
        yaw=yaw,
        power_mw=power_mw,
        wind_speed=wind_speed,
        wind_dir=wind_dir,
        reward=reward,
        cumulative_reward=cumulative,
        flow=flow,
    )


def _pack_flow(
    grids: list[NDArray[np.float64]], steps: list[int], size: int
) -> ReplayFlow:
    stack = np.stack(grids)
    vmin = float(np.nanmin(stack))
    vmax = float(np.nanmax(stack))
    scale = 255.0 / (vmax - vmin) if vmax > vmin else 0.0
    frames = [
        base64.b64encode(
            np.ascontiguousarray(
                np.clip((grid - vmin) * scale, 0.0, 255.0).astype(np.uint8)
            )
        ).decode("ascii")
        for grid in grids
    ]
    return ReplayFlow(size=size, vmin=vmin, vmax=vmax, steps=steps, frames=frames)
