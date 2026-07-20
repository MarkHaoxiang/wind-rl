"""Capture an eval episode into a typed, JSON-serialisable replay trajectory.

The recorder steps the eval env manually (rather than :meth:`EnvBase.rollout`)
so that after each ``env.step`` it can read the *live* FLORIS interface for that
step -- yaw command, per-turbine power, free-stream wind -- and, optionally, a
downsampled hub-height flow-speed snapshot. The resulting :class:`ReplayTrajectory`
round-trips through JSON and is consumed by :func:`wind_rl.viz.player.build_replay_html`.

Angle conventions match :mod:`wind_rl.env.render` (meteorological, degrees):
``wind_dir`` is the bearing the wind blows *from*; the air flows towards
``(-sin phi, -cos phi)`` in map ``(x, y)``. A turbine's nacelle axis is the upwind
unit turned by its yaw offset: ``(sin(phi - yaw*pi/180), cos(...))``.

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
    fi = designable.floris
    interface = designable.mdp.interface
    scenario = designable.scenario
    hub_height = float(np.asarray(fi.floris.farm.hub_heights).reshape(-1)[0])
    rotor_diameter = float(np.asarray(fi.floris.farm.rotor_diameters).reshape(-1)[0])
    n_turbines = int(scenario.n_turbines)

    layout0 = np.column_stack([np.asarray(fi.layout_x), np.asarray(fi.layout_y)])

    yaw: list[list[float]] = []
    power_mw: list[list[float]] = []
    wind_speed: list[float] = []
    wind_dir: list[float] = []
    reward: list[float] = []
    cumulative: list[float] = []
    flow_steps: list[int] = []
    flow_grids: list[NDArray[np.float64]] = []

    action_key = env.action_key
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.reset()
        for step in range(scenario.max_steps):
            td = policy(td)
            td = env.step(td)

            step_yaw = np.asarray(interface.get_yaw_command(), dtype=float).reshape(
                n_turbines
            )
            step_power = np.asarray(interface.avg_powers(), dtype=float).reshape(
                n_turbines
            )
            ws = float(np.asarray(interface.wind_speed).reshape(-1)[0])
            wd = float(np.asarray(interface.wind_dir).reshape(-1)[0])

            yaw.append(step_yaw.tolist())
            power_mw.append((step_power / 1e6).tolist())
            wind_speed.append(ws)
            wind_dir.append(wd)
            reward.append(float(td[_REWARD_KEY].mean()))
            cumulative.append(float(td[_EPISODE_REWARD_KEY].mean()))

            if cfg.capture_flow and step % cfg.flow_every == 0:
                grid = _sample_speed_grid(
                    fi,
                    yaw=step_yaw,
                    hub_height=hub_height,
                    bounds=(scenario.map_x_length, scenario.map_y_length),
                    wind_speed=ws,
                    resolution=cfg.flow_size,
                )
                flow_steps.append(step)
                flow_grids.append(grid)

            if bool(td[_DONE_KEY].any()):
                break
            td = step_mdp(td)

    _ = action_key  # env.step consumes the action written by policy under this key
    flow = _pack_flow(flow_grids, flow_steps, cfg.flow_size) if flow_grids else None
    return ReplayTrajectory(
        static=ReplayStatic(
            map_x=float(scenario.map_x_length),
            map_y=float(scenario.map_y_length),
            n_turbines=n_turbines,
            min_distance=float(scenario.min_distance_between_turbines),
            rotor_diameter=rotor_diameter,
            layout=layout0.astype(float).tolist(),
        ),
        yaw=yaw,
        power_mw=power_mw,
        wind_speed=wind_speed,
        wind_dir=wind_dir,
        reward=reward,
        cumulative_reward=cumulative,
        flow=flow,
    )


def _sample_speed_grid(
    fi: object,
    *,
    yaw: NDArray[np.float64],
    hub_height: float,
    bounds: tuple[float, float],
    wind_speed: float,
    resolution: int,
) -> NDArray[np.float64]:
    """Wake-resolved hub-height speed magnitude on a regular ``(res, res)`` grid.

    Rows index map ``y`` (ascending), columns map ``x``. FLORIS samples on a
    wind-aligned (for off-axis wind, rotated) grid, so the scattered speed is
    interpolated onto the regular map grid; points outside the sampled hull fall
    back to the ambient free-stream speed.
    """
    map_x, map_y = bounds
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, map_x, resolution), np.linspace(0.0, map_y, resolution)
    )
    plane = fi.calculate_horizontal_plane(  # type: ignore[attr-defined]
        height=hub_height,
        x_resolution=resolution,
        y_resolution=resolution,
        x_bounds=(0.0, map_x),
        y_bounds=(0.0, map_y),
        yaw_angles=yaw.reshape(1, 1, -1),
    )
    px = np.asarray(plane.df.x1.values, dtype=np.float64)
    py = np.asarray(plane.df.x2.values, dtype=np.float64)
    speed = np.hypot(
        np.asarray(plane.df.u.values, dtype=np.float64),
        np.asarray(plane.df.v.values, dtype=np.float64),
    )

    from matplotlib.tri import LinearTriInterpolator, Triangulation

    tri = Triangulation(px, py)
    interpolated = LinearTriInterpolator(tri, speed)(grid_x, grid_y)
    return np.asarray(np.ma.filled(interpolated, wind_speed), dtype=np.float64)


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
