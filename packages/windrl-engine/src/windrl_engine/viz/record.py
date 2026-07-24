"""Record a single env lane's rollout into a replayable ``EpisodeRecord``.

Numpy-facing (host arrays, ``.npz`` persistence), so it stays outside the
jaxtyping/beartype import hook that guards the single-farm core.
"""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Float, Key

from windrl_engine.env.actions import YAW_LIMIT
from windrl_engine.env.env import BatchedWindFarmEnv, Observation
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import solve_farm

RecordActor = Callable[[Key[Array, ""], Observation], Float[Array, "envs turbines"]]


class EpisodeRecord(NamedTuple):
    """One env lane's rollout, everything a replay viewer needs, as host arrays.

    Frame 0 is the reset state; frames ``1..T-1`` follow each agent step. Angles
    are degrees, powers watts, speeds m/s. ``truncated[t]`` marks a horizon
    boundary whose frame already shows the auto-reset lane.
    """

    layout_x: npt.NDArray[np.float32]  # (turbines,) world meters
    layout_y: npt.NDArray[np.float32]
    hub_height: float
    rotor_diameter: float
    yaw_limit: float
    seconds_per_step: float

    yaw: npt.NDArray[np.float32]  # (frames, turbines) absolute deg
    action: npt.NDArray[np.float32]  # (frames, turbines) raw delta command
    power: npt.NDArray[np.float32]  # (frames, turbines) watts
    reward: npt.NDArray[np.float32]  # (frames,)
    wind_speed: npt.NDArray[np.float32]  # (frames,) freestream
    wind_direction: npt.NDArray[np.float32]  # (frames,) freestream deg
    local_wind_speed: npt.NDArray[np.float32]  # (frames, turbines)
    local_wind_direction: npt.NDArray[np.float32]  # (frames, turbines)
    truncated: npt.NDArray[np.bool_]  # (frames,)
    step_count: npt.NDArray[np.int32]  # (frames,)


def _frame_powers(
    env: BatchedWindFarmEnv,
    wind_speed: Float[Array, "frames"],
    wind_direction: Float[Array, "frames"],
    yaw: Float[Array, "frames turbines"],
) -> Float[Array, "frames turbines"]:
    layout = env.layout
    turbine = env.turbine

    def one_frame(
        speed: Float[Array, ""],
        direction: Float[Array, ""],
        frame_yaw: Float[Array, "turbines"],
    ) -> Float[Array, "turbines"]:
        wind = WindCondition(speed=speed, direction=direction)
        solution = solve_farm(layout, wind, frame_yaw, turbine=turbine)
        return turbine_powers(solution.u, frame_yaw, turbine=turbine)

    return jax.vmap(one_frame)(wind_speed, wind_direction, yaw)


def record_episode(
    env: BatchedWindFarmEnv,
    key: Key[Array, ""],
    n_steps: int,
    actor: RecordActor | None = None,
    *,
    env_index: int = 0,
) -> EpisodeRecord:
    """Roll ``env`` for ``n_steps`` and capture lane ``env_index`` as an ``EpisodeRecord``.

    ``actor`` maps ``(key, batched observation) -> (envs, turbines)`` actions;
    ``None`` holds yaw. The record has ``n_steps + 1`` frames (reset first).
    """
    key, reset_key = jax.random.split(key)
    obs = env.reset(reset_key)
    n_turbines = env.n_turbines

    yaw_frames = [obs.yaw[env_index]]
    action_frames = [jnp.zeros(n_turbines)]
    reward_frames = [jnp.asarray(0.0)]
    freestream_speed = [obs.freewind[env_index, 0]]
    freestream_direction = [obs.freewind[env_index, 1]]
    local_speed = [obs.wind_speed[env_index]]
    local_direction = [obs.wind_direction[env_index]]
    truncated_frames = [jnp.asarray(False)]
    step_count_frames = [jnp.asarray(1)]

    for _ in range(n_steps):
        key, action_key = jax.random.split(key)
        actions = (
            jnp.zeros((env.config.n_envs, n_turbines))
            if actor is None
            else actor(action_key, obs)
        )
        obs, reward, truncated = env.step(actions)
        yaw_frames.append(obs.yaw[env_index])
        action_frames.append(actions[env_index])
        reward_frames.append(reward[env_index])
        freestream_speed.append(obs.freewind[env_index, 0])
        freestream_direction.append(obs.freewind[env_index, 1])
        local_speed.append(obs.wind_speed[env_index])
        local_direction.append(obs.wind_direction[env_index])
        truncated_frames.append(truncated[env_index])
        step_count_frames.append(step_count_frames[-1] + 1)

    yaw = jnp.stack(yaw_frames)
    wind_speed = jnp.stack(freestream_speed)
    wind_direction = jnp.stack(freestream_direction)
    power = _frame_powers(env, wind_speed, wind_direction, yaw)

    return EpisodeRecord(
        layout_x=np.asarray(env.layout.x, dtype=np.float32),
        layout_y=np.asarray(env.layout.y, dtype=np.float32),
        hub_height=float(env.turbine.hub_height),
        rotor_diameter=float(env.turbine.rotor_diameter),
        yaw_limit=float(YAW_LIMIT),
        seconds_per_step=60.0,  # env.actions.DT: FLORIS interface timestep
        yaw=np.asarray(yaw, dtype=np.float32),
        action=np.asarray(jnp.stack(action_frames), dtype=np.float32),
        power=np.asarray(power, dtype=np.float32),
        reward=np.asarray(jnp.stack(reward_frames), dtype=np.float32),
        wind_speed=np.asarray(wind_speed, dtype=np.float32),
        wind_direction=np.asarray(wind_direction, dtype=np.float32),
        local_wind_speed=np.asarray(jnp.stack(local_speed), dtype=np.float32),
        local_wind_direction=np.asarray(jnp.stack(local_direction), dtype=np.float32),
        truncated=np.asarray(jnp.stack(truncated_frames), dtype=np.bool_),
        step_count=np.asarray(jnp.stack(step_count_frames), dtype=np.int32),
    )


def sweeping_actor(yaw_step: float) -> RecordActor:
    """Steer each turbine toward a fixed distinct yaw target, one ``yaw_step`` at a time.

    A demo policy that visibly ramps yaws to spread offsets, then holds — the
    wake field deflects and per-turbine power shifts over the opening frames.
    """

    def actor(key: Key[Array, ""], obs: Observation) -> Float[Array, "envs turbines"]:
        del key
        n_turbines = obs.yaw.shape[-1]
        target = jnp.linspace(-0.75, 0.75, n_turbines) * YAW_LIMIT
        return jnp.clip(target - obs.yaw, -yaw_step, yaw_step)

    return actor


def save_record(record: EpisodeRecord, path: str | Path) -> None:
    np.savez(path, **record._asdict())


def load_record(path: str | Path) -> EpisodeRecord:
    with np.load(path) as data:
        return EpisodeRecord(
            layout_x=data["layout_x"],
            layout_y=data["layout_y"],
            hub_height=float(data["hub_height"]),
            rotor_diameter=float(data["rotor_diameter"]),
            yaw_limit=float(data["yaw_limit"]),
            seconds_per_step=float(data["seconds_per_step"]),
            yaw=data["yaw"],
            action=data["action"],
            power=data["power"],
            reward=data["reward"],
            wind_speed=data["wind_speed"],
            wind_direction=data["wind_direction"],
            local_wind_speed=data["local_wind_speed"],
            local_wind_direction=data["local_wind_direction"],
            truncated=data["truncated"],
            step_count=data["step_count"],
        )
