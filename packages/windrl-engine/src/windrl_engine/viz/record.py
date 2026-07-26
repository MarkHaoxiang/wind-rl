"""Record one env of a batch into a replayable ``EpisodeRecord``."""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Float, Key

from windrl_engine.env.actions import YAW_LIMIT
from windrl_engine.env.batched import BatchedWindFarmEnv
from windrl_engine.env.single_farm import Observation
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import Fidelity, solve_farm

RecordActor = Callable[[Key[Array, ""], Observation], Float[Array, "envs turbines"]]


class EpisodeRecord(NamedTuple):
    """One env's rollout, everything a replay viewer needs, as host arrays.

    Frame 0 is the reset state; frame ``t`` is the state the ``t``-th action
    produced, before any auto-reset — so ``truncated[t]`` marks an episode's
    true final frame, and its successor is one step into the fresh episode.
    Angles are degrees, powers watts, speeds m/s.
    """

    layout_x: npt.NDArray[np.float32]  # (turbines,) world meters
    layout_y: npt.NDArray[np.float32]
    hub_height: float
    rotor_diameter: float
    yaw_limit: float
    seconds_per_step: float
    fidelity: Fidelity  # the wake model the recorded rewards were computed under

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


def _reset_frame_powers(
    env: BatchedWindFarmEnv, obs: Observation, env_index: int
) -> Float[Array, "turbines"]:
    # Every other frame's powers come free out of the step; only the reset frame
    # has to be solved for, because reset returns the observation alone.
    wind = WindCondition(
        speed=obs.freewind[env_index, 0], direction=obs.freewind[env_index, 1]
    )
    yaw = obs.yaw[env_index]
    solution = solve_farm(
        env.layout, wind, yaw, fidelity=env.params.fidelity, turbine=env.turbine
    )
    return turbine_powers(solution.u, yaw, turbine=env.turbine)


def record_episode(
    env: BatchedWindFarmEnv,
    key: Key[Array, ""],
    n_steps: int,
    actor: RecordActor | None = None,
    *,
    env_index: int = 0,
) -> EpisodeRecord:
    """Roll ``env`` for ``n_steps``, capturing ``env_index`` as an ``EpisodeRecord``.

    ``actor`` maps ``(key, batched observation) -> (envs, turbines)`` actions;
    ``None`` holds yaw. The record has ``n_steps + 1`` frames (reset first).
    """
    key, reset_key = jax.random.split(key)
    obs = env.reset(reset_key)
    n_turbines = env.n_turbines

    yaw_frames = [obs.yaw[env_index]]
    action_frames = [jnp.zeros(n_turbines)]
    power_frames = [_reset_frame_powers(env, obs, env_index)]
    reward_frames = [jnp.asarray(0.0)]
    freestream_speed = [obs.freewind[env_index, 0]]
    freestream_direction = [obs.freewind[env_index, 1]]
    local_speed = [obs.wind_speed[env_index]]
    local_direction = [obs.wind_direction[env_index]]
    truncated_frames = [jnp.asarray(False)]
    step_counts = [1]

    for _ in range(n_steps):
        key, action_key = jax.random.split(key)
        actions = (
            jnp.zeros((env.config.n_envs, n_turbines))
            if actor is None
            else actor(action_key, obs)
        )
        obs, reward, truncated, extras = env.step(actions)
        # The frame is the state the action produced, so it reads the terminal
        # observation: on a truncating step `obs` has already been auto-reset.
        frame_obs = extras.terminal_obs
        yaw_frames.append(frame_obs.yaw[env_index])
        action_frames.append(actions[env_index])
        power_frames.append(extras.powers[env_index])
        reward_frames.append(reward[env_index])
        freestream_speed.append(frame_obs.freewind[env_index, 0])
        freestream_direction.append(frame_obs.freewind[env_index, 1])
        local_speed.append(frame_obs.wind_speed[env_index])
        local_direction.append(frame_obs.wind_direction[env_index])
        truncated_frames.append(truncated[env_index])
        # A truncated frame is an episode's last, so the next one is the first
        # step of a fresh episode (whose reset state already counts as step 1).
        step_counts.append(2 if bool(truncated_frames[-2]) else step_counts[-1] + 1)

    return EpisodeRecord(
        layout_x=np.asarray(env.layout.x, dtype=np.float32),
        layout_y=np.asarray(env.layout.y, dtype=np.float32),
        hub_height=float(env.turbine.hub_height),
        rotor_diameter=float(env.turbine.rotor_diameter),
        yaw_limit=float(YAW_LIMIT),
        seconds_per_step=60.0,  # env.actions.DT: FLORIS interface timestep
        fidelity=env.params.fidelity,
        yaw=np.asarray(jnp.stack(yaw_frames), dtype=np.float32),
        action=np.asarray(jnp.stack(action_frames), dtype=np.float32),
        power=np.asarray(jnp.stack(power_frames), dtype=np.float32),
        reward=np.asarray(jnp.stack(reward_frames), dtype=np.float32),
        wind_speed=np.asarray(jnp.stack(freestream_speed), dtype=np.float32),
        wind_direction=np.asarray(jnp.stack(freestream_direction), dtype=np.float32),
        local_wind_speed=np.asarray(jnp.stack(local_speed), dtype=np.float32),
        local_wind_direction=np.asarray(jnp.stack(local_direction), dtype=np.float32),
        truncated=np.asarray(jnp.stack(truncated_frames), dtype=np.bool_),
        step_count=np.asarray(step_counts, dtype=np.int32),
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
            # records written before fidelity was captured predate any
            # "corrected" env, so they are all reference-fidelity runs.
            fidelity=cast(Fidelity, str(data["fidelity"]))
            if "fidelity" in data
            else "floris",
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
