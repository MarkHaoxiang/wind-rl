import functools
from collections.abc import Callable
from typing import Final, NamedTuple, TypedDict, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Key

from windrl_engine.env.actions import YAW_LIMIT, ControlMode, apply_action
from windrl_engine.env.config import WindFarmEnvConfig
from windrl_engine.env.spaces import Box, MultiDiscrete
from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.state import FarmState, make_state
from windrl_engine.farm.wind import WindCondition, sample_wind
from windrl_engine.physics.power import load_proxies, local_wind, turbine_powers
from windrl_engine.physics.solver import solve_farm

WIND_SPEED_MAX: Final = 28.0  # DEFAULT_BOUNDS["wind_speed"]
WIND_DIRECTION_MAX: Final = 360.0  # DEFAULT_BOUNDS["wind_direction"]


class Observation(NamedTuple):
    yaw: Float[Array, "turbines"]
    freewind: Float[Array, "2"]  # [speed, direction]
    wind_speed: Float[Array, "turbines"]  # local ∛(mean u³)
    wind_direction: Float[Array, "turbines"]


def _observation(
    yaw: Float[Array, "turbines"],
    wind: WindCondition,
    local_speed: Float[Array, "turbines"],
    local_direction: Float[Array, "turbines"],
) -> Observation:
    freewind = jnp.stack(
        (
            jnp.clip(wind.speed, 0.0, WIND_SPEED_MAX),
            jnp.clip(wind.direction, 0.0, WIND_DIRECTION_MAX),
        )
    )
    return Observation(
        yaw=jnp.clip(yaw, -YAW_LIMIT, YAW_LIMIT),
        freewind=freewind,
        wind_speed=jnp.clip(local_speed, 0.0, WIND_SPEED_MAX),
        wind_direction=jnp.clip(local_direction, 0.0, WIND_DIRECTION_MAX),
    )


def reset(
    layout: FarmLayout,
    key: Key[Array, ""],
    wind: WindCondition | None = None,
) -> tuple[FarmState, Observation]:
    wind_key, state_key = jax.random.split(key)
    resolved = sample_wind(wind_key) if wind is None else wind
    state = make_state(layout, resolved, state_key)
    solution = solve_farm(layout, resolved, state.yaw)
    speed, direction = local_wind(solution, resolved)
    return state, _observation(state.yaw, resolved, speed, direction)


def _reward(
    powers_watts: Float[Array, "turbines"],
    loads: Float[Array, "turbines 4"],
    freestream_speed: Float[Array, ""],
    load_coef: float,
) -> Float[Array, ""]:
    powers_mw = powers_watts / 1e6
    normalized = powers_mw * 1e3 / freestream_speed**3
    load_penalty = jnp.mean(jnp.abs(loads))
    return jnp.mean(normalized) - load_coef * load_penalty


def _step_core(
    layout: FarmLayout,
    state: FarmState,
    action: Float[Array, "turbines"],
    *,
    yaw_step: float,
    load_coef: float,
    horizon: int,
    control_mode: ControlMode,
) -> tuple[FarmState, Observation, Float[Array, ""], Bool[Array, ""]]:
    freestream_speed = state.wind.speed
    applied = apply_action(
        state.yaw,
        state.yaw_accumulator,
        state.step_count,
        action,
        yaw_step=yaw_step,
        control_mode=control_mode,
    )
    solution = solve_farm(layout, state.wind, applied.yaw)
    powers = turbine_powers(solution.u, applied.yaw)
    loads = load_proxies(solution)
    speed, direction = local_wind(solution, state.wind)

    step_count = state.step_count + 1
    new_state = state._replace(
        yaw=applied.yaw,
        yaw_accumulator=applied.accumulator,
        step_count=step_count,
    )
    obs = _observation(applied.yaw, state.wind, speed, direction)
    reward = _reward(powers, loads, freestream_speed, load_coef)
    truncated = step_count == horizon
    return new_state, obs, reward, truncated


def step(
    layout: FarmLayout,
    state: FarmState,
    action: Float[Array, "turbines"],
    *,
    yaw_step: float,
    load_coef: float,
    horizon: int,
) -> tuple[FarmState, Observation, Float[Array, ""], Bool[Array, ""]]:
    return _step_core(
        layout,
        state,
        action,
        yaw_step=yaw_step,
        load_coef=load_coef,
        horizon=horizon,
        control_mode="continuous",
    )


Actor = Callable[[Key[Array, ""], Observation], Float[Array, "envs turbines"]]


def _where_lane(mask: Bool[Array, "envs"], a: Array, b: Array) -> Array:
    m = mask.reshape((mask.shape[0],) + (1,) * (a.ndim - 1))
    return jnp.where(m, a, b)


def _select_key(mask: Bool[Array, "envs"], a: Array, b: Array) -> Array:
    ad, bd = jax.random.key_data(a), jax.random.key_data(b)
    m = mask.reshape((mask.shape[0],) + (1,) * (ad.ndim - 1))
    return cast(Array, jax.random.wrap_key_data(jnp.where(m, ad, bd)))


def _tree_where_lane[T](mask: Bool[Array, "envs"], a: T, b: T) -> T:
    def sel(x: Array, y: Array) -> Array:
        if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            return _select_key(mask, x, y)
        return _where_lane(mask, x, y)

    return cast(T, jax.tree.map(sel, a, b))


class _StepOut(NamedTuple):
    state: FarmState
    obs: Observation
    reward: Float[Array, "envs"]
    truncated: Bool[Array, "envs"]
    key: Key[Array, ""]


def _batched_step(
    layout: FarmLayout,
    state: FarmState,
    actions: Float[Array, "envs turbines"],
    key: Key[Array, ""],
    *,
    yaw_step: float,
    load_coef: float,
    horizon: int,
    control_mode: ControlMode,
) -> _StepOut:
    def one(
        st: FarmState, act: Float[Array, "turbines"]
    ) -> tuple[FarmState, Observation, Float[Array, ""], Bool[Array, ""]]:
        return _step_core(
            layout,
            st,
            act,
            yaw_step=yaw_step,
            load_coef=load_coef,
            horizon=horizon,
            control_mode=control_mode,
        )

    new_state, obs, reward, truncated = jax.vmap(one)(state, actions)
    key, reset_key = jax.random.split(key)

    def do_reset(
        operand: tuple[FarmState, Observation],
    ) -> tuple[FarmState, Observation]:
        st, ob = operand
        keys = jax.random.split(reset_key, truncated.shape[0])
        fresh_state, fresh_obs = jax.vmap(lambda k: reset(layout, k))(keys)
        return _tree_where_lane(truncated, (fresh_state, fresh_obs), (st, ob))

    def no_reset(
        operand: tuple[FarmState, Observation],
    ) -> tuple[FarmState, Observation]:
        return operand

    reset_state, reset_obs = cast(
        tuple[FarmState, Observation],
        jax.lax.cond(jnp.any(truncated), do_reset, no_reset, (new_state, obs)),
    )
    return _StepOut(reset_state, reset_obs, reward, truncated, key)


_STEP_STATIC: Final = ("yaw_step", "load_coef", "horizon", "control_mode")


class _StepStatics(TypedDict):
    yaw_step: float
    load_coef: float
    horizon: int
    control_mode: ControlMode


class BatchedWindFarmEnv:
    """A batch of shared-layout wind farms behind a jointly-stepped parallel API.

    Multi-agent view (design #8): the turbine axis *is* the agent axis, so
    observations are per-turbine ``(envs, turbines)`` and the scalar reward is
    broadcast per turbine by the consumer. Terminated lanes auto-reset on device
    with freshly sampled wind.
    """

    def __init__(self, config: WindFarmEnvConfig) -> None:
        self.config = config
        self.layout = config.build_layout()
        self.n_envs = config.n_envs
        self.n_turbines = int(self.layout.x.shape[0])
        self.yaw_step = config.yaw_step
        self.load_coef = config.load_coef
        self.horizon = config.horizon
        self.control_mode = config.control_mode
        self._reset_jit = jax.jit(jax.vmap(reset, in_axes=(None, 0)))
        self._step_jit = jax.jit(
            functools.partial(_batched_step, **self._step_kwargs()),
        )
        self._state: FarmState | None = None
        self._obs: Observation | None = None
        self._key: Key[Array, ""] = jax.random.key(0)

    def _step_kwargs(self) -> _StepStatics:
        return {
            "yaw_step": self.yaw_step,
            "load_coef": self.load_coef,
            "horizon": self.horizon,
            "control_mode": self.control_mode,
        }

    def reset(self, key: Key[Array, ""]) -> Observation:
        key, self._key = jax.random.split(key)
        keys = jax.random.split(key, self.n_envs)
        state, obs = cast(
            tuple[FarmState, Observation], self._reset_jit(self.layout, keys)
        )
        self._state, self._obs = state, obs
        return obs

    def step(
        self, actions: Float[Array, "envs turbines"]
    ) -> tuple[Observation, Float[Array, "envs"], Bool[Array, "envs"]]:
        if self._state is None:
            raise RuntimeError("call reset before step")
        self._key, step_key = jax.random.split(self._key)
        out = cast(
            _StepOut, self._step_jit(self.layout, self._state, actions, step_key)
        )
        self._state, self._obs = out.state, out.obs
        return out.obs, out.reward, out.truncated

    def rollout(
        self,
        key: Key[Array, ""],
        n_steps: int,
        actor: Actor | None = None,
    ) -> Float[Array, "steps envs"]:
        """Advance every lane ``n_steps`` steps as one fused ``lax.scan``.

        ``actor`` maps ``(step key, observation) -> (envs, turbines)`` actions and
        runs inside the scan, so it must be traceable; ``None`` is a do-nothing
        policy (zero delta / discrete no-change). Returns per-step rewards and
        leaves the env at the final state.
        """
        if self._state is None or self._obs is None:
            raise RuntimeError("call reset before rollout")
        self._key, scan_key = jax.random.split(self._key)
        state, obs, rewards = cast(
            tuple[FarmState, Observation, Float[Array, "steps envs"]],
            _rollout_core(
                self.layout,
                self._state,
                self._obs,
                scan_key,
                n_steps,
                actor,
                self.n_turbines,
                **self._step_kwargs(),
            ),
        )
        self._state, self._obs = state, obs
        return rewards

    def action_space(self) -> Box | MultiDiscrete:
        if self.control_mode == "continuous":
            return Box((self.n_turbines,), -self.yaw_step, self.yaw_step)
        return MultiDiscrete((3,) * self.n_turbines)

    def observation_space(self) -> dict[str, Box]:
        return {
            "yaw": Box((self.n_turbines,), -YAW_LIMIT, YAW_LIMIT),
            "freewind": Box((2,), 0.0, WIND_DIRECTION_MAX),
            "wind_speed": Box((self.n_turbines,), 0.0, WIND_SPEED_MAX),
            "wind_direction": Box((self.n_turbines,), 0.0, WIND_DIRECTION_MAX),
        }


@functools.partial(
    jax.jit, static_argnames=(*_STEP_STATIC, "n_steps", "n_turbines", "actor")
)
def _rollout_core(
    layout: FarmLayout,
    state: FarmState,
    obs: Observation,
    key: Key[Array, ""],
    n_steps: int,
    actor: Actor | None,
    n_turbines: int,
    *,
    yaw_step: float,
    load_coef: float,
    horizon: int,
    control_mode: ControlMode,
) -> tuple[FarmState, Observation, Float[Array, "steps envs"]]:
    n_envs = state.step_count.shape[0]
    idle = 1.0 if control_mode == "discrete" else 0.0
    default_actions = jnp.full((n_envs, n_turbines), idle)

    Carry = tuple[FarmState, Observation, Key[Array, ""], Key[Array, ""]]

    def body(carry: Carry, _: None) -> tuple[Carry, Float[Array, "envs"]]:
        st, ob, env_key, sample_key = carry
        sample_key, k_act = jax.random.split(sample_key)
        actions = default_actions if actor is None else actor(k_act, ob)
        out = _batched_step(
            layout,
            st,
            actions,
            env_key,
            yaw_step=yaw_step,
            load_coef=load_coef,
            horizon=horizon,
            control_mode=control_mode,
        )
        return (out.state, out.obs, out.key, sample_key), out.reward

    env_key, sample_key = jax.random.split(key)
    (final_state, final_obs, _, _), rewards = jax.lax.scan(
        body, (state, obs, env_key, sample_key), None, length=n_steps
    )
    return final_state, final_obs, rewards
