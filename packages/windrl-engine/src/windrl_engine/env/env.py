import functools
from collections.abc import Callable
from typing import Final, NamedTuple, TypedDict, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Key

from windrl_engine.env.actions import YAW_LIMIT, ControlMode, Fidelity, apply_action
from windrl_engine.env.config import WindFarmEnvConfig
from windrl_engine.env.spaces import Box, MultiDiscrete
from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.state import FarmState, make_state
from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec
from windrl_engine.farm.wind import WindCondition, sample_wind
from windrl_engine.physics.power import load_proxies, local_wind, turbine_powers
from windrl_engine.physics.solver import solve_farm

WIND_SPEED_MAX: Final = 28.0  # m/s, matches WFCRL's default wind-speed bound
WIND_DIRECTION_MAX: Final = 360.0  # deg, matches WFCRL's default wind-direction bound


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
    *,
    fidelity: Fidelity = "floris",
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> tuple[FarmState, Observation]:
    wind_key, state_key = jax.random.split(key)
    resolved = sample_wind(wind_key) if wind is None else wind
    state = make_state(layout, resolved, state_key)
    solution = solve_farm(
        layout, resolved, state.yaw, fidelity=fidelity, turbine=turbine
    )
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
    fidelity: Fidelity = "floris",
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> tuple[FarmState, Observation, Float[Array, ""], Bool[Array, ""]]:
    freestream_speed = state.wind.speed
    applied = apply_action(
        state.yaw,
        state.yaw_accumulator,
        state.step_count,
        action,
        yaw_step=yaw_step,
        control_mode=control_mode,
        fidelity=fidelity,
    )
    solution = solve_farm(
        layout, state.wind, applied.yaw, fidelity=fidelity, turbine=turbine
    )
    powers = turbine_powers(solution.u, applied.yaw, turbine=turbine)
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
    fidelity: Fidelity = "floris",
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> tuple[FarmState, Observation, Float[Array, ""], Bool[Array, ""]]:
    return _step_core(
        layout,
        state,
        action,
        yaw_step=yaw_step,
        load_coef=load_coef,
        horizon=horizon,
        control_mode="continuous",
        fidelity=fidelity,
        turbine=turbine,
    )


Actor = Callable[[Key[Array, ""], Observation], Float[Array, "envs turbines"]]


def _where_lane(
    mask: Bool[Array, "envs"], true_value: Array, false_value: Array
) -> Array:
    mask_broadcast = mask.reshape((mask.shape[0],) + (1,) * (true_value.ndim - 1))
    return jnp.where(mask_broadcast, true_value, false_value)


def _select_key(mask: Bool[Array, "envs"], true_key: Array, false_key: Array) -> Array:
    true_key_data, false_key_data = (
        jax.random.key_data(true_key),
        jax.random.key_data(false_key),
    )
    mask_broadcast = mask.reshape((mask.shape[0],) + (1,) * (true_key_data.ndim - 1))
    return cast(
        Array,
        jax.random.wrap_key_data(
            jnp.where(mask_broadcast, true_key_data, false_key_data)
        ),
    )


def _tree_where_lane[T](mask: Bool[Array, "envs"], true_tree: T, false_tree: T) -> T:
    def select_leaf(true_leaf: Array, false_leaf: Array) -> Array:
        if jnp.issubdtype(true_leaf.dtype, jax.dtypes.prng_key):
            return _select_key(mask, true_leaf, false_leaf)
        return _where_lane(mask, true_leaf, false_leaf)

    return cast(T, jax.tree.map(select_leaf, true_tree, false_tree))


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
    turbine: TurbineSpec,
    *,
    yaw_step: float,
    load_coef: float,
    horizon: int,
    control_mode: ControlMode,
    fidelity: Fidelity,
    per_env_layout: bool,
) -> _StepOut:
    # A lane's layout is fixed between explicit resets: it is threaded (not stored
    # in FarmState) so auto-reset resamples wind while keeping the lane's layout.
    layout_axis = 0 if per_env_layout else None

    def step_one_farm(
        layout: FarmLayout, state: FarmState, action: Float[Array, "turbines"]
    ) -> tuple[FarmState, Observation, Float[Array, ""], Bool[Array, ""]]:
        return _step_core(
            layout,
            state,
            action,
            yaw_step=yaw_step,
            load_coef=load_coef,
            horizon=horizon,
            control_mode=control_mode,
            fidelity=fidelity,
            turbine=turbine,
        )

    new_state, obs, reward, truncated = jax.vmap(
        step_one_farm, in_axes=(layout_axis, 0, 0)
    )(layout, state, actions)
    key, reset_key = jax.random.split(key)

    def do_reset(
        operand: tuple[FarmState, Observation],
    ) -> tuple[FarmState, Observation]:
        current_state, current_obs = operand

        def reset_one_farm(
            layout: FarmLayout, key: Key[Array, ""]
        ) -> tuple[FarmState, Observation]:
            return reset(layout, key, fidelity=fidelity, turbine=turbine)

        keys = jax.random.split(reset_key, truncated.shape[0])
        fresh_state, fresh_obs = jax.vmap(reset_one_farm, in_axes=(layout_axis, 0))(
            layout, keys
        )
        return _tree_where_lane(
            truncated, (fresh_state, fresh_obs), (current_state, current_obs)
        )

    def no_reset(
        operand: tuple[FarmState, Observation],
    ) -> tuple[FarmState, Observation]:
        return operand

    reset_state, reset_obs = cast(
        tuple[FarmState, Observation],
        jax.lax.cond(jnp.any(truncated), do_reset, no_reset, (new_state, obs)),
    )
    return _StepOut(reset_state, reset_obs, reward, truncated, key)


def _batched_reset(
    layout: FarmLayout,
    keys: Key[Array, "envs"],
    turbine: TurbineSpec,
    *,
    fidelity: Fidelity,
    per_env_layout: bool,
) -> tuple[FarmState, Observation]:
    layout_axis = 0 if per_env_layout else None

    def reset_one_farm(
        layout: FarmLayout, key: Key[Array, ""]
    ) -> tuple[FarmState, Observation]:
        return reset(layout, key, fidelity=fidelity, turbine=turbine)

    return jax.vmap(reset_one_farm, in_axes=(layout_axis, 0))(layout, keys)


_STEP_STATIC: Final = ("yaw_step", "load_coef", "horizon", "control_mode", "fidelity")


class _StepStatics(TypedDict):
    yaw_step: float
    load_coef: float
    horizon: int
    control_mode: ControlMode
    fidelity: Fidelity


class BatchedWindFarmEnv:
    """A batch of wind farms behind a jointly-stepped parallel API.

    The turbine axis is the multi-agent axis: observations and actions are
    per-turbine with a leading ``(envs, turbines)`` shape, and the scalar
    per-env reward is broadcast per turbine by the consumer. A lane that hits
    its horizon auto-resets on device with freshly sampled wind. ``reset``
    optionally takes per-env ``layouts`` (leading ``(envs,)`` axis), letting
    each lane solve its own fixed layout instead of the shared default.
    """

    def __init__(self, config: WindFarmEnvConfig) -> None:
        self.config = config
        self.layout = config.build_layout()
        self.n_turbines = int(self.layout.x.shape[0])
        self.turbine = config.build_turbine()
        self._reset_jit = jax.jit(
            _batched_reset, static_argnames=("fidelity", "per_env_layout")
        )
        self._step_jit = jax.jit(
            _batched_step, static_argnames=(*_STEP_STATIC, "per_env_layout")
        )
        self._active_layout: FarmLayout = self.layout
        self._per_env_layout: bool = False
        self._state: FarmState | None = None
        self._obs: Observation | None = None
        self._key: Key[Array, ""] = jax.random.key(0)

    def _step_kwargs(self) -> _StepStatics:
        return {
            "yaw_step": self.config.yaw_step,
            "load_coef": self.config.load_coef,
            "horizon": self.config.horizon,
            "control_mode": self.config.control_mode,
            "fidelity": self.config.fidelity,
        }

    def reset(
        self, key: Key[Array, ""], layouts: FarmLayout | None = None
    ) -> Observation:
        """Reset every lane; ``layouts`` (leading ``(envs,)`` axis) gives each lane
        its own layout, else the shared config layout is used for all lanes."""
        if layouts is None:
            self._active_layout, self._per_env_layout = self.layout, False
        else:
            self._active_layout, self._per_env_layout = (
                self._validate_layouts(layouts),
                True,
            )
        key, self._key = jax.random.split(key)
        keys = jax.random.split(key, self.config.n_envs)
        state, obs = cast(
            tuple[FarmState, Observation],
            self._reset_jit(
                self._active_layout,
                keys,
                self.turbine,
                fidelity=self.config.fidelity,
                per_env_layout=self._per_env_layout,
            ),
        )
        self._state, self._obs = state, obs
        return obs

    def _validate_layouts(self, layouts: FarmLayout) -> FarmLayout:
        if layouts.x.shape != (self.config.n_envs, self.n_turbines):
            raise ValueError(
                "per-env layouts must have shape "
                f"(n_envs={self.config.n_envs}, n_turbines={self.n_turbines}), got "
                f"{tuple(layouts.x.shape)}"
            )
        return layouts

    def step(
        self, actions: Float[Array, "envs turbines"]
    ) -> tuple[Observation, Float[Array, "envs"], Bool[Array, "envs"]]:
        if self._state is None:
            raise RuntimeError("call reset before step")
        self._key, step_key = jax.random.split(self._key)
        out = cast(
            _StepOut,
            self._step_jit(
                self._active_layout,
                self._state,
                actions,
                step_key,
                self.turbine,
                per_env_layout=self._per_env_layout,
                **self._step_kwargs(),
            ),
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
                self._active_layout,
                self._state,
                self._obs,
                scan_key,
                n_steps,
                actor,
                self.n_turbines,
                self.turbine,
                per_env_layout=self._per_env_layout,
                **self._step_kwargs(),
            ),
        )
        self._state, self._obs = state, obs
        return rewards

    def action_space(self) -> Box | MultiDiscrete:
        if self.config.control_mode == "continuous":
            return Box((self.n_turbines,), -self.config.yaw_step, self.config.yaw_step)
        return MultiDiscrete((3,) * self.n_turbines)

    def observation_space(self) -> dict[str, Box]:
        return {
            "yaw": Box((self.n_turbines,), -YAW_LIMIT, YAW_LIMIT),
            "freewind": Box(
                (2,),
                jnp.asarray([0.0, 0.0]),
                jnp.asarray([WIND_SPEED_MAX, WIND_DIRECTION_MAX]),
            ),
            "wind_speed": Box((self.n_turbines,), 0.0, WIND_SPEED_MAX),
            "wind_direction": Box((self.n_turbines,), 0.0, WIND_DIRECTION_MAX),
        }


@functools.partial(
    jax.jit,
    static_argnames=(
        *_STEP_STATIC,
        "n_steps",
        "n_turbines",
        "actor",
        "per_env_layout",
    ),
)
def _rollout_core(
    layout: FarmLayout,
    state: FarmState,
    obs: Observation,
    key: Key[Array, ""],
    n_steps: int,
    actor: Actor | None,
    n_turbines: int,
    turbine: TurbineSpec,
    *,
    yaw_step: float,
    load_coef: float,
    horizon: int,
    control_mode: ControlMode,
    fidelity: Fidelity,
    per_env_layout: bool,
) -> tuple[FarmState, Observation, Float[Array, "steps envs"]]:
    n_envs = state.step_count.shape[0]
    idle = 1.0 if control_mode == "discrete" else 0.0
    default_actions = jnp.full((n_envs, n_turbines), idle)

    Carry = tuple[FarmState, Observation, Key[Array, ""], Key[Array, ""]]

    def advance_all_lanes(carry: Carry, _: None) -> tuple[Carry, Float[Array, "envs"]]:
        state, obs, env_key, sample_key = carry
        sample_key, action_key = jax.random.split(sample_key)
        actions = default_actions if actor is None else actor(action_key, obs)
        out = _batched_step(
            layout,
            state,
            actions,
            env_key,
            turbine,
            yaw_step=yaw_step,
            load_coef=load_coef,
            horizon=horizon,
            control_mode=control_mode,
            fidelity=fidelity,
            per_env_layout=per_env_layout,
        )
        return (out.state, out.obs, out.key, sample_key), out.reward

    env_key, sample_key = jax.random.split(key)
    (final_state, final_obs, _, _), rewards = jax.lax.scan(
        advance_all_lanes, (state, obs, env_key, sample_key), None, length=n_steps
    )
    return final_state, final_obs, rewards
