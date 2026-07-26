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

#: A ``FarmLayout`` pytree whose every leaf carries a leading ``(envs,)`` axis.
#: vmap consumes and produces the single-farm class, so the batched form cannot
#: be a distinct runtime type; this alias marks the seams that expect it.
PerEnvLayouts = FarmLayout


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


#: Computes a farm's scalar reward from quantities the step already solved for
#: (no re-solving the wake): per-turbine power and load proxies, plus the
#: freestream speed used to normalize power. Must be jit/vmap-compatible.
RewardFn = Callable[
    [Float[Array, "turbines"], Float[Array, "turbines 4"], Float[Array, ""]],
    Float[Array, ""],
]


def wfcrl_reward(load_coef: float) -> RewardFn:
    """The WFCRL reward: mean normalized power minus ``load_coef`` times mean |load|."""

    def reward_fn(
        powers_watts: Float[Array, "turbines"],
        loads: Float[Array, "turbines 4"],
        freestream_speed: Float[Array, ""],
    ) -> Float[Array, ""]:
        powers_mw = powers_watts / 1e6
        normalized = powers_mw * 1e3 / freestream_speed**3
        load_penalty = jnp.mean(jnp.abs(loads))
        return jnp.mean(normalized) - load_coef * load_penalty

    return reward_fn


def _step_core(
    layout: FarmLayout,
    state: FarmState,
    action: Float[Array, "turbines"],
    *,
    yaw_step: float,
    reward_fn: RewardFn,
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
    reward = reward_fn(powers, loads, freestream_speed)
    truncated = step_count == horizon
    return new_state, obs, reward, truncated


def step(
    layout: FarmLayout,
    state: FarmState,
    action: Float[Array, "turbines"],
    *,
    yaw_step: float,
    reward_fn: RewardFn,
    horizon: int,
    fidelity: Fidelity = "floris",
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> tuple[FarmState, Observation, Float[Array, ""], Bool[Array, ""]]:
    return _step_core(
        layout,
        state,
        action,
        yaw_step=yaw_step,
        reward_fn=reward_fn,
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


def _tree_where_lane[T](mask: Bool[Array, "envs"], true_tree: T, false_tree: T) -> T:
    # jnp.where handles typed PRNG key leaves natively, so no per-dtype branch.
    return cast(
        T,
        jax.tree.map(lambda a, b: _where_lane(mask, a, b), true_tree, false_tree),
    )


class _StepOut(NamedTuple):
    state: FarmState
    obs: Observation
    reward: Float[Array, "envs"]
    truncated: Bool[Array, "envs"]


def _batched_step(
    layout: PerEnvLayouts,
    state: FarmState,
    actions: Float[Array, "envs turbines"],
    key: Key[Array, ""],
    turbine: TurbineSpec,
    *,
    yaw_step: float,
    reward_fn: RewardFn,
    horizon: int,
    control_mode: ControlMode,
    fidelity: Fidelity,
) -> _StepOut:
    def step_one_farm(
        layout: FarmLayout, state: FarmState, action: Float[Array, "turbines"]
    ) -> tuple[FarmState, Observation, Float[Array, ""], Bool[Array, ""]]:
        return _step_core(
            layout,
            state,
            action,
            yaw_step=yaw_step,
            reward_fn=reward_fn,
            horizon=horizon,
            control_mode=control_mode,
            fidelity=fidelity,
            turbine=turbine,
        )

    new_state, obs, reward, truncated = jax.vmap(step_one_farm)(layout, state, actions)
    _, reset_key = jax.random.split(key)

    def do_reset(
        operand: tuple[FarmState, Observation],
    ) -> tuple[FarmState, Observation]:
        current_state, current_obs = operand

        def reset_one_farm(
            layout: FarmLayout, key: Key[Array, ""]
        ) -> tuple[FarmState, Observation]:
            return reset(layout, key, fidelity=fidelity, turbine=turbine)

        keys = jax.random.split(reset_key, truncated.shape[0])
        fresh_state, fresh_obs = jax.vmap(reset_one_farm)(layout, keys)
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
    return _StepOut(reset_state, reset_obs, reward, truncated)


def _batched_reset(
    layout: PerEnvLayouts,
    keys: Key[Array, "envs"],
    turbine: TurbineSpec,
    *,
    fidelity: Fidelity,
) -> tuple[FarmState, Observation]:
    def reset_one_farm(
        layout: FarmLayout, key: Key[Array, ""]
    ) -> tuple[FarmState, Observation]:
        return reset(layout, key, fidelity=fidelity, turbine=turbine)

    return jax.vmap(reset_one_farm)(layout, keys)


_STEP_STATIC: Final = ("yaw_step", "reward_fn", "horizon", "control_mode", "fidelity")


class _StepStatics(TypedDict):
    yaw_step: float
    reward_fn: RewardFn
    horizon: int
    control_mode: ControlMode
    fidelity: Fidelity


class EnvState(NamedTuple):
    """Everything ``step_fn`` needs to advance a batch of lanes, all leaves
    batched over a leading ``(envs,)`` axis.

    ``layout`` rides in the state rather than being resampled: device-side
    auto-reset redraws wind only, so a ``lax.scan`` carrying this tuple keeps
    each lane's layout across episode boundaries.
    """

    farm: FarmState
    layout: PerEnvLayouts


class BatchedWindFarmEnv:
    """A batch of wind farms behind a jointly-stepped parallel API.

    The turbine axis is the multi-agent axis: observations and actions are
    per-turbine with a leading ``(envs, turbines)`` shape, and the scalar
    per-env reward is broadcast per turbine by the consumer. A lane that hits
    its horizon auto-resets on device with freshly sampled wind. ``reset``
    optionally takes per-env ``layouts`` (leading ``(envs,)`` axis), letting
    each lane solve its own fixed layout instead of the shared default.

    ``reset_fn``/``step_fn`` are the pure, ``lax.scan``-safe form of that API —
    they thread an explicit :class:`EnvState` instead of stashing it. The
    stateful ``reset``/``step``/``rollout`` are shells over them for scripts and
    notebooks; a trainer should scan ``step_fn``.
    """

    def __init__(
        self, config: WindFarmEnvConfig, reward_fn: RewardFn | None = None
    ) -> None:
        self.config = config
        self.layout = config.build_layout()
        self.n_turbines = int(self.layout.x.shape[0])
        self.turbine = config.build_turbine()
        self.reward_fn = (
            wfcrl_reward(config.load_coef) if reward_fn is None else reward_fn
        )
        self._reset_jit = jax.jit(_batched_reset, static_argnames=("fidelity",))
        self._step_jit = jax.jit(_batched_step, static_argnames=_STEP_STATIC)
        self._rollout_jit = jax.jit(
            self._scan_rollout, static_argnames=("n_steps", "actor")
        )
        self._state: EnvState | None = None
        self._obs: Observation | None = None
        self._key: Key[Array, ""] = jax.random.key(0)

    def _step_kwargs(self) -> _StepStatics:
        return {
            "yaw_step": self.config.yaw_step,
            "reward_fn": self.reward_fn,
            "horizon": self.config.horizon,
            "control_mode": self.config.control_mode,
            "fidelity": self.config.fidelity,
        }

    def reset_fn(
        self, key: Key[Array, ""], layouts: PerEnvLayouts | None = None
    ) -> tuple[EnvState, Observation]:
        """Reset every lane; ``layouts`` (leading ``(envs,)`` axis) gives each lane
        its own layout, else the shared config layout is tiled across lanes.

        Pure in its arguments — ``self`` contributes only static config and the
        turbine tables — so the returned state can be carried through a scan.
        """
        layout = self._batched_layout(layouts)
        keys = jax.random.split(key, self.config.n_envs)
        farm, obs = cast(
            tuple[FarmState, Observation],
            self._reset_jit(layout, keys, self.turbine, fidelity=self.config.fidelity),
        )
        return EnvState(farm=farm, layout=layout), obs

    def step_fn(
        self,
        state: EnvState,
        actions: Float[Array, "envs turbines"],
        key: Key[Array, ""],
    ) -> tuple[EnvState, Observation, Float[Array, "envs"], Bool[Array, "envs"]]:
        """Advance every lane one step. ``key`` seeds the wind redraw of whichever
        lanes hit their horizon; those lanes keep the layout held in ``state``."""
        out = cast(
            _StepOut,
            self._step_jit(
                state.layout,
                state.farm,
                actions,
                key,
                self.turbine,
                **self._step_kwargs(),
            ),
        )
        return (
            EnvState(farm=out.state, layout=state.layout),
            out.obs,
            out.reward,
            out.truncated,
        )

    def _batched_layout(self, layouts: PerEnvLayouts | None) -> PerEnvLayouts:
        # step_fn always vmaps the layout over axis 0, so the shared config
        # layout has to be tiled once here rather than broadcast per step.
        if layouts is not None:
            return self._validate_layouts(layouts)
        shape = (self.config.n_envs, self.n_turbines)
        return FarmLayout(
            x=jnp.broadcast_to(self.layout.x, shape),
            y=jnp.broadcast_to(self.layout.y, shape),
        )

    def _validate_layouts(self, layouts: PerEnvLayouts) -> PerEnvLayouts:
        expected = (self.config.n_envs, self.n_turbines)
        for field_name, leaf in zip(FarmLayout._fields, layouts, strict=True):
            if leaf.shape != expected:
                raise ValueError(
                    f"per-env layouts: {field_name} must have shape "
                    f"(n_envs={expected[0]}, n_turbines={expected[1]}), got "
                    f"{tuple(leaf.shape)}"
                )
        return layouts

    def reset(
        self, key: Key[Array, ""], layouts: PerEnvLayouts | None = None
    ) -> Observation:
        """Stateful ``reset_fn``: stashes the new state and returns the observation."""
        reset_key, self._key = jax.random.split(key)
        state, obs = self.reset_fn(reset_key, layouts)
        self._state, self._obs = state, obs
        return obs

    def step(
        self, actions: Float[Array, "envs turbines"]
    ) -> tuple[Observation, Float[Array, "envs"], Bool[Array, "envs"]]:
        """Stateful ``step_fn``: advances the stashed state, which ``reset`` must
        have created."""
        if self._state is None:
            raise RuntimeError("call reset before step")
        self._key, step_key = jax.random.split(self._key)
        state, obs, reward, truncated = self.step_fn(self._state, actions, step_key)
        self._state, self._obs = state, obs
        return obs, reward, truncated

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
            tuple[EnvState, Observation, Float[Array, "steps envs"]],
            self._rollout_jit(
                self._state, self._obs, scan_key, n_steps=n_steps, actor=actor
            ),
        )
        self._state, self._obs = state, obs
        return rewards

    def _scan_rollout(
        self,
        state: EnvState,
        obs: Observation,
        key: Key[Array, ""],
        *,
        n_steps: int,
        actor: Actor | None,
    ) -> tuple[EnvState, Observation, Float[Array, "steps envs"]]:
        idle = 1.0 if self.config.control_mode == "discrete" else 0.0
        default_actions = jnp.full((self.config.n_envs, self.n_turbines), idle)

        Carry = tuple[EnvState, Observation, Key[Array, ""], Key[Array, ""]]

        def advance_all_lanes(
            carry: Carry, _step: None
        ) -> tuple[Carry, Float[Array, "envs"]]:
            state, obs, env_key, sample_key = carry
            sample_key, action_key = jax.random.split(sample_key)
            actions = default_actions if actor is None else actor(action_key, obs)
            env_key, step_key = jax.random.split(env_key)
            state, obs, reward, _ = self.step_fn(state, actions, step_key)
            return (state, obs, env_key, sample_key), reward

        env_key, sample_key = jax.random.split(key)
        (final_state, final_obs, _, _), rewards = jax.lax.scan(
            advance_all_lanes, (state, obs, env_key, sample_key), None, length=n_steps
        )
        return final_state, final_obs, rewards

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
