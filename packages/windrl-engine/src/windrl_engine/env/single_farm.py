"""One farm's reset/step: the un-batched env core the batched layer vmaps."""

from dataclasses import dataclass
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from windrl_engine.env.actions import YAW_LIMIT, ControlMode, Fidelity, apply_action
from windrl_engine.env.reward import RewardFn
from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.state import FarmState, make_state
from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec
from windrl_engine.farm.wind import (
    WIND_DIRECTION_MAX,
    WIND_SPEED_MAX,
    WindCondition,
    sample_wind,
)
from windrl_engine.physics.power import load_proxies, local_wind, turbine_powers
from windrl_engine.physics.solver import solve_farm


class Observation(NamedTuple):
    yaw: Float[Array, "turbines"]
    freewind: Float[Array, "2"]  # [speed, direction]
    wind_speed: Float[Array, "turbines"]  # local ∛(mean u³)
    wind_direction: Float[Array, "turbines"]


@dataclass(frozen=True, slots=True)
class EnvParams:
    """The env's compile-time knobs, passed as one jit-static argument.

    Hashable and compared by value, so two envs configured alike share a
    compiled step — which requires ``reward_fn`` to compare by value too (a
    fresh closure per env forces a fresh trace).
    """

    yaw_step: float  # deg per unit action
    reward_fn: RewardFn
    horizon: int  # agent steps per episode, counting reset's burn-in step
    control_mode: ControlMode = "continuous"
    fidelity: Fidelity = "floris"


class StepOut(NamedTuple):
    state: FarmState
    obs: Observation
    reward: Float[Array, ""]
    truncated: Bool[Array, ""]
    powers: Float[Array, "turbines"]  # watts, at the yaw this step applied


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
    key: PRNGKeyArray,
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


def step(
    layout: FarmLayout,
    state: FarmState,
    action: Float[Array, "turbines"],
    params: EnvParams,
    *,
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> StepOut:
    freestream_speed = state.wind.speed
    applied = apply_action(
        state.yaw,
        state.yaw_accumulator,
        state.step_count,
        action,
        yaw_step=params.yaw_step,
        control_mode=params.control_mode,
        fidelity=params.fidelity,
    )
    solution = solve_farm(
        layout, state.wind, applied.yaw, fidelity=params.fidelity, turbine=turbine
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
    reward = params.reward_fn(powers, loads, freestream_speed)
    truncated = step_count == params.horizon
    return StepOut(new_state, obs, reward, truncated, powers)


def auto_reset(
    layout: FarmLayout,
    out: StepOut,
    params: EnvParams,
    *,
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> tuple[FarmState, Observation]:
    """``out``'s state and observation, or a fresh episode's if ``out`` truncated.

    The farm keeps its layout across the boundary and redraws only its wind, from
    the key it carries — so the caller need not feed one in, and each farm's
    stream stays its own under ``vmap``.
    """
    fresh = reset(layout, out.state.key, fidelity=params.fidelity, turbine=turbine)
    return cast(
        tuple[FarmState, Observation],
        # jnp.where handles typed PRNG key leaves natively, so no per-dtype branch.
        jax.tree.map(
            lambda new, old: jnp.where(out.truncated, new, old),
            fresh,
            (out.state, out.obs),
        ),
    )
