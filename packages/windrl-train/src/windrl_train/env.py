from functools import cached_property
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from jumanji.env import Environment
from jumanji.specs import Array, BoundedArray, Spec
from jumanji.types import StepType, TimeStep, restart

# Import the wrappers from their leaf modules, not `mava.utils.make_env`: the
# latter pulls in `mava.utils.network_utils`, which has a circular import that
# only resolves when a `mava.systems.*` entrypoint is imported first. These leaf
# modules depend only on jumanji + `mava.types`, so importing this module in
# isolation stays safe.
from mava.types import Observation, ObservationGlobalState
from mava.wrappers.auto_reset_wrapper import AutoResetWrapper
from mava.wrappers.episode_metrics import RecordEpisodeMetrics
from omegaconf import DictConfig, OmegaConf

from windrl_engine.env.actions import YAW_LIMIT
from windrl_engine.env.config import WindFarmEnvConfig
from windrl_engine.env.env import (
    WIND_DIRECTION_MAX,
    WIND_SPEED_MAX,
)
from windrl_engine.env.env import (
    reset as core_reset,
)
from windrl_engine.env.env import (
    step as core_step,
)
from windrl_engine.farm.state import FarmState
from windrl_engine.farm.wind import WindCondition

# Per-agent (per-turbine) feature vector packed into ``agents_view[i]``:
#   [ own_yaw_deg, local_wind_speed, local_wind_direction,
#     freestream_speed, freestream_direction ]
# Own yaw + local wind (∛mean-u³ speed, deg direction) give the turbine its
# controllable state and wake exposure; the two freestream globals are shared.
_AGENT_FEATURES = 5
_FEATURE_LOW = jnp.asarray([-YAW_LIMIT, 0.0, 0.0, 0.0, 0.0])
_FEATURE_HIGH = jnp.asarray(
    [YAW_LIMIT, WIND_SPEED_MAX, WIND_DIRECTION_MAX, WIND_SPEED_MAX, WIND_DIRECTION_MAX]
)


class WindFarm(Environment):
    """Single-farm Mava/Jumanji env; the turbine axis is the agent axis.

    Wraps ``windrl_engine``'s pure functional single-farm ``reset``/``step``
    core. Mava applies its own ``jax.vmap`` over ``num_envs``, so this env is
    unbatched. Continuous per-agent action is a scalar in ``[-1, 1]`` (the
    ``ContinuousActionHead`` tanh range), rescaled here to the engine's
    ``[-yaw_step, +yaw_step]`` delta-yaw box.
    """

    def __init__(self, env_config: DictConfig, add_global_state: bool = False) -> None:
        kwargs = OmegaConf.to_container(env_config.kwargs, resolve=True)
        self._core_config = WindFarmEnvConfig(control_mode="continuous", **kwargs)  # type: ignore[arg-type]
        # Optional constant free-stream wind (wrapper-level, not a WindFarmEnvConfig
        # field): fixes the Scenario-I aligned-wind regime so wake-steering headroom
        # is deterministic. Absent -> the engine samples wind per reset (Scenario II).
        fixed = OmegaConf.select(env_config, "fixed_wind")
        self._fixed_wind = (
            WindCondition(
                speed=jnp.asarray(float(fixed.speed)),
                direction=jnp.asarray(float(fixed.direction)),
            )
            if fixed is not None
            else None
        )
        self.layout = self._core_config.build_layout()
        self.add_global_state = add_global_state
        self.num_agents = int(self.layout.x.shape[0])
        self.time_limit = self._core_config.horizon
        self.action_dim = 1
        self._yaw_step = self._core_config.yaw_step
        self._load_coef = self._core_config.load_coef
        super().__init__()

    def _observation(
        self, obs: Any, state: FarmState
    ) -> Observation | ObservationGlobalState:
        per_agent = jnp.stack([obs.yaw, obs.wind_speed, obs.wind_direction], axis=-1)
        globals_ = jnp.broadcast_to(obs.freewind, (self.num_agents, 2))
        agents_view = jnp.concatenate([per_agent, globals_], axis=-1)
        action_mask = jnp.ones((self.num_agents, self.action_dim), dtype=bool)
        step_count = jnp.repeat(state.step_count, self.num_agents)
        if self.add_global_state:
            global_state = jnp.tile(agents_view.reshape(-1), (self.num_agents, 1))
            return ObservationGlobalState(
                agents_view=agents_view,
                action_mask=action_mask,
                global_state=global_state,
                step_count=step_count,
            )
        return Observation(
            agents_view=agents_view, action_mask=action_mask, step_count=step_count
        )

    def reset(self, key: chex.PRNGKey) -> tuple[FarmState, TimeStep]:
        state, obs = core_reset(self.layout, key, wind=self._fixed_wind)
        timestep = restart(
            self._observation(obs, state),
            shape=(self.num_agents,),
            extras={"env_metrics": {}},
        )
        return state, timestep

    def step(self, state: FarmState, action: chex.Array) -> tuple[FarmState, TimeStep]:
        delta_yaw = cast(jax.Array, action.reshape(self.num_agents) * self._yaw_step)
        new_state, obs, reward, truncated = core_step(
            self.layout,
            state,
            delta_yaw,
            yaw_step=self._yaw_step,
            load_coef=self._load_coef,
            horizon=self.time_limit,
        )
        # A traced array stands in for the StepType enum, as in Mava's own wrappers.
        step_type = cast(
            StepType, jax.lax.select(truncated, StepType.LAST, StepType.MID)
        )
        # Wind-farm episodes only ever end by hitting the horizon (truncation),
        # never a genuine terminal state, so discount is always 1: PPO bootstraps
        # the value at the time limit.
        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.repeat(reward, self.num_agents),
            discount=jnp.ones(self.num_agents),
            observation=self._observation(obs, new_state),
            extras={"env_metrics": {}},
        )
        return new_state, timestep

    @cached_property
    def observation_spec(self) -> Spec:
        agents_view = BoundedArray(
            (self.num_agents, _AGENT_FEATURES),
            float,
            jnp.broadcast_to(_FEATURE_LOW, (self.num_agents, _AGENT_FEATURES)),
            jnp.broadcast_to(_FEATURE_HIGH, (self.num_agents, _AGENT_FEATURES)),
            "agents_view",
        )
        action_mask = BoundedArray(
            (self.num_agents, self.action_dim), bool, False, True, "action_mask"
        )
        step_count = BoundedArray(
            (self.num_agents,), int, 0, self.time_limit, "step_count"
        )
        if self.add_global_state:
            global_state = Array(
                (self.num_agents, self.num_agents * _AGENT_FEATURES),
                float,
                "global_state",
            )
            return Spec(
                ObservationGlobalState,
                "ObservationSpec",
                agents_view=agents_view,
                action_mask=action_mask,
                global_state=global_state,
                step_count=step_count,
            )
        return Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

    @cached_property
    def action_spec(self) -> Spec:
        return BoundedArray(
            (self.num_agents, self.action_dim), float, -1.0, 1.0, "action"
        )

    @cached_property
    def reward_spec(self) -> Array:
        return Array((self.num_agents,), float, "reward")

    @cached_property
    def discount_spec(self) -> BoundedArray:
        return BoundedArray((self.num_agents,), float, 0.0, 1.0, "discount")

    @property
    def unwrapped(self) -> "WindFarm":
        return self


def make_windfarm_envs(
    config: DictConfig, add_global_state: bool = False
) -> tuple[Environment, Environment]:
    # Mirrors mava.utils.make_env.add_extra_wrappers for our case (no GNN, and
    # implicit_agent_id=True disables the AgentID wrapper): the train env
    # auto-resets and records episode metrics; the eval env only records.
    train_env: Environment = WindFarm(config.env, add_global_state=add_global_state)
    eval_env: Environment = WindFarm(config.env, add_global_state=add_global_state)
    train_env = RecordEpisodeMetrics(AutoResetWrapper(train_env))
    eval_env = RecordEpisodeMetrics(eval_env)
    return train_env, eval_env
