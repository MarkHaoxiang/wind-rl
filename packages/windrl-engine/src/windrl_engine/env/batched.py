"""A batch of farms stepped jointly: the single-farm core under ``jit(vmap)``.

Every array here carries a leading ``(envs,)`` axis, so the shape aliases are
the batched ones; the un-batched ``single_farm`` aliases apply only inside the
vmapped inner functions.
"""

from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Key, PRNGKeyArray

from windrl_engine.env.actions import YAW_LIMIT, Fidelity
from windrl_engine.env.config import WindFarmEnvConfig
from windrl_engine.env.reward import RewardFn, WfcrlReward
from windrl_engine.env.single_farm import (
    EnvParams,
    Observation,
    StepOut,
    auto_reset,
    reset,
    step,
)
from windrl_engine.env.spaces import Box, MultiDiscrete
from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.state import FarmState
from windrl_engine.farm.turbine import TurbineSpec
from windrl_engine.farm.wind import WIND_DIRECTION_MAX, WIND_SPEED_MAX

Actor = Callable[[PRNGKeyArray, Observation], Float[Array, "envs turbines"]]


class EnvState(NamedTuple):
    """Everything ``batched_step`` needs to advance a batch of envs, all leaves
    batched over a leading ``(envs,)`` axis.

    ``layout`` rides in the state rather than being resampled: device-side
    auto-reset redraws wind only, so a ``lax.scan`` carrying this tuple keeps
    each env's layout across episode boundaries.
    """

    farm: FarmState
    layout: FarmLayout  # every leaf carries the leading ``(envs,)`` axis


class StepExtras(NamedTuple):
    """Per-step quantities the next action does not need, but a learner or a
    replay viewer does."""

    #: The observation of the state the action actually produced. It differs
    #: from the returned observation exactly on a truncating env, whose
    #: observation has already been replaced by its auto-reset — so this is what
    #: bootstraps V(s_T) under a time limit, and the episode's true final frame.
    terminal_obs: Observation
    powers: Float[Array, "envs turbines"]  # watts, at the yaw this step applied


class BatchedStepOut(NamedTuple):
    state: EnvState
    obs: Observation
    reward: Float[Array, "envs"]
    truncated: Bool[Array, "envs"]
    extras: StepExtras


def batched_step(
    state: EnvState,
    actions: Float[Array, "envs turbines"],
    turbine: TurbineSpec,
    params: EnvParams,
) -> BatchedStepOut:
    """Advance every env one step, auto-resetting the envs that hit the horizon.

    Deterministic in its arguments: an env redraws its wind from its own key in
    ``state``, and keeps the layout held there. ``params`` is jit-static.
    """

    def step_one_farm(
        layout: FarmLayout, farm: FarmState, action: Float[Array, "turbines"]
    ) -> StepOut:
        return step(layout, farm, action, params, turbine=turbine)

    def continue_one_farm(
        layout: FarmLayout, out: StepOut
    ) -> tuple[FarmState, Observation]:
        return auto_reset(layout, out, params, turbine=turbine)

    stepped = jax.vmap(step_one_farm)(state.layout, state.farm, actions)
    # The batch shares one horizon, so its envs truncate together: gating on
    # ``any`` skips a whole farm solve per env on the steps in between.
    farm, obs = jax.lax.cond(
        jnp.any(stepped.truncated),
        lambda: jax.vmap(continue_one_farm)(state.layout, stepped),
        lambda: (stepped.state, stepped.obs),
    )
    return BatchedStepOut(
        state=EnvState(farm=farm, layout=state.layout),
        obs=obs,
        reward=stepped.reward,
        truncated=stepped.truncated,
        extras=StepExtras(terminal_obs=stepped.obs, powers=stepped.powers),
    )


def batched_reset(
    layout: FarmLayout,
    keys: Key[Array, "envs"],
    turbine: TurbineSpec,
    *,
    fidelity: Fidelity,
) -> tuple[EnvState, Observation]:
    """Reset every env on its own key; ``layout`` carries a leading ``(envs,)`` axis."""

    def reset_one_farm(
        layout: FarmLayout, key: PRNGKeyArray
    ) -> tuple[FarmState, Observation]:
        return reset(layout, key, fidelity=fidelity, turbine=turbine)

    farm, obs = jax.vmap(reset_one_farm)(layout, keys)
    return EnvState(farm=farm, layout=layout), obs


def _scan_rollout(
    state: EnvState,
    obs: Observation,
    key: PRNGKeyArray,
    turbine: TurbineSpec,
    params: EnvParams,
    *,
    n_steps: int,
    actor: Actor | None,
) -> tuple[EnvState, Observation, Float[Array, "steps envs"]]:
    idle = 1.0 if params.control_mode == "discrete" else 0.0
    default_actions = jnp.full(state.layout.x.shape, idle)

    Carry = tuple[EnvState, Observation, PRNGKeyArray]

    def advance_all_envs(
        carry: Carry, _step: None
    ) -> tuple[Carry, Float[Array, "envs"]]:
        state, obs, sample_key = carry
        sample_key, action_key = jax.random.split(sample_key)
        actions = default_actions if actor is None else actor(action_key, obs)
        out = batched_step(state, actions, turbine, params)
        return (out.state, out.obs, sample_key), out.reward

    (final_state, final_obs, _), rewards = jax.lax.scan(
        advance_all_envs, (state, obs, key), None, length=n_steps
    )
    return final_state, final_obs, rewards


# One cache per jitted function, keyed on EnvParams by value: identically
# configured envs share their compiled step instead of retracing per instance.
_batched_reset_jit = jax.jit(batched_reset, static_argnames=("fidelity",))
_batched_step_jit = jax.jit(batched_step, static_argnames=("params",))
_scan_rollout_jit = jax.jit(
    _scan_rollout, static_argnames=("params", "n_steps", "actor")
)


class BatchedWindFarmEnv:
    """A batch of wind farms behind a jointly-stepped parallel API.

    The turbine axis is the multi-agent axis: observations and actions are
    per-turbine with a leading ``(envs, turbines)`` shape, and the scalar
    per-env reward is broadcast per turbine by the consumer. An env that hits
    its horizon auto-resets on device with freshly sampled wind. ``reset``
    optionally takes per-env ``layouts`` (leading ``(envs,)`` axis), letting
    each env solve its own fixed layout instead of the shared default.

    ``reset_fn``/``step_fn`` bind this env's layout, turbine tables and
    :class:`EnvParams` to the pure :func:`batched_reset`/:func:`batched_step`,
    which thread an explicit :class:`EnvState`. The stateful
    ``reset``/``step``/``rollout`` are shells over them for scripts and
    notebooks; a trainer should scan ``step_fn``.
    """

    def __init__(
        self, config: WindFarmEnvConfig, reward_fn: RewardFn | None = None
    ) -> None:
        self.config = config
        self.layout = config.build_layout()
        self.n_turbines = int(self.layout.x.shape[0])
        self.turbine = config.build_turbine()
        self.params = EnvParams(
            yaw_step=config.yaw_step,
            reward_fn=WfcrlReward(config.load_coef) if reward_fn is None else reward_fn,
            horizon=config.horizon,
            control_mode=config.control_mode,
            fidelity=config.fidelity,
        )
        self._state: EnvState | None = None
        self._obs: Observation | None = None

    def reset_fn(
        self, key: PRNGKeyArray, layouts: FarmLayout | None = None
    ) -> tuple[EnvState, Observation]:
        """Reset every env; ``layouts`` (leading ``(envs,)`` axis) gives each env
        its own layout, else the shared config layout is tiled across envs.

        Pure in its arguments — ``self`` contributes only static config and the
        turbine tables — so the returned state can be carried through a scan.
        """
        layout = self._batched_layout(layouts)
        keys = jax.random.split(key, self.config.n_envs)
        return cast(
            tuple[EnvState, Observation],
            _batched_reset_jit(
                layout, keys, self.turbine, fidelity=self.params.fidelity
            ),
        )

    def step_fn(
        self, state: EnvState, actions: Float[Array, "envs turbines"]
    ) -> BatchedStepOut:
        """:func:`batched_step` under this env's compiled step."""
        return cast(
            BatchedStepOut,
            _batched_step_jit(state, actions, self.turbine, self.params),
        )

    def _batched_layout(self, layouts: FarmLayout | None) -> FarmLayout:
        # step_fn always vmaps the layout over axis 0, so the shared config
        # layout has to be tiled once here rather than broadcast per step.
        if layouts is not None:
            return self._validate_layouts(layouts)
        shape = (self.config.n_envs, self.n_turbines)
        return FarmLayout(
            x=jnp.broadcast_to(self.layout.x, shape),
            y=jnp.broadcast_to(self.layout.y, shape),
        )

    def _validate_layouts(self, layouts: FarmLayout) -> FarmLayout:
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
        self, key: PRNGKeyArray, layouts: FarmLayout | None = None
    ) -> Observation:
        """Stateful ``reset_fn``: stashes the new state and returns the observation."""
        state, obs = self.reset_fn(key, layouts)
        self._state, self._obs = state, obs
        return obs

    def step(
        self, actions: Float[Array, "envs turbines"]
    ) -> tuple[Observation, Float[Array, "envs"], Bool[Array, "envs"], StepExtras]:
        """Stateful ``step_fn``: advances the stashed state, which ``reset`` must
        have created."""
        if self._state is None:
            raise RuntimeError("call reset before step")
        out = self.step_fn(self._state, actions)
        self._state, self._obs = out.state, out.obs
        return out.obs, out.reward, out.truncated, out.extras

    def rollout(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        actor: Actor | None = None,
    ) -> Float[Array, "steps envs"]:
        """Advance every env ``n_steps`` steps as one fused ``lax.scan``.

        ``key`` seeds the per-step action keys; ``actor`` maps
        ``(action key, observation) -> (envs, turbines)`` actions and runs inside
        the scan, so it must be traceable; ``None`` is a do-nothing policy (zero
        delta / discrete no-change). Returns per-step rewards and leaves the env
        at the final state.
        """
        if self._state is None or self._obs is None:
            raise RuntimeError("call reset before rollout")
        state, obs, rewards = cast(
            tuple[EnvState, Observation, Float[Array, "steps envs"]],
            _scan_rollout_jit(
                self._state,
                self._obs,
                key,
                self.turbine,
                self.params,
                n_steps=n_steps,
                actor=actor,
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
