"""BatchedWindFarmEnv: batching, per-lane PRNG, auto-reset, rollout, control modes.

Design doc decision #8 (multi-agent = the turbine axis) and the "env" package
tree entry (`jit(vmap(...))` surface, device-side auto-reset, `lax.scan`
rollout) are the spec for this file; expectations are cross-checked against
the single-farm functional core (`reset`/`step`), never against the batched
implementation's own internals.
"""

import jax
import jax.numpy as jnp
import pytest

from windrl_engine.env.config import WindFarmEnvConfig
from windrl_engine.env.env import BatchedWindFarmEnv, Observation, reset, step
from windrl_engine.env.spaces import MultiDiscrete
from windrl_engine.farm.layout import FarmLayout

_LAYOUT = [(0.0, 0.0), (504.0, 0.0)]


def _lane_reset_keys(seed_key: jax.Array, n_envs: int) -> jax.Array:
    """Replicates `BatchedWindFarmEnv.reset`'s own key split (env/env.py)."""
    internal_key, _ = jax.random.split(seed_key)
    return jax.random.split(internal_key, n_envs)


def _per_env_layouts() -> FarmLayout:
    """Three distinct 2-turbine layouts stacked on a leading (envs,) axis.

    Same turbine count as `_LAYOUT` (config-shared layout) so only the positions
    differ per lane; the downstream turbine's x varies so each lane's wake solve
    -- and hence its local wind / power -- is genuinely lane-specific.
    """
    x = jnp.asarray([[0.0, 504.0], [0.0, 630.0], [0.0, 756.0]])
    y = jnp.zeros((3, 2))
    return FarmLayout(x=x, y=y)


def _lane_layout(layouts: FarmLayout, i: int) -> FarmLayout:
    return FarmLayout(x=layouts.x[i], y=layouts.y[i])


def test_batched_reset_matches_stacked_single_farm_reset_calls() -> None:
    config = WindFarmEnvConfig(layout=_LAYOUT, n_envs=3, horizon=50)
    env = BatchedWindFarmEnv(config)
    key = jax.random.key(7)

    obs = env.reset(key)
    lane_keys = _lane_reset_keys(key, config.n_envs)

    for i, lane_key in enumerate(lane_keys):
        _, expected_obs = reset(env.layout, lane_key)
        assert jnp.allclose(obs.yaw[i], expected_obs.yaw)
        assert jnp.allclose(obs.freewind[i], expected_obs.freewind)
        assert jnp.allclose(obs.wind_speed[i], expected_obs.wind_speed)
        assert jnp.allclose(obs.wind_direction[i], expected_obs.wind_direction)


def test_batched_step_matches_stacked_single_farm_step_calls() -> None:
    # horizon is large enough that no lane truncates, so the comparison isn't
    # complicated by device-side auto-reset (covered separately below).
    config = WindFarmEnvConfig(layout=_LAYOUT, n_envs=3, horizon=50)
    env = BatchedWindFarmEnv(config)
    key = jax.random.key(7)

    env.reset(key)
    lane_states = [
        reset(env.layout, k)[0] for k in _lane_reset_keys(key, config.n_envs)
    ]

    actions = jnp.asarray([[3.0, -2.0], [0.0, 0.0], [5.0, -5.0]])
    obs, reward, truncated = env.step(actions)

    for i, lane_state in enumerate(lane_states):
        _, expected_obs, expected_reward, expected_truncated = step(
            env.layout,
            lane_state,
            actions[i],
            yaw_step=env.yaw_step,
            load_coef=env.load_coef,
            horizon=env.horizon,
        )
        assert jnp.allclose(obs.yaw[i], expected_obs.yaw, atol=1e-9)
        assert jnp.allclose(obs.freewind[i], expected_obs.freewind, atol=1e-9)
        assert jnp.allclose(reward[i], expected_reward, atol=1e-9)
        assert bool(truncated[i]) == bool(expected_truncated)


def test_reset_gives_each_lane_an_independent_wind_draw() -> None:
    config = WindFarmEnvConfig(layout=_LAYOUT, n_envs=4, horizon=50)
    env = BatchedWindFarmEnv(config)
    obs = env.reset(jax.random.key(21))

    for i in range(config.n_envs):
        for j in range(i + 1, config.n_envs):
            assert not jnp.allclose(obs.freewind[i], obs.freewind[j])


def test_lane_that_truncates_gets_a_fresh_reset_observation_on_the_same_call() -> None:
    # horizon=3 with reset's step_count=1 floor means truncation fires on the
    # 2nd agent step; both lanes share one horizon so they truncate together
    # (this env has no mechanism for lanes to desynchronize their step_count).
    config = WindFarmEnvConfig(layout=_LAYOUT, n_envs=2, horizon=3)
    env = BatchedWindFarmEnv(config)
    key = jax.random.key(3)
    env.reset(key)
    actions = jnp.zeros((2, 2))

    obs_before, _, truncated_before = env.step(actions)  # step_count 1 -> 2
    assert bool(jnp.all(~truncated_before))
    assert env._state is not None
    assert jnp.array_equal(env._state.step_count, jnp.asarray([2, 2]))

    key_before_reset_call = env._key
    obs_after, _, truncated_after = env.step(actions)  # step_count 2 -> 3 == horizon
    assert bool(jnp.all(truncated_after))

    # Replicate BatchedWindFarmEnv.step's key derivation (env/env.py) to get
    # the exact reset key `_batched_step` used for the auto-reset this call.
    _, step_key = jax.random.split(key_before_reset_call)
    _, reset_key = jax.random.split(step_key)
    lane_keys = jax.random.split(reset_key, config.n_envs)
    expected = [reset(env.layout, k)[1] for k in lane_keys]

    assert env._state is not None
    assert jnp.array_equal(env._state.step_count, jnp.asarray([1, 1]))
    for i, expected_obs in enumerate(expected):
        assert jnp.array_equal(obs_after.yaw[i], expected_obs.yaw)
        assert jnp.array_equal(obs_after.freewind[i], expected_obs.freewind)
    # The wind was genuinely resampled, not carried over from the truncated episode.
    assert not jnp.allclose(obs_after.freewind, obs_before.freewind)


def test_rollout_matches_a_python_loop_of_step_with_the_same_actor() -> None:
    config = WindFarmEnvConfig(layout=_LAYOUT, n_envs=2, horizon=50)
    key = jax.random.key(11)
    n_steps = 4

    def actor(_key: jax.Array, obs: Observation) -> jax.Array:
        # Deterministic given obs alone (ignores its key argument) so the
        # action sequence can't depend on the scan's internal key genealogy
        # diverging from a manually-managed loop's.
        return jnp.clip(3.0 - 0.2 * obs.yaw, -config.yaw_step, config.yaw_step)

    rollout_env = BatchedWindFarmEnv(config)
    rollout_env.reset(key)
    rewards = rollout_env.rollout(jax.random.key(0), n_steps, actor)

    loop_env = BatchedWindFarmEnv(config)
    obs = loop_env.reset(key)
    manual_rewards = []
    for _ in range(n_steps):
        actions = actor(jax.random.key(0), obs)
        obs, reward, truncated = loop_env.step(actions)
        assert bool(jnp.all(~truncated))
        manual_rewards.append(reward)

    assert rewards.shape == (n_steps, config.n_envs)
    assert jnp.allclose(rewards, jnp.stack(manual_rewards), atol=1e-9)


def test_discrete_control_mode_matches_the_equivalent_continuous_delta_stream() -> None:
    # yaw_step is kept small (<1.8) so the §4a duty-cycle limiter's zeroed-
    # action semantics never trigger: a zeroed *discrete index* maps to
    # -yaw_step (env/actions.py), not 0, so it would break equivalence with
    # the continuous stream's zeroed-*delta*-maps-to-0 semantics if it fired.
    yaw_step = 1.0
    horizon = 50
    discrete_env = BatchedWindFarmEnv(
        WindFarmEnvConfig(
            layout=_LAYOUT,
            control_mode="discrete",
            yaw_step=yaw_step,
            horizon=horizon,
            n_envs=2,
        )
    )
    continuous_env = BatchedWindFarmEnv(
        WindFarmEnvConfig(
            layout=_LAYOUT,
            control_mode="continuous",
            yaw_step=yaw_step,
            horizon=horizon,
            n_envs=2,
        )
    )
    action_space = discrete_env.action_space()
    assert isinstance(action_space, MultiDiscrete)
    assert action_space.nvec == (3, 3)

    key = jax.random.key(5)
    obs_d = discrete_env.reset(key)
    obs_c = continuous_env.reset(key)
    assert jnp.allclose(obs_d.yaw, obs_c.yaw)

    discrete_streams = [
        jnp.asarray([[0.0, 2.0], [1.0, 1.0]]),
        jnp.asarray([[2.0, 0.0], [0.0, 0.0]]),
        jnp.asarray([[1.0, 2.0], [2.0, 1.0]]),
        jnp.asarray([[2.0, 2.0], [0.0, 0.0]]),
        jnp.asarray([[0.0, 0.0], [2.0, 2.0]]),
    ]
    for discrete_actions in discrete_streams:
        continuous_actions = (discrete_actions - 1.0) * yaw_step
        obs_d, reward_d, truncated_d = discrete_env.step(discrete_actions)
        obs_c, reward_c, truncated_c = continuous_env.step(continuous_actions)
        assert jnp.allclose(obs_d.yaw, obs_c.yaw, atol=1e-9)
        assert jnp.allclose(reward_d, reward_c, atol=1e-9)
        assert jnp.array_equal(truncated_d, truncated_c)


# --- Per-env layouts (co-design seam, contract A) -----------------------------


def test_per_env_layouts_make_each_lane_solve_its_own_layout() -> None:
    # Three lanes, three distinct 2-turbine layouts; each lane's first
    # observation and first-step reward must equal the single-farm functional
    # core run on that lane's layout with that lane's own reset key -- exactly,
    # under x64 (the batched path is jit(vmap) of the same per-lane math).
    config = WindFarmEnvConfig(layout=_LAYOUT, n_envs=3, horizon=50)
    env = BatchedWindFarmEnv(config)
    key = jax.random.key(7)
    layouts = _per_env_layouts()

    obs = env.reset(key, layouts)
    lane_keys = _lane_reset_keys(key, config.n_envs)
    lane_states = []
    for i, lane_key in enumerate(lane_keys):
        lane_state, expected_obs = reset(_lane_layout(layouts, i), lane_key)
        lane_states.append(lane_state)
        # Raw wind draw is bitwise-identical; solve-derived fields agree to
        # solver precision (the jit(vmap) path reassociates float64 ops ~1e-8,
        # matching the sibling stacked-singles tests' allclose tolerance).
        assert jnp.array_equal(obs.yaw[i], expected_obs.yaw)
        assert jnp.array_equal(obs.freewind[i], expected_obs.freewind)
        assert jnp.allclose(obs.wind_speed[i], expected_obs.wind_speed)
        assert jnp.allclose(obs.wind_direction[i], expected_obs.wind_direction)

    actions = jnp.asarray([[3.0, -2.0], [1.0, 0.0], [5.0, -5.0]])
    _, reward, _ = env.step(actions)
    for i, lane_state in enumerate(lane_states):
        _, _, expected_reward, _ = step(
            _lane_layout(layouts, i),
            lane_state,
            actions[i],
            yaw_step=env.yaw_step,
            load_coef=env.load_coef,
            horizon=env.horizon,
        )
        assert jnp.allclose(reward[i], expected_reward)


def test_layouts_none_reproduces_the_config_shared_layout_trajectory() -> None:
    # Passing layouts=None is a no-op vs the config-shared layout path: reset and
    # every subsequent step must be bitwise-identical between the two.
    config = WindFarmEnvConfig(layout=_LAYOUT, n_envs=3, horizon=50)
    key = jax.random.key(7)

    shared_env = BatchedWindFarmEnv(config)
    none_env = BatchedWindFarmEnv(config)
    shared_obs = shared_env.reset(key)
    none_obs = none_env.reset(key, None)

    assert jnp.array_equal(shared_obs.yaw, none_obs.yaw)
    assert jnp.array_equal(shared_obs.freewind, none_obs.freewind)
    assert jnp.array_equal(shared_obs.wind_speed, none_obs.wind_speed)

    action_stream = [
        jnp.asarray([[3.0, -2.0], [0.0, 0.0], [5.0, -5.0]]),
        jnp.asarray([[-1.0, 4.0], [2.0, -3.0], [0.0, 1.0]]),
        jnp.asarray([[5.0, 5.0], [-5.0, -5.0], [1.0, -1.0]]),
    ]
    for actions in action_stream:
        s_obs, s_reward, s_trunc = shared_env.step(actions)
        n_obs, n_reward, n_trunc = none_env.step(actions)
        assert jnp.array_equal(s_obs.yaw, n_obs.yaw)
        assert jnp.array_equal(s_obs.wind_speed, n_obs.wind_speed)
        assert jnp.array_equal(s_reward, n_reward)
        assert jnp.array_equal(s_trunc, n_trunc)


def test_per_env_layout_is_fixed_across_auto_reset_while_wind_resamples() -> None:
    # A lane's layout is FIXED across device-side auto-reset (auto-reset resamples
    # wind/state only). Drive both lanes to truncation, then confirm the fresh
    # observation matches the single-farm solve on the *same* per-lane layout with
    # a *new* wind draw, and the stored layout is unchanged.
    config = WindFarmEnvConfig(layout=_LAYOUT, n_envs=2, horizon=3)
    env = BatchedWindFarmEnv(config)
    key = jax.random.key(3)
    layouts = _per_env_layouts()
    layouts = FarmLayout(x=layouts.x[:2], y=layouts.y[:2])
    env.reset(key, layouts)
    actions = jnp.zeros((2, 2))

    obs_before, _, truncated_before = env.step(actions)  # step_count 1 -> 2
    assert bool(jnp.all(~truncated_before))

    key_before_reset_call = env._key
    obs_after, _, truncated_after = env.step(actions)  # 2 -> 3 == horizon
    assert bool(jnp.all(truncated_after))

    # Replicate the exact auto-reset key genealogy (env/env.py) for this call.
    _, step_key = jax.random.split(key_before_reset_call)
    _, reset_key = jax.random.split(step_key)
    lane_keys = jax.random.split(reset_key, config.n_envs)

    for i, lane_key in enumerate(lane_keys):
        _, expected_obs = reset(_lane_layout(layouts, i), lane_key)
        assert jnp.array_equal(obs_after.yaw[i], expected_obs.yaw)
        assert jnp.array_equal(obs_after.freewind[i], expected_obs.freewind)
        assert jnp.allclose(obs_after.wind_speed[i], expected_obs.wind_speed)
    # Wind was genuinely resampled, not carried from the truncated episode.
    assert not jnp.allclose(obs_after.freewind, obs_before.freewind)
    # The layout the env solves is unchanged by the auto-reset.
    assert jnp.array_equal(env._active_layout.x, layouts.x)
    assert jnp.array_equal(env._active_layout.y, layouts.y)


def test_per_env_layout_with_wrong_turbine_count_raises_eagerly() -> None:
    # config layout has 2 turbines; a 3-turbine per-env layout is a mismatch that
    # must be rejected eagerly (host-side), not silently mis-solved.
    config = WindFarmEnvConfig(layout=_LAYOUT, n_envs=2, horizon=50)
    env = BatchedWindFarmEnv(config)
    three_turbines = FarmLayout(
        x=jnp.asarray([[0.0, 504.0, 1008.0], [0.0, 504.0, 1008.0]]),
        y=jnp.zeros((2, 3)),
    )
    with pytest.raises(ValueError):
        env.reset(jax.random.key(0), three_turbines)
