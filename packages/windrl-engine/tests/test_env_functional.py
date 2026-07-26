"""The public pure functional env API: EnvState + reset_fn/step_fn under lax.scan."""

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from windrl_engine.env.config import WindFarmEnvConfig
from windrl_engine.env.env import BatchedWindFarmEnv, EnvState
from windrl_engine.farm.layout import FarmLayout

_LAYOUT = [(0.0, 0.0), (504.0, 0.0), (1008.0, 0.0)]
_ACTIONS = jnp.asarray([[3.0, -2.0, 1.0], [0.0, 1.0, 0.0], [5.0, -5.0, 2.0]])


@pytest.fixture
def env3() -> BatchedWindFarmEnv:
    return BatchedWindFarmEnv(WindFarmEnvConfig(layout=_LAYOUT, n_envs=3, horizon=50))


def _assert_trees_identical(left: Any, right: Any) -> None:
    leaves = zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    for a, b in leaves:
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        assert jnp.array_equal(a, b)


def test_step_fn_is_pure(env3: BatchedWindFarmEnv) -> None:
    # Same (state, actions, key) twice must give bitwise-identical pytrees, and
    # neither call may touch the stateful shell's stashed state.
    state, _ = env3.reset_fn(jax.random.key(0))
    step_key = jax.random.key(1)

    first = env3.step_fn(state, _ACTIONS, step_key)
    second = env3.step_fn(state, _ACTIONS, step_key)

    _assert_trees_identical(first, second)
    assert env3._state is None
    assert env3._obs is None


def test_scan_matches_stateful() -> None:
    # horizon=6 with reset's step_count=1 floor truncates on the 5th and 10th
    # step, so this covers auto-reset inside the scan too. The functional path
    # replicates the stateful path's key derivation exactly, so the two must
    # agree step for step -- not merely in distribution.
    config = WindFarmEnvConfig(layout=_LAYOUT, n_envs=3, horizon=6)
    env = BatchedWindFarmEnv(config)
    key = jax.random.key(0)
    n_steps = 10

    env.reset(key)
    stateful_rewards, stateful_truncated = [], []
    for _ in range(n_steps):
        _, reward, truncated, _ = env.step(_ACTIONS)
        stateful_rewards.append(reward)
        stateful_truncated.append(truncated)

    reset_key, chain_key = jax.random.split(key)
    state, _ = env.reset_fn(reset_key)
    step_keys = []
    for _ in range(n_steps):
        chain_key, step_key = jax.random.split(chain_key)
        step_keys.append(step_key)

    def scan_step(
        state: EnvState, step_key: jax.Array
    ) -> tuple[EnvState, tuple[jax.Array, jax.Array]]:
        out = env.step_fn(state, _ACTIONS, step_key)
        return out.state, (out.reward, out.truncated)

    _, (rewards, truncated) = jax.lax.scan(scan_step, state, jnp.stack(step_keys))

    assert rewards.shape == (n_steps, config.n_envs)
    assert bool(jnp.any(truncated))
    assert jnp.array_equal(truncated, jnp.stack(stateful_truncated))
    assert jnp.allclose(rewards, jnp.stack(stateful_rewards), atol=1e-9)


def test_reset_fn_tiles_shared_layout(env3: BatchedWindFarmEnv) -> None:
    # step_fn always vmaps the layout over axis 0, so a shared config layout has
    # to arrive from reset_fn already tiled per lane.
    state, _ = env3.reset_fn(jax.random.key(0))

    expected = (env3.config.n_envs, env3.n_turbines)
    assert expected == (3, 3)
    for lane_leaf, shared_leaf in zip(state.layout, env3.layout, strict=True):
        assert lane_leaf.shape == expected
        assert jnp.array_equal(lane_leaf, jnp.broadcast_to(shared_leaf, expected))


def test_per_env_layouts_survive_autoreset() -> None:
    # Auto-reset resamples wind only: the lane's layout rides in EnvState and
    # must come out of a scan that crosses two truncations bit-for-bit unchanged.
    config = WindFarmEnvConfig(layout=_LAYOUT, n_envs=2, horizon=3)
    env = BatchedWindFarmEnv(config)
    layouts = FarmLayout(
        x=jnp.asarray([[0.0, 504.0, 1008.0], [0.0, 630.0, 1260.0]]),
        y=jnp.zeros((2, 3)),
    )
    state, _ = env.reset_fn(jax.random.key(0), layouts)

    def scan_step(state: EnvState, step_key: jax.Array) -> tuple[EnvState, jax.Array]:
        out = env.step_fn(state, jnp.zeros((2, 3)), step_key)
        return out.state, out.truncated

    state, truncated = jax.lax.scan(
        scan_step, state, jax.random.split(jax.random.key(1), 4)
    )

    # step_count 1 -> 2, 2 -> 3 == horizon (truncate, reset to 1), then again.
    assert jnp.array_equal(truncated, jnp.asarray([[False] * 2, [True] * 2] * 2))
    assert jnp.array_equal(state.farm.step_count, jnp.asarray([1, 1]))
    assert jnp.array_equal(state.layout.x, layouts.x)
    assert jnp.array_equal(state.layout.y, layouts.y)
