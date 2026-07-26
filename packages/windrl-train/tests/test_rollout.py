import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from jaxtyping import Array, Float

from windrl_engine.env import BatchedWindFarmEnv, WindFarmEnvConfig
from windrl_train.algo.ppo.featurize import NFEAT
from windrl_train.algo.ppo.types import LearnerState, Transition
from windrl_train.nn import Actor, Critic

# rollout.py is the owner's hand-written learning exercise; this file must
# collect against it without breaking other sessions' `pytest -q` gate while
# it doesn't exist yet, so a missing module skips the file instead of
# failing collection.
rollout = pytest.importorskip("windrl_train.algo.ppo.rollout")

N_ENVS = 2
HORIZON = 8
ACTION_SCALE = 5.0


def _env() -> BatchedWindFarmEnv:
    return BatchedWindFarmEnv(
        WindFarmEnvConfig(layout="turb3_row1", n_envs=N_ENVS, horizon=HORIZON)
    )


def _learner_state(env: BatchedWindFarmEnv, seed: int = 0) -> LearnerState:
    actor = Actor(
        NFEAT, width=8, depth=1, action_scale=ACTION_SCALE, key=jax.random.key(seed)
    )
    critic = Critic(NFEAT, width=8, depth=1, key=jax.random.key(seed + 1))
    actor_optimizer = optax.adam(1e-3)
    critic_optimizer = optax.adam(1e-3)
    env_state, obs = env.reset_fn(jax.random.key(seed + 2))
    return LearnerState(
        actor=actor,
        critic=critic,
        actor_opt=actor_optimizer.init(eqx.filter(actor, eqx.is_array)),
        critic_opt=critic_optimizer.init(eqx.filter(critic, eqx.is_array)),
        env_state=env_state,
        obs=obs,
        key=jax.random.key(seed + 3),
        timestep=jnp.asarray(0),
    )


def _leaves_equal(a: object, b: object) -> bool:
    return all(
        bool(jnp.array_equal(left, right))
        for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


def test_rollout_shapes() -> None:
    env = _env()
    state = _learner_state(env)
    n_steps = 6

    _, traj = rollout.collect_rollout(state, env, n_steps)

    n_turbines = env.n_turbines
    assert traj.obs.shape == (n_steps, N_ENVS, n_turbines, NFEAT)
    assert traj.action.shape == (n_steps, N_ENVS, n_turbines)
    assert traj.log_prob.shape == (n_steps, N_ENVS, n_turbines)
    assert traj.value.shape == (n_steps, N_ENVS, n_turbines)
    assert traj.reward.shape == (n_steps, N_ENVS)
    assert traj.done.shape == (n_steps, N_ENVS)


def test_rollout_advances_learner_state() -> None:
    env = _env()
    state = _learner_state(env)
    n_steps = 6

    new_state, _ = rollout.collect_rollout(state, env, n_steps)

    assert int(new_state.timestep) == int(state.timestep) + n_steps * N_ENVS
    assert not _leaves_equal(state.obs, new_state.obs)
    assert not _leaves_equal(state.env_state, new_state.env_state)
    assert not jnp.array_equal(
        jax.random.key_data(state.key), jax.random.key_data(new_state.key)
    )


def test_rollout_leaves_params_untouched() -> None:
    env = _env()
    state = _learner_state(env)

    new_state, _ = rollout.collect_rollout(state, env, 6)

    assert _leaves_equal(state.actor, new_state.actor)
    assert _leaves_equal(state.critic, new_state.critic)
    assert _leaves_equal(state.actor_opt, new_state.actor_opt)
    assert _leaves_equal(state.critic_opt, new_state.critic_opt)


def test_rollout_log_prob_matches_recompute() -> None:
    env = _env()
    state = _learner_state(env)
    n_steps = 6

    _, traj = rollout.collect_rollout(state, env, n_steps)

    for t, lane in ((0, 0), (n_steps - 1, N_ENVS - 1), (n_steps // 2, 0)):
        feats: Float[Array, "agents feat"] = traj.obs[t, lane]
        recomputed_log_prob = state.actor(feats).log_prob(traj.action[t, lane])
        assert jnp.allclose(recomputed_log_prob, traj.log_prob[t, lane], atol=1e-5)

        recomputed_value = state.critic(feats)
        assert jnp.allclose(recomputed_value, traj.value[t, lane], atol=1e-5)


def test_rollout_crosses_autoreset() -> None:
    env = _env()
    state = _learner_state(env)
    n_steps = 12  # > horizon=8, so every lane truncates and auto-resets once

    _, traj = rollout.collect_rollout(state, env, n_steps)

    assert traj.done.shape == (n_steps, N_ENVS)
    # Every lane starts a fresh episode at reset, so truncation lands on
    # exactly the same step index (horizon - 1) across the whole batch.
    truncation_step = HORIZON - 1
    other_steps = jnp.arange(n_steps) != truncation_step
    assert bool(jnp.all(traj.done[truncation_step]))
    assert not bool(jnp.any(traj.done[other_steps]))


@eqx.filter_jit
def _collect_rollout_jit(
    state: LearnerState, env: BatchedWindFarmEnv, n_steps: int
) -> tuple[LearnerState, Transition]:
    return rollout.collect_rollout(state, env, n_steps)  # type: ignore[no-any-return]


def test_rollout_jits() -> None:
    env = _env()
    n_steps = 6

    jitted_state, jitted_traj = _collect_rollout_jit(_learner_state(env), env, n_steps)
    plain_state, plain_traj = rollout.collect_rollout(_learner_state(env), env, n_steps)

    assert jitted_traj.obs.shape == plain_traj.obs.shape
    assert jitted_traj.action.shape == plain_traj.action.shape
    assert jitted_traj.reward.shape == plain_traj.reward.shape
    assert jitted_traj.done.shape == plain_traj.done.shape
    assert int(jitted_state.timestep) == int(plain_state.timestep)
