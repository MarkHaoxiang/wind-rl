import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from jaxtyping import Array, Float

from windrl_engine.env import BatchedWindFarmEnv, WindFarmEnvConfig
from windrl_train.algo.ppo.types import LearnerState, Transition
from windrl_train.features import NFEAT, agent_features
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
    assert traj.pre_tanh_action.shape == (n_steps, N_ENVS, n_turbines)
    assert traj.action.shape == (n_steps, N_ENVS, n_turbines)
    assert traj.log_prob.shape == (n_steps, N_ENVS, n_turbines)
    assert traj.value.shape == (n_steps, N_ENVS, n_turbines)
    assert traj.next_value.shape == (n_steps, N_ENVS, n_turbines)
    assert traj.reward.shape == (n_steps, N_ENVS, n_turbines)
    assert traj.done.shape == (n_steps, N_ENVS, n_turbines)


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

    # Tolerances accommodate TF32 matmuls inside the compiled rollout scan on
    # GPU vs this eager recompute (~1e-4 relative observed); a genuine
    # misalignment is O(1), orders of magnitude larger.
    for t, lane in ((0, 0), (n_steps - 1, N_ENVS - 1), (n_steps // 2, 0)):
        feats: Float[Array, "agents feat"] = traj.obs[t, lane]
        recomputed_log_prob = state.actor(feats).log_prob(traj.action[t, lane])
        assert jnp.allclose(
            recomputed_log_prob, traj.log_prob[t, lane], rtol=1e-3, atol=1e-4
        )

        recomputed_value = state.critic(feats)
        assert jnp.allclose(recomputed_value, traj.value[t, lane], rtol=1e-3, atol=1e-4)

        pre_tanh_action = traj.pre_tanh_action[t, lane]
        recomputed_action = jnp.tanh(pre_tanh_action) * ACTION_SCALE
        assert jnp.allclose(
            traj.action[t, lane], recomputed_action, rtol=1e-3, atol=1e-4
        )

        dist = state.actor(feats)
        forward_log_det = dist.bijector.forward_and_log_det(pre_tanh_action)[1]
        recomputed_log_prob_from_pre_tanh = (
            dist.distribution.log_prob(pre_tanh_action) - forward_log_det
        )
        assert jnp.allclose(
            recomputed_log_prob_from_pre_tanh,
            traj.log_prob[t, lane],
            rtol=1e-3,
            atol=1e-4,
        )


def test_rollout_crosses_autoreset() -> None:
    env = _env()
    state = _learner_state(env)
    n_steps = 12  # > horizon=8, so every lane truncates and auto-resets once

    _, traj = rollout.collect_rollout(state, env, n_steps)

    assert traj.done.shape == (n_steps, N_ENVS, env.n_turbines)
    # Every lane starts a fresh episode at reset, so truncation lands on the
    # same step index across the whole batch. The reset burn-in solve counts
    # as step 1 (WFCRL _num_iter semantics; see the engine's
    # test_step_truncates_on_the_horizon_minus_1th_agent_step), so an episode
    # has horizon - 1 agent steps and truncates at scan index horizon - 2.
    truncation_step = HORIZON - 2
    other_steps = jnp.arange(n_steps) != truncation_step
    assert bool(jnp.all(traj.done[truncation_step]))
    assert not bool(jnp.any(traj.done[other_steps]))


def test_rollout_truncation_next_value_is_pre_reset() -> None:
    env = _env()
    state = _learner_state(env)
    n_steps = 12  # > horizon=8, so every lane truncates and auto-resets once

    _, traj = rollout.collect_rollout(state, env, n_steps)

    # extras.terminal_obs is the pre-reset final frame at a truncation step,
    # not the fresh episode's first observation the buffer's next row holds.
    truncation_step = HORIZON - 2
    assert not bool(
        jnp.allclose(
            traj.next_value[truncation_step],
            traj.value[truncation_step + 1],
            rtol=1e-3,
            atol=1e-4,
        )
    )


def test_rollout_matches_manual_step() -> None:
    env = _env()
    state = _learner_state(env)
    n_steps = 6

    _, traj = rollout.collect_rollout(state, env, n_steps)

    # Tolerances accommodate TF32 matmuls inside the compiled rollout scan on
    # GPU vs this eager recompute (~1e-4 relative observed); a genuine
    # misalignment is O(1), orders of magnitude larger.
    manual_step_out = env.step_fn(state=state.env_state, actions=traj.action[0])
    assert jnp.allclose(
        manual_step_out.reward, traj.reward[0, :, 0], rtol=1e-3, atol=1e-4
    )


def test_rollout_next_value_matches_bootstrap() -> None:
    env = _env()
    state = _learner_state(env)
    n_steps = 6  # last step (index 5) isn't the horizon - 2 truncation step

    new_state, traj = rollout.collect_rollout(state, env, n_steps)

    # Tolerances accommodate TF32 matmuls inside the compiled rollout scan on
    # GPU vs this eager recompute (~1e-4 relative observed); a genuine
    # misalignment is O(1), orders of magnitude larger.
    bootstrap_value = state.critic(agent_features(new_state.obs))
    assert jnp.allclose(bootstrap_value, traj.next_value[-1], rtol=1e-3, atol=1e-4)


def test_rollout_pre_tanh_action_differs_across_steps() -> None:
    env = _env()
    state = _learner_state(env)
    n_steps = 6

    _, traj = rollout.collect_rollout(state, env, n_steps)

    assert not bool(jnp.allclose(traj.pre_tanh_action[0], traj.pre_tanh_action[1]))


def test_rollout_reward_broadcast_at_n_envs_eq_n_turbines() -> None:
    env = BatchedWindFarmEnv(
        WindFarmEnvConfig(layout="turb3_row1", n_envs=3, horizon=HORIZON)
    )
    state = _learner_state(env)
    n_steps = 4

    _, traj = rollout.collect_rollout(state, env, n_steps)

    assert traj.reward.shape == (n_steps, 3, env.n_turbines)
    assert bool(
        jnp.all(
            traj.reward == jnp.broadcast_to(traj.reward[..., :1], traj.reward.shape)
        )
    )
    assert not bool(jnp.all(traj.reward[:, 0, 0] == traj.reward[:, 1, 0]))


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
