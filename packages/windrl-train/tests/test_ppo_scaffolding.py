import jax
import jax.numpy as jnp
import optax
import pytest
from pydantic import ValidationError

from windrl_engine.env import BatchedWindFarmEnv, Observation, WindFarmEnvConfig
from windrl_train.algo.ppo.config import IPPOConfig
from windrl_train.algo.ppo.types import LearnerState, Transition
from windrl_train.features import NFEAT, agent_features
from windrl_train.nn import Actor, Critic


def test_ippo_config_builds_with_default_env() -> None:
    config = IPPOConfig(env=WindFarmEnvConfig())
    assert config.rollout_length == 128
    assert config.gamma == 0.99


def test_ippo_config_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        IPPOConfig(env=WindFarmEnvConfig(), totla_timesteps=10)  # typo


def _reset_two_envs() -> tuple[int, Observation]:
    env = BatchedWindFarmEnv(WindFarmEnvConfig(layout="turb3_row1", n_envs=2))
    _, obs = env.reset_fn(jax.random.key(0))
    return env.n_turbines, obs


def test_agent_features_shape_and_bounds_on_real_observation() -> None:
    n_turbines, obs = _reset_two_envs()
    feats = agent_features(obs)
    assert feats.shape == (2, n_turbines, NFEAT)
    assert bool(jnp.all(jnp.isfinite(feats)))
    sin_cos = feats[..., 2:4]
    fw_sin_cos = feats[..., 5:7]
    assert bool(jnp.all(jnp.abs(sin_cos) <= 1.0))
    assert bool(jnp.all(jnp.abs(fw_sin_cos) <= 1.0))


def test_agent_features_broadcasts_freewind_across_agent_axis() -> None:
    _, obs = _reset_two_envs()
    feats = agent_features(obs)
    fw_speed_feat = feats[..., 4]
    # freewind is shared across turbines within an env: every agent sees the
    # same freestream-speed feature.
    assert bool(jnp.all(fw_speed_feat == fw_speed_feat[:, :1]))


def test_transition_constructs_and_is_a_pytree() -> None:
    envs, agents = 2, 3
    transition = Transition(
        obs=jnp.zeros((envs, agents, NFEAT)),
        action=jnp.zeros((envs, agents)),
        log_prob=jnp.zeros((envs, agents)),
        value=jnp.zeros((envs, agents)),
        reward=jnp.zeros((envs,)),
        done=jnp.zeros((envs,), dtype=bool),
    )
    leaves = jax.tree.leaves(transition)
    assert len(leaves) == 6


def test_learner_state_constructs_and_is_a_pytree() -> None:
    env = BatchedWindFarmEnv(WindFarmEnvConfig(layout="turb3_row1", n_envs=2))
    env_state, obs = env.reset_fn(jax.random.key(1))

    actor = Actor(NFEAT, width=8, depth=1, action_scale=5.0, key=jax.random.key(2))
    critic = Critic(NFEAT, width=8, depth=1, key=jax.random.key(3))
    actor_optimizer = optax.adam(3e-4)
    critic_optimizer = optax.adam(3e-4)

    learner_state = LearnerState(
        actor=actor,
        critic=critic,
        actor_opt=actor_optimizer.init(actor),
        critic_opt=critic_optimizer.init(critic),
        env_state=env_state,
        obs=obs,
        key=jax.random.key(4),
        timestep=jnp.asarray(0),
    )
    leaves = jax.tree.leaves(learner_state)
    assert len(leaves) > 0
