import equinox as eqx
import jax
import jax.numpy as jnp

from windrl_train.nn import MLP, Actor, Critic

ENVS = 3
AGENTS = 4
FEAT = 5


def test_mlp_maps_trailing_axis_of_envs_agents_feat_batch() -> None:
    mlp = MLP(FEAT, 2, width=8, depth=2, key=jax.random.key(0))
    x = jax.random.normal(jax.random.key(1), (ENVS, AGENTS, FEAT))
    out = mlp(x)
    assert out.shape == (ENVS, AGENTS, 2)


def test_mlp_depth_is_the_hidden_layer_count() -> None:
    mlp = MLP(FEAT, 2, width=8, depth=0, key=jax.random.key(2))
    assert len(mlp.weights) == 1  # depth=0: a single linear layer, no hidden


def test_actor_sample_lies_strictly_inside_action_scale() -> None:
    scale = 2.5
    actor = Actor(FEAT, width=8, depth=1, action_scale=scale, key=jax.random.key(3))
    feats = jax.random.normal(jax.random.key(4), (ENVS, AGENTS, FEAT))
    sample = actor(feats).sample(seed=jax.random.key(5))
    assert sample.shape == (ENVS, AGENTS)
    assert jnp.all(jnp.abs(sample) < scale)


def test_actor_log_prob_of_its_own_sample_is_finite() -> None:
    actor = Actor(FEAT, width=8, depth=1, action_scale=1.5, key=jax.random.key(6))
    feats = jax.random.normal(jax.random.key(7), (ENVS, AGENTS, FEAT))
    dist = actor(feats)
    sample = dist.sample(seed=jax.random.key(8))
    log_prob = dist.log_prob(sample)
    assert log_prob.shape == (ENVS, AGENTS)
    assert jnp.all(jnp.isfinite(log_prob))


def test_actor_mode_lies_strictly_inside_action_scale() -> None:
    scale = 0.5
    actor = Actor(FEAT, width=8, depth=1, action_scale=scale, key=jax.random.key(9))
    feats = jax.random.normal(jax.random.key(10), (ENVS, AGENTS, FEAT))
    mode = actor.mode(feats)
    assert mode.shape == (ENVS, AGENTS)
    assert jnp.all(jnp.abs(mode) < scale)


def test_critic_returns_one_value_per_agent() -> None:
    critic = Critic(FEAT, width=8, depth=2, key=jax.random.key(11))
    feats = jax.random.normal(jax.random.key(12), (ENVS, AGENTS, FEAT))
    value = critic(feats)
    assert value.shape == (ENVS, AGENTS)


def test_actor_and_critic_are_non_empty_pytrees() -> None:
    actor = Actor(FEAT, width=8, depth=1, action_scale=1.0, key=jax.random.key(13))
    critic = Critic(FEAT, width=8, depth=1, key=jax.random.key(14))
    assert len(jax.tree.leaves(actor)) > 0
    assert len(jax.tree.leaves(critic)) > 0


def test_critic_call_survives_filter_jit() -> None:
    critic = Critic(FEAT, width=8, depth=1, key=jax.random.key(15))
    feats = jax.random.normal(jax.random.key(16), (ENVS, AGENTS, FEAT))
    jitted = eqx.filter_jit(lambda model, x: model(x))
    assert jnp.allclose(jitted(critic, feats), critic(feats))


def test_actor_mode_survives_filter_jit() -> None:
    actor = Actor(FEAT, width=8, depth=1, action_scale=1.0, key=jax.random.key(17))
    feats = jax.random.normal(jax.random.key(18), (ENVS, AGENTS, FEAT))
    jitted = eqx.filter_jit(lambda model, x: model.mode(x))
    assert jnp.allclose(jitted(actor, feats), actor.mode(feats))
