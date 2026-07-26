import jax
import jax.numpy as jnp

from windrl_engine.env import BatchedWindFarmEnv, WindFarmEnvConfig
from windrl_train.algo.ppo.featurize import NFEAT
from windrl_train.eval.evaluator import evaluate
from windrl_train.nn import Actor

N_STEPS = 12  # crosses horizon=8 at least once, forcing an auto-reset mid-eval


def _env() -> BatchedWindFarmEnv:
    return BatchedWindFarmEnv(
        WindFarmEnvConfig(layout="turb3_row1", n_envs=2, horizon=8)
    )


def _actor(seed: int) -> Actor:
    return Actor(NFEAT, width=8, depth=1, action_scale=5.0, key=jax.random.key(seed))


def test_evaluate_returns_finite_mean_reward() -> None:
    result = evaluate(_env(), _actor(0), jax.random.key(1), n_steps=N_STEPS)
    assert set(result) == {"eval/mean_reward"}
    value = result["eval/mean_reward"]
    assert value.shape == ()
    assert bool(jnp.isfinite(value))


def test_evaluate_is_deterministic_for_the_same_key() -> None:
    actor = _actor(2)
    key = jax.random.key(3)
    first = evaluate(_env(), actor, key, n_steps=N_STEPS)
    second = evaluate(_env(), actor, key, n_steps=N_STEPS)
    assert first["eval/mean_reward"] == second["eval/mean_reward"]


def test_evaluate_differs_across_actor_params() -> None:
    key = jax.random.key(4)
    result_a = evaluate(_env(), _actor(5), key, n_steps=N_STEPS)
    result_b = evaluate(_env(), _actor(6), key, n_steps=N_STEPS)
    assert result_a["eval/mean_reward"] != result_b["eval/mean_reward"]
