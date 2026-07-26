import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from windrl_engine.env import BatchedWindFarmEnv, EnvState, Observation
from windrl_train.algo.ppo.featurize import agent_features
from windrl_train.nn import Actor


def evaluate(
    env: BatchedWindFarmEnv,
    actor: Actor,
    key: Key[Array, ""],
    n_steps: int,
) -> dict[str, Float[Array, ""]]:
    """Deterministic-policy rollout scored by ``verdict.windowed_delta``.

    A fresh ``reset_fn`` followed by ``n_steps`` of ``actor.mode`` actions (no
    sampling), so a fixed ``key`` reproduces the same result run to run.
    """
    return {"eval/mean_reward": _evaluate_jit(env, actor, key, n_steps)}


@eqx.filter_jit
def _evaluate_jit(
    env: BatchedWindFarmEnv,
    actor: Actor,
    key: Key[Array, ""],
    n_steps: int,
) -> Float[Array, ""]:
    reset_key, scan_key = jax.random.split(key)
    state, obs = env.reset_fn(reset_key)

    Carry = tuple[EnvState, Observation, Key[Array, ""]]

    def advance_one_step(
        carry: Carry, _step: None
    ) -> tuple[Carry, Float[Array, "envs"]]:
        state, obs, scan_key = carry
        scan_key, step_key = jax.random.split(scan_key)
        actions = actor.mode(agent_features(obs))
        state, obs, reward, _ = env.step_fn(state, actions, step_key)
        return (state, obs, scan_key), reward

    _, rewards = jax.lax.scan(
        advance_one_step, (state, obs, scan_key), None, length=n_steps
    )
    return jnp.mean(rewards)
