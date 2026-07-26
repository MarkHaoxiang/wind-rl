import jax
import jax.numpy as jnp

from windrl_engine.env import BatchedWindFarmEnv
from windrl_train.algo.ppo.types import LearnerState, Transition
from windrl_train.features import agent_features


def collect_rollout(
    state: LearnerState, env: BatchedWindFarmEnv, n_steps: int
) -> tuple[LearnerState, Transition]:

    def _env_step(state: LearnerState, _: None) -> tuple[LearnerState, Transition]:
        # Compute Value
        features = agent_features(state.obs)
        value = state.critic(features)

        # Select Action
        key, subkey = jax.random.split(state.key)
        action_dist = state.actor(features)
        raw_action, raw_log_prob = action_dist.sample_and_log_prob(seed=subkey)
        # distrax annotates chex.Array (a union with np.ndarray); narrow to jax.Array
        action, log_prob = jnp.asarray(raw_action), jnp.asarray(raw_log_prob)

        # Step Environment
        batched_step_out = env.step_fn(state=state.env_state, actions=action)

        # Update Learner State
        state_next = state.update_on_env_step(
            batched_step_out=batched_step_out, key=key
        )

        transition = Transition(
            obs=features,
            action=action,
            log_prob=log_prob,
            value=value,
            reward=batched_step_out.reward,
            done=batched_step_out.truncated,
        )

        return state_next, transition

    learner_state, transitions = jax.lax.scan(_env_step, state, xs=None, length=n_steps)
    return learner_state, transitions
