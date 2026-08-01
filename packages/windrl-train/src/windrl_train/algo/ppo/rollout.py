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
        dist = state.actor(features)
        raw_pre_tanh_action = dist.distribution.sample(seed=subkey)
        raw_action, raw_forward_log_det = dist.bijector.forward_and_log_det(
            raw_pre_tanh_action
        )
        # distrax annotates chex.Array (a union with np.ndarray); narrow to jax.Array
        pre_tanh_action = jnp.asarray(raw_pre_tanh_action)
        action, forward_log_det = (
            jnp.asarray(raw_action),
            jnp.asarray(raw_forward_log_det),
        )
        log_prob = dist.distribution.log_prob(pre_tanh_action) - forward_log_det

        # Step Environment
        batched_step_out = env.step_fn(state=state.env_state, actions=action)
        next_value = state.critic(agent_features(batched_step_out.extras.terminal_obs))

        # Update Learner State
        state_next = state.update_on_env_step(
            batched_step_out=batched_step_out, key=key
        )

        reward = jnp.broadcast_to(batched_step_out.reward[:, None], value.shape)
        done = jnp.broadcast_to(batched_step_out.truncated[:, None], value.shape)

        transition = Transition(
            obs=features,
            pre_tanh_action=pre_tanh_action,
            action=action,
            log_prob=log_prob,
            value=value,
            next_value=next_value,
            reward=reward,
            done=done,
        )

        return state_next, transition

    learner_state, transitions = jax.lax.scan(_env_step, state, xs=None, length=n_steps)
    return learner_state, transitions
