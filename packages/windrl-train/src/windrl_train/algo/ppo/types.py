from typing import NamedTuple

import optax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from windrl_engine.env import BatchedStepOut, EnvState, Observation
from windrl_train.nn import Actor, Critic


class Transition(NamedTuple):
    obs: Float[Array, "envs agents feat"]
    pre_tanh_action: Float[Array, "envs agents"]  # base Normal sample, pre Tanh/scale
    action: Float[Array, "envs agents"]
    log_prob: Float[Array, "envs agents"]
    value: Float[Array, "envs agents"]
    next_value: Float[Array, "envs agents"]  # critic on the true successor obs
    reward: Float[Array, "envs agents"]  # cooperative reward, broadcast per agent
    done: Bool[Array, "envs agents"]  # truncation (env auto-resets), per agent


class LearnerState(NamedTuple):
    actor: Actor
    critic: Critic
    actor_opt: optax.OptState
    critic_opt: optax.OptState
    env_state: EnvState
    obs: Observation  # raw env obs (featurized at use)
    key: PRNGKeyArray
    timestep: Int[Array, ""]  # env steps so far (envs * steps)

    def update_on_env_step(
        self, batched_step_out: BatchedStepOut, key: PRNGKeyArray
    ) -> "LearnerState":
        return LearnerState(
            actor=self.actor,
            critic=self.critic,
            actor_opt=self.actor_opt,
            critic_opt=self.critic_opt,
            env_state=batched_step_out.state,
            obs=batched_step_out.obs,
            key=key,
            timestep=self.timestep + batched_step_out.reward.shape[0],
        )
