from typing import NamedTuple

import optax
from jaxtyping import Array, Bool, Float, Int, Key

from windrl_engine.env import EnvState, Observation
from windrl_train.nn import Actor, Critic


class Transition(NamedTuple):
    obs: Float[Array, "envs agents feat"]
    action: Float[Array, "envs agents"]
    log_prob: Float[Array, "envs agents"]
    value: Float[Array, "envs agents"]
    reward: Float[Array, "envs"]  # shared cooperative reward
    done: Bool[Array, "envs"]  # truncation (env auto-resets)


class LearnerState(NamedTuple):
    actor: Actor
    critic: Critic
    actor_opt: optax.OptState
    critic_opt: optax.OptState
    env_state: EnvState
    obs: Observation  # raw env obs (featurized at use)
    key: Key[Array, ""]
    timestep: Int[Array, ""]  # env steps so far (envs * steps)
