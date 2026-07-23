"""Permutation-equivariance checks for the GCN torsos.

Run standalone in the train venv:

    packages/windrl-train/.venv/bin/python packages/windrl-train/tests/test_equivariance.py

or under pytest:

    cd packages/windrl-train && ./.venv/bin/python -m pytest tests/test_equivariance.py
"""

import jax
import jax.numpy as jnp
from mava.networks import FeedForwardActor, FeedForwardValueNet
from mava.networks.heads import ContinuousActionHead
from mava.types import ObservationGlobalState

from windrl_train.networks import GCNGlobalCritic, GCNTorso

_NUM_AGENTS = 6
_NUM_FEATURES = 7
_ATOL = 1e-6


def _observation(agents_view: jax.Array) -> ObservationGlobalState:
    num_agents = agents_view.shape[0]
    return ObservationGlobalState(
        agents_view=agents_view[None],
        action_mask=jnp.ones((1, num_agents, 1), dtype=bool),
        global_state=jnp.tile(agents_view.reshape(-1), (num_agents, 1))[None],
        step_count=jnp.zeros((1, num_agents), dtype=jnp.int32),
    )


def _permute(
    observation: ObservationGlobalState, permutation: jax.Array
) -> ObservationGlobalState:
    return _observation(observation.agents_view[0][permutation])


def _build() -> tuple[
    FeedForwardActor,
    FeedForwardValueNet,
    dict[str, object],
    dict[str, object],
    ObservationGlobalState,
]:
    actor = FeedForwardActor(
        torso=GCNTorso(embed_dim=32),
        action_head=ContinuousActionHead(action_dim=1),
    )
    critic = FeedForwardValueNet(
        torso=GCNGlobalCritic(embed_dim=32, num_features=_NUM_FEATURES),
        centralised_critic=True,
    )
    key = jax.random.PRNGKey(0)
    agents_view = jax.random.normal(key, (_NUM_AGENTS, _NUM_FEATURES))
    observation = _observation(agents_view)
    actor_params = actor.init(jax.random.PRNGKey(1), observation)
    critic_params = critic.init(jax.random.PRNGKey(2), observation)
    return actor, critic, actor_params, critic_params, observation


def test_actor_equivariance_and_critic_invariance() -> None:
    actor, critic, actor_params, critic_params, observation = _build()
    permutation = jax.random.permutation(jax.random.PRNGKey(3), _NUM_AGENTS)
    observation_perm = _permute(observation, permutation)

    mode = actor.apply(actor_params, observation).mode()[0]
    mode_perm = actor.apply(actor_params, observation_perm).mode()[0]
    actor_err = float(jnp.max(jnp.abs(mode[permutation] - mode_perm)))
    assert actor_err < _ATOL, f"actor not equivariant: {actor_err}"

    value = critic.apply(critic_params, observation)[0]
    value_perm = critic.apply(critic_params, observation_perm)[0]
    critic_err = float(jnp.max(jnp.abs(value[permutation] - value_perm)))
    assert critic_err < _ATOL, f"critic not invariant: {critic_err}"

    print(f"actor equivariance max abs err: {actor_err:.2e}")
    print(f"critic invariance  max abs err: {critic_err:.2e}")


if __name__ == "__main__":
    test_actor_equivariance_and_critic_invariance()
    print("OK")
