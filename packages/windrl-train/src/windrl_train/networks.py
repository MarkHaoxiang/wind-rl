"""Permutation-equivariant GCN torsos for Mava's ff_mappo actor/critic.

A plain graph-convolution torso over the turbine (agent) axis. The adjacency is
a dense, row-normalized Gaussian kernel on pairwise turbine distances rather
than a kNN graph: a smooth affinity has no discrete top-k cutoff (top-k ties
would silently break permutation-equivariance) and is trivially differentiable,
which matters more than sparsity at N<=92 turbines. Both torsos plug into
Mava's ``FeedForwardActor``/``FeedForwardValueNet`` unchanged: the actor torso
consumes ``agents_view`` ``(..., N, F)`` and stays equivariant along ``N``; the
critic torso consumes the tiled ``global_state`` and pools to a value invariant
to turbine permutation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import orthogonal
from mava.networks.torsos import _parse_activation_fn


def _row_normalized_gaussian_adjacency(positions: jax.Array, sigma: float) -> jax.Array:
    """Dense row-stochastic Gaussian affinity from pairwise ``positions``.

    ``positions``: ``(..., N, 2)`` -> adjacency ``(..., N, N)``. Permuting the
    turbines permutes the adjacency identically, keeping message passing
    equivariant.
    """
    diff = positions[..., :, None, :] - positions[..., None, :, :]
    sq_dist = jnp.sum(diff * diff, axis=-1)
    affinity = jnp.exp(-sq_dist / (2.0 * sigma * sigma))
    return affinity / jnp.sum(affinity, axis=-1, keepdims=True)


class GCNTorso(nn.Module):
    """Parameter-shared, permutation-equivariant GCN over the agent axis.

    The last ``pos_dim`` channels of each agent's feature vector are its
    normalized ``(x, y)`` position, used to build the adjacency; all channels
    feed the node embedding. ``num_rounds`` residual message-passing steps mix
    neighbours; a turbine permutation permutes the output rows identically.
    """

    embed_dim: int = 128
    num_rounds: int = 2
    sigma: float = 0.5
    activation: str = "relu"
    pos_dim: int = 2

    @nn.compact
    def __call__(self, agents_view: jax.Array) -> jax.Array:
        activation_fn = _parse_activation_fn(self.activation)
        adjacency = _row_normalized_gaussian_adjacency(
            agents_view[..., -self.pos_dim :], self.sigma
        )
        node_embed = nn.Dense(self.embed_dim, kernel_init=orthogonal(np.sqrt(2)))(
            agents_view
        )
        for _ in range(self.num_rounds):
            message = nn.Dense(self.embed_dim, kernel_init=orthogonal(np.sqrt(2)))(
                adjacency @ node_embed
            )
            node_embed = activation_fn(node_embed + message)
        return node_embed


class GCNGlobalCritic(nn.Module):
    """GCN critic torso over the centralised global state.

    Mava's centralised critic hands the torso ``global_state`` ``(..., N, N*F)``,
    where every agent row is the same flattened ``agents_view``. We recover the
    ``(..., N, F)`` node matrix from one row, run the shared GCN, then mean-pool
    over turbines and broadcast back to ``N`` rows so the downstream value is
    invariant to turbine permutation.
    """

    embed_dim: int = 128
    num_rounds: int = 2
    sigma: float = 0.5
    num_features: int = 7
    activation: str = "relu"
    pos_dim: int = 2

    @nn.compact
    def __call__(self, global_state: jax.Array) -> jax.Array:
        activation_fn = _parse_activation_fn(self.activation)
        num_agents = global_state.shape[-2]
        nodes = global_state[..., 0, :].reshape(
            *global_state.shape[:-2], num_agents, self.num_features
        )
        adjacency = _row_normalized_gaussian_adjacency(
            nodes[..., -self.pos_dim :], self.sigma
        )
        node_embed = nn.Dense(self.embed_dim, kernel_init=orthogonal(np.sqrt(2)))(nodes)
        for _ in range(self.num_rounds):
            message = nn.Dense(self.embed_dim, kernel_init=orthogonal(np.sqrt(2)))(
                adjacency @ node_embed
            )
            node_embed = activation_fn(node_embed + message)
        pooled = jnp.mean(node_embed, axis=-2, keepdims=True)
        return jnp.broadcast_to(
            pooled, (*pooled.shape[:-2], num_agents, self.embed_dim)
        )
