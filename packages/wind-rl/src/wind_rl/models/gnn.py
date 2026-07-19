"""GCN actor/critic for the shared-parameter ``turbine`` agent group (research v0).

Dense-adjacency graph convolution, torch-native (no ``torch-scatter`` / PyG). The
graph is rebuilt every forward from the turbine layout carried in the per-agent
observation: a symmetric-normalised ``D^-1/2 (A + I) D^-1/2`` adjacency over KNN
(or fully-connected) edges, with distances from :func:`torch.cdist`. Node
features are the same engineered per-agent vector the MLP model uses (see
:mod:`wind_rl.models.mlp`); the graph is built from its trailing layout
coordinates.

Permutation equivariance is structural: ``Â X W`` with per-node-shared weights is
equivariant, so the policy's per-node yaws permute with the turbines and the
critic's mean-pooled value is permutation invariant.
"""

from __future__ import annotations

from typing import Literal, override

import torch
from pydantic import Field
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import Tensor, nn
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor, TanhNormal

from wind_rl.config import Config
from wind_rl.models.mlp import (
    _FEATURE_DIM,
    _LOC_KEY,
    _LOG_PROB_KEY,
    _OBS_VEC_KEY,
    _POS_SLICE,
    _SCALE_KEY,
    _VALUE_KEY,
    _feature_module,
)
from wind_rl.scenario import ScenarioConfig


class GcnModelConfig(Config):
    kind: Literal["gcn"] = "gcn"
    hidden_dim: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, ge=1)
    connectivity: Literal["knn", "full"] = "knn"
    #: KNN neighbours per node; ``None`` selects ``min(5, num_turbines - 1)``.
    k: int | None = Field(default=None, ge=1)
    initial_std: float = 1.0


def _knn_adjacency(positions: Tensor, k: int) -> Tensor:
    """Symmetric binary KNN adjacency (zero diagonal) from ``(*b, N, 2)`` coords."""
    n = positions.shape[-2]
    distances = torch.cdist(positions, positions)
    self_mask = torch.eye(n, dtype=torch.bool, device=positions.device)
    distances = distances.masked_fill(self_mask, float("inf"))
    # Rank each row by distance (double argsort) rather than scatter a topk index,
    # so the adjacency build carries a vmap batching rule under the PPO loss.
    rank = distances.argsort(dim=-1).argsort(dim=-1)
    directed = rank < min(k, n - 1)
    return (directed | directed.transpose(-1, -2)).to(positions.dtype)


def _full_adjacency(positions: Tensor) -> Tensor:
    n = positions.shape[-2]
    eye = torch.eye(n, dtype=positions.dtype, device=positions.device)
    ones = torch.ones(
        (*positions.shape[:-1], n), dtype=positions.dtype, device=positions.device
    )
    return ones - eye


def _normalize_adjacency(adjacency: Tensor) -> Tensor:
    n = adjacency.shape[-1]
    eye = torch.eye(n, dtype=adjacency.dtype, device=adjacency.device)
    with_self_loops = adjacency + eye
    d_inv_sqrt = with_self_loops.sum(-1).pow(-0.5)
    return d_inv_sqrt.unsqueeze(-1) * with_self_loops * d_inv_sqrt.unsqueeze(-2)


class _GcnEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        connectivity: Literal["knn", "full"],
        k: int,
    ) -> None:
        super().__init__()
        self.connectivity = connectivity
        self.k = k
        widths = [in_dim, *([hidden_dim] * num_layers)]
        self.layers = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]) for i in range(num_layers)
        )

    def _adjacency(self, positions: Tensor) -> Tensor:
        if self.connectivity == "full":
            return _full_adjacency(positions)
        # `positions` are per-axis [-1, 1] normalised (see mlp._ObservationFeatures),
        # so cdist distances are anisotropic on a non-square map -- KNN neighbours
        # can differ from the true metre-scale nearest neighbours. Accepted for v0.
        return _knn_adjacency(positions, self.k)

    @override
    def forward(self, features: Tensor) -> Tensor:
        adjacency = _normalize_adjacency(self._adjacency(features[..., _POS_SLICE]))
        hidden = features
        for layer in self.layers:
            transformed: Tensor = layer(hidden)
            hidden = torch.tanh(adjacency @ transformed)
        return hidden


class _GcnGaussianParams(nn.Module):
    def __init__(self, encoder: _GcnEncoder, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    @override
    def forward(self, features: Tensor) -> Tensor:
        loc: Tensor = self.head(self.encoder(features))
        return torch.cat([loc, torch.ones_like(loc) * self.log_std], dim=-1)


class _GcnCritic(nn.Module):
    def __init__(self, encoder: _GcnEncoder, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, 1)

    @override
    def forward(self, features: Tensor) -> Tensor:
        node_embeddings: Tensor = self.encoder(features)
        pooled = node_embeddings.mean(dim=-2)
        value: Tensor = self.head(pooled)
        n = features.shape[-2]
        return value.unsqueeze(-2).expand(*value.shape[:-1], n, 1)


def build_gcn_actor_critic(
    env: EnvBase,
    scenario: ScenarioConfig,
    cfg: GcnModelConfig,
    device: str | torch.device,
) -> tuple[ProbabilisticActor, TensorDictSequential]:
    """Build the (policy, critic) pair for ``env``, initialised on a reset tensordict."""
    feature_module = _feature_module(scenario, env)
    action_dim = env.full_action_spec[env.action_key].shape[-1]
    k = cfg.k if cfg.k is not None else min(5, env.num_agents - 1)

    def make_encoder() -> _GcnEncoder:
        return _GcnEncoder(
            in_dim=_FEATURE_DIM,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            connectivity=cfg.connectivity,
            k=k,
        )

    policy_head = nn.Sequential(
        _GcnGaussianParams(make_encoder(), cfg.hidden_dim, action_dim),
        NormalParamExtractor(
            scale_mapping=f"biased_softplus_{cfg.initial_std}", scale_lb=0.01
        ),
    )
    policy_body = TensorDictSequential(
        feature_module,
        TensorDictModule(
            policy_head, in_keys=[_OBS_VEC_KEY], out_keys=[_LOC_KEY, _SCALE_KEY]
        ),
        selected_out_keys=[_LOC_KEY, _SCALE_KEY],
    )
    action_space = env.full_action_spec_unbatched[env.action_key].space
    policy = ProbabilisticActor(
        module=policy_body,
        spec=env.action_spec_unbatched,
        in_keys=[_LOC_KEY, _SCALE_KEY],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        default_interaction_type=InteractionType.RANDOM,
        distribution_kwargs={"low": action_space.low, "high": action_space.high},
        return_log_prob=True,
        log_prob_key=_LOG_PROB_KEY,
    ).to(device)

    critic = TensorDictSequential(
        feature_module,
        TensorDictModule(
            _GcnCritic(make_encoder(), cfg.hidden_dim),
            in_keys=[_OBS_VEC_KEY],
            out_keys=[_VALUE_KEY],
        ),
        selected_out_keys=[_VALUE_KEY],
    ).to(device)

    reset_td = env.reset().to(device)
    with torch.no_grad():
        policy(reset_td)
        critic(reset_td)

    return policy, critic
