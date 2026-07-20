"""Permutation-equivariant set-transformer actor/critic (research v1).

A standard pre-LN multi-head self-attention encoder over turbine tokens, one
token per turbine built from the shared engineered feature vector (see
:mod:`wind_rl.models.mlp`). No positional encodings: permutation equivariance is
then structural (attention is a symmetric function of the token set, so the
policy's per-token yaws permute with the turbines and the critic's mean-pooled
value is permutation invariant), and geometry enters purely through the
per-token position features.

Rotation robustness comes from **wind-frame canonicalisation** rather than
equivariant machinery: the reward is wind-relative, so before encoding we rotate
every token's geometry into the frame where the wind blows along ``+x`` (see
:func:`_canonicalise_wind_frame`). Yaw actions in FLORIS/wfcrl are misalignments
*relative to the inflow*, hence already wind-relative and rotation-invariant --
so the network emits them directly with no rotate-back step.
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
    _WIND_SLICE,
    _feature_module,
)
from wind_rl.scenario import ScenarioConfig


class SetTransformerModelConfig(Config):
    kind: Literal["set_transformer"] = "set_transformer"
    embed_dim: int = Field(default=64, gt=0)
    num_heads: int = Field(default=4, gt=0)
    num_layers: int = Field(default=2, ge=1)
    mlp_ratio: float = Field(default=2.0, gt=0)
    canonicalize_wind: bool = True
    initial_std: float = 1.0


def _canonicalise_wind_frame(features: Tensor) -> Tensor:
    """Rotate each token's geometry into the frame where wind blows along ``+x``.

    Frame convention: with wind angle ``theta = atan2(wind_y, wind_x)`` read from
    the token's cartesian wind feature, positions and the wind vector are rotated
    by ``-theta`` (``R(-theta) = [[cos, sin], [-sin, cos]]``). The wind feature
    thereby collapses to ``[speed, 0]`` and positions become wind-relative.
    Features between the wind and position slices (yaw misalignment) are already
    wind-relative and pass through unchanged. Purely elementwise + ``cat`` so it
    carries a vmap batching rule (the critic is vmapped in GAE).
    """
    wind = features[..., _WIND_SLICE]
    invariant = features[..., _WIND_SLICE.stop : _POS_SLICE.start]
    positions = features[..., _POS_SLICE]

    wind_x = wind[..., 0:1]
    wind_y = wind[..., 1:2]
    speed = torch.sqrt(wind_x * wind_x + wind_y * wind_y)
    # cos/sin of the wind angle without atan2->trig; clamp guards the zero-wind
    # degeneracy (direction undefined, so any frame is fine).
    inv_speed = speed.clamp_min(1e-8).reciprocal()
    cos = wind_x * inv_speed
    sin = wind_y * inv_speed

    pos_x = positions[..., 0:1]
    pos_y = positions[..., 1:2]
    rot_x = cos * pos_x + sin * pos_y
    rot_y = -sin * pos_x + cos * pos_y

    return torch.cat([speed, torch.zeros_like(speed), invariant, rot_x, rot_y], dim=-1)


class _TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        hidden = max(1, int(embed_dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    @override
    def forward(self, tokens: Tensor) -> Tensor:
        normed = self.norm_attn(tokens)
        # need_weights=False keeps nn.MultiheadAttention off the fused fast path
        # that lacks a vmap batching rule; the math fallback is vmap-safe.
        attended: Tensor = self.attn(normed, normed, normed, need_weights=False)[0]
        tokens = tokens + attended
        mlp_out: Tensor = self.mlp(self.norm_mlp(tokens))
        return tokens + mlp_out


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        canonicalize_wind: bool,
    ) -> None:
        super().__init__()
        self.canonicalize_wind = canonicalize_wind
        self.embed = nn.Linear(_FEATURE_DIM, embed_dim)
        self.blocks = nn.ModuleList(
            _TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    @override
    def forward(self, features: Tensor) -> Tensor:
        if self.canonicalize_wind:
            features = _canonicalise_wind_frame(features)
        # nn.MultiheadAttention wants (batch, tokens, embed); collapse any extra
        # leading dims (time/agent-batch, or none under vmap) into one batch axis.
        lead_shape = features.shape[:-2]
        n_tokens = features.shape[-2]
        embedded: Tensor = self.embed(features)
        tokens = embedded.reshape(-1, n_tokens, self.embed.out_features)
        for block in self.blocks:
            tokens = block(tokens)
        pooled: Tensor = self.norm_out(tokens)
        return pooled.reshape(*lead_shape, n_tokens, self.embed.out_features)


class _SetTransformerGaussianParams(nn.Module):
    def __init__(
        self, encoder: _TransformerEncoder, embed_dim: int, action_dim: int
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    @override
    def forward(self, features: Tensor) -> Tensor:
        loc: Tensor = self.head(self.encoder(features))
        return torch.cat([loc, torch.ones_like(loc) * self.log_std], dim=-1)


class _SetTransformerCritic(nn.Module):
    def __init__(self, encoder: _TransformerEncoder, embed_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, 1)

    @override
    def forward(self, features: Tensor) -> Tensor:
        token_embeddings: Tensor = self.encoder(features)
        pooled = token_embeddings.mean(dim=-2)
        value: Tensor = self.head(pooled)
        n = features.shape[-2]
        return value.unsqueeze(-2).expand(*value.shape[:-1], n, 1)


def build_set_transformer_actor_critic(
    env: EnvBase,
    scenario: ScenarioConfig,
    cfg: SetTransformerModelConfig,
    device: str | torch.device,
) -> tuple[ProbabilisticActor, TensorDictSequential]:
    """Build the (policy, critic) pair for ``env``, initialised on a reset tensordict."""
    feature_module = _feature_module(scenario, env)
    action_dim = env.full_action_spec[env.action_key].shape[-1]

    def make_encoder() -> _TransformerEncoder:
        return _TransformerEncoder(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            mlp_ratio=cfg.mlp_ratio,
            canonicalize_wind=cfg.canonicalize_wind,
        )

    policy_head = nn.Sequential(
        _SetTransformerGaussianParams(make_encoder(), cfg.embed_dim, action_dim),
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
            _SetTransformerCritic(make_encoder(), cfg.embed_dim),
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
