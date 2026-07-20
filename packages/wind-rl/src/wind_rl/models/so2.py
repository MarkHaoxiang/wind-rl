"""SO(2) circular-harmonic attention actor/critic (research v2).

An exactly rotation-equivariant transformer over turbine tokens. Where v1
(:mod:`wind_rl.models.transformer`) buys rotation robustness by canonicalising
into the wind frame, v2 gets it *structurally*: features live in SO(2) irreps
(circular-harmonic / Fourier modes ``e^{im theta}``), every layer preserves the
irrep type, and the output reads only invariant channels -- so co-rotating all
positions **and** the wind vector by any angle leaves the yaw actions identical,
with no canonicalisation step.

Representation. In 2D the rotation group is abelian and its irreps are the
integer Fourier modes: a type-``m`` feature picks up ``e^{im theta}`` under a
global rotation by ``theta``. A node feature is one real angular function per
channel, stored (Hermitian-symmetric) as its non-negative modes: an
**invariant scalar part** ``s`` (mode ``m=0``, real) plus a **complex vector
part** ``v`` for modes ``1..max_m``. Complex tensors (not ``(re, im)`` pairs)
are used throughout: ``e^{im theta}`` phase arithmetic and the tensor-product
FFTs are native complex ops with clean vmap batching rules, and keeping ``m=0``
as a separate real tensor makes the invariant readout unambiguous and avoids
carrying dead imaginary parts.

Geometry. Positions enter only through *relative* edges (translation- as well
as rotation-friendly): edge ``(i, j)`` contributes an invariant radial embedding
of ``|p_j - p_i|`` and the type-``m`` phases ``e^{im theta_ij}`` built directly
from the unit direction (no ``atan2``, matching v1's trig-free style). The wind
vector is injected as a type-1 feature ``wind_x + i wind_y`` (co-rotates
correctly); wind speed and the wind-relative yaw seed the invariant scalars.

Message passing. Attention logits are formed from **invariant** quantities only
(scalar channels, ``|v|`` magnitudes, radial edge bias) so the weights are
rotation invariant; the values carry every irrep and are phase-composed against
the edge harmonics by an SO(2) tensor product -- ``m``-index addition
``c_m = sum_{m1} a_{m1} b_{m-m1}``, computed as a pointwise product on an angular
grid via :func:`torch.fft` (exact for band-limited factors, unlike a grid
*nonlinearity*). Channel mixing within each mode is an SO(2)-linear (per-mode
complex ``1x1`` map).

Nonlinearity. Magnitude-gated, **not** a grid pointwise nonlinearity: a pointwise
nonlinearity on a fixed angular grid injects high harmonics that alias under an
off-grid rotation, so it is only *approximately* equivariant -- incompatible with
the exact-invariance guarantee here. Instead type-``m`` channels are gated by an
invariant scalar (a function of ``s`` and ``|v|``), which is exactly equivariant.
The FFT grid is still used for the tensor product, where it is exact.
"""

from __future__ import annotations

import math
from typing import Literal, override

import torch
import torch.nn.functional as F
from pydantic import Field, model_validator
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import Tensor, nn
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor, TanhNormal

from wind_rl.config import Config
from wind_rl.models.mlp import (
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

_RADIAL_BASIS_DIM = 4


class So2ModelConfig(Config):
    kind: Literal["so2"] = "so2"
    #: Highest circular-harmonic frequency retained (irreps ``m = 0..max_m``).
    max_m: int = Field(default=3, ge=1)
    #: Channels per irrep type (scalar channels and per-mode vector channels).
    embed_dim: int = Field(default=32, gt=0)
    num_layers: int = Field(default=2, ge=1)
    num_heads: int = Field(default=4, gt=0)
    initial_std: float = 1.0

    @model_validator(mode="after")
    def _heads_divide_embed_dim(self) -> So2ModelConfig:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        return self


def _grid_size(max_m: int) -> int:
    # Product of two band-limited-``max_m`` functions is band-limited to 2*max_m;
    # sampling at >= 4*max_m + 1 points makes the FFT tensor product alias-free
    # and therefore exact (verified against direct index-addition in the tests).
    return 4 * max_m + 1


def _to_grid(s: Tensor, v: Tensor, grid_size: int) -> Tensor:
    """Angular samples of the real function with modes ``(s, v_1..v_max_m)``.

    ``s`` is ``(*b, C)`` real, ``v`` is ``(*b, max_m, C)`` complex; returns
    ``(*b, grid_size, C)`` real. Out-of-place (``cat``/``pad``, no index
    assignment) so it carries a vmap batching rule (the critic is vmapped in GAE).
    """
    max_m = v.shape[-2]
    n_freq = grid_size // 2 + 1
    half_spectrum = torch.cat([s.to(v.dtype).unsqueeze(-2), v], dim=-2)
    half_spectrum = F.pad(half_spectrum, (0, 0, 0, n_freq - (max_m + 1)))
    grid: Tensor = torch.fft.irfft(half_spectrum, n=grid_size, dim=-2)
    return grid * grid_size


def _from_grid(grid: Tensor, max_m: int) -> tuple[Tensor, Tensor]:
    grid_size = grid.shape[-2]
    coeffs: Tensor = torch.fft.rfft(grid, dim=-2) / grid_size
    return coeffs[..., 0, :].real, coeffs[..., 1 : max_m + 1, :]


def _tensor_product(
    s_a: Tensor, v_a: Tensor, s_b: Tensor, v_b: Tensor, grid_size: int
) -> tuple[Tensor, Tensor]:
    """SO(2) tensor product (``m``-index addition) of two irrep features."""
    grid = _to_grid(s_a, v_a, grid_size) * _to_grid(s_b, v_b, grid_size)
    return _from_grid(grid, v_a.shape[-2])


def _radial_basis(r: Tensor) -> Tensor:
    return torch.cat([r, r * r, torch.exp(-r), torch.exp(-r * r)], dim=-1)


def _edge_geometry(positions: Tensor, max_m: int) -> tuple[Tensor, Tensor]:
    """Edge harmonics and radial basis from ``(*b, N, 2)`` positions.

    Returns ``phases`` ``(*b, N, N, max_m+1)`` complex with
    ``phases[..., i, j, m] = e^{im theta_ij}`` (``theta_ij`` the angle of
    ``p_j - p_i``, computed trig-free from the unit direction) and ``basis``
    ``(*b, N, N, _RADIAL_BASIS_DIM)`` of the invariant edge length.
    """
    rel = positions.unsqueeze(-3) - positions.unsqueeze(-2)
    dx, dy = rel[..., 0], rel[..., 1]
    r = torch.sqrt((dx * dx + dy * dy).clamp_min(1e-24))
    inv_r = r.clamp_min(1e-12).reciprocal()
    unit = torch.complex(dx * inv_r, dy * inv_r)
    powers = [torch.ones_like(unit)]
    for _ in range(max_m):
        powers.append(powers[-1] * unit)
    phases = torch.stack(powers, dim=-1)
    return phases, _radial_basis(r.unsqueeze(-1))


class _So2Linear(nn.Module):
    """Type-preserving linear map: real ``1x1`` on ``s``, per-mode complex on ``v``."""

    def __init__(self, max_m: int, c_in: int, c_out: int) -> None:
        super().__init__()
        self.scalar = nn.Linear(c_in, c_out)
        bound = 1.0 / math.sqrt(c_in)
        self.weight_real = nn.Parameter(
            torch.empty(max_m, c_in, c_out).uniform_(-bound, bound)
        )
        self.weight_imag = nn.Parameter(
            torch.empty(max_m, c_in, c_out).uniform_(-bound, bound)
        )

    @override
    def forward(self, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        weight = torch.complex(self.weight_real, self.weight_imag)
        return self.scalar(s), torch.matmul(v.unsqueeze(-2), weight).squeeze(-2)


def _invariants(s: Tensor, v: Tensor) -> Tensor:
    return torch.cat([s, v.abs().flatten(-2)], dim=-1)


def _split_last(x: Tensor, first: int, second: int) -> Tensor:
    return x.reshape(*x.shape[:-1], first, second)


class _So2Norm(nn.Module):
    def __init__(self, max_m: int, embed_dim: int) -> None:
        super().__init__()
        self.scalar_norm = nn.LayerNorm(embed_dim)
        self.vector_gain = nn.Parameter(torch.ones(max_m, 1))

    @override
    def forward(self, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        rms = v.abs().square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        return self.scalar_norm(s), v / rms * self.vector_gain


class _So2FeedForward(nn.Module):
    def __init__(self, max_m: int, embed_dim: int) -> None:
        super().__init__()
        hidden = 2 * embed_dim
        self.lin_in = _So2Linear(max_m, embed_dim, hidden)
        self.lin_out = _So2Linear(max_m, hidden, embed_dim)
        self.gate = nn.Linear(hidden + max_m * hidden, max_m * hidden)
        self.max_m = max_m
        self.hidden = hidden

    @override
    def forward(self, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        s, v = self.lin_in(s, v)
        gate = torch.sigmoid(self.gate(_invariants(s, v)))
        gate = _split_last(gate, self.max_m, self.hidden)
        out: tuple[Tensor, Tensor] = self.lin_out(F.silu(s), v * gate)
        return out


class _So2Attention(nn.Module):
    def __init__(
        self, max_m: int, embed_dim: int, num_heads: int, grid_size: int
    ) -> None:
        super().__init__()
        self.max_m = max_m
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.grid_size = grid_size
        inv_dim = embed_dim + max_m * embed_dim
        self.query = nn.Linear(inv_dim, embed_dim)
        self.key = nn.Linear(inv_dim, embed_dim)
        self.edge_bias = nn.Linear(_RADIAL_BASIS_DIM, num_heads)
        self.value = _So2Linear(max_m, embed_dim, embed_dim)
        self.edge_scalar = nn.Linear(_RADIAL_BASIS_DIM, embed_dim)
        self.edge_vector = nn.Linear(_RADIAL_BASIS_DIM, max_m * embed_dim)
        self.out = _So2Linear(max_m, embed_dim, embed_dim)

    @override
    def forward(
        self, s: Tensor, v: Tensor, phases: Tensor, basis: Tensor
    ) -> tuple[Tensor, Tensor]:
        n = s.shape[-2]
        invariant = _invariants(s, v)
        q = _split_last(self.query(invariant), self.num_heads, self.head_dim)
        k = _split_last(self.key(invariant), self.num_heads, self.head_dim)
        # (*b, heads, N_i, N_j)
        scores = torch.matmul(
            q.transpose(-3, -2), k.transpose(-3, -2).transpose(-1, -2)
        ) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(basis).movedim(-1, -3)
        diagonal = torch.eye(n, dtype=torch.bool, device=s.device)
        # Drop self-edges: their direction is undefined (r=0), so alpha_ii=0 keeps
        # the (finite but ill-conditioned) diagonal phase gradients from flowing;
        # a node keeps its own state through the residual connection instead.
        scores = scores.masked_fill(diagonal, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)

        edge_s = self.edge_scalar(basis)
        edge_v = _split_last(self.edge_vector(basis), self.max_m, self.embed_dim)
        edge_v = edge_v * phases[..., 1:].unsqueeze(-1)

        value_s, value_v = self.value(s, v)
        msg_s, msg_v = _tensor_product(
            value_s.unsqueeze(-3), value_v.unsqueeze(-4), edge_s, edge_v, self.grid_size
        )

        head_dim = self.embed_dim // self.num_heads
        weight = alpha.movedim(-3, -1)
        out_s = torch.einsum(
            "...ijh,...ijhc->...ihc",
            weight,
            _split_last(msg_s, self.num_heads, head_dim),
        )
        out_v = torch.einsum(
            "...ijh,...ijmhc->...imhc",
            weight.to(msg_v.dtype),
            _split_last(msg_v, self.num_heads, head_dim),
        )
        out: tuple[Tensor, Tensor] = self.out(out_s.flatten(-2), out_v.flatten(-2))
        return out


class _So2Block(nn.Module):
    def __init__(
        self, max_m: int, embed_dim: int, num_heads: int, grid_size: int
    ) -> None:
        super().__init__()
        self.norm_attn = _So2Norm(max_m, embed_dim)
        self.attn = _So2Attention(max_m, embed_dim, num_heads, grid_size)
        self.norm_ff = _So2Norm(max_m, embed_dim)
        self.ff = _So2FeedForward(max_m, embed_dim)

    @override
    def forward(
        self, s: Tensor, v: Tensor, phases: Tensor, basis: Tensor
    ) -> tuple[Tensor, Tensor]:
        norm_s, norm_v = self.norm_attn(s, v)
        attn_s, attn_v = self.attn(norm_s, norm_v, phases, basis)
        s, v = s + attn_s, v + attn_v
        norm_s, norm_v = self.norm_ff(s, v)
        ff_s, ff_v = self.ff(norm_s, norm_v)
        return s + ff_s, v + ff_v


class _So2Encoder(nn.Module):
    def __init__(
        self, max_m: int, embed_dim: int, num_layers: int, num_heads: int
    ) -> None:
        super().__init__()
        self.max_m = max_m
        self.embed_dim = embed_dim
        grid_size = _grid_size(max_m)
        self.scalar_embed = nn.Sequential(
            nn.Linear(2, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        # No bias: a type-1 feature admits no invariant additive term.
        self.wind_embed = nn.Linear(1, embed_dim, bias=False, dtype=torch.cfloat)
        self.radial = nn.Sequential(
            nn.Linear(_RADIAL_BASIS_DIM, _RADIAL_BASIS_DIM),
            nn.SiLU(),
            nn.Linear(_RADIAL_BASIS_DIM, _RADIAL_BASIS_DIM),
        )
        self.blocks = nn.ModuleList(
            _So2Block(max_m, embed_dim, num_heads, grid_size) for _ in range(num_layers)
        )
        self.norm_out = _So2Norm(max_m, embed_dim)

    @override
    def forward(self, features: Tensor) -> Tensor:
        wind = features[..., _WIND_SLICE]
        yaw = features[..., _WIND_SLICE.stop : _POS_SLICE.start]
        positions = features[..., _POS_SLICE]

        speed = wind.square().sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
        s = self.scalar_embed(torch.cat([yaw, speed], dim=-1))
        wind_c = torch.complex(wind[..., 0:1], wind[..., 1:2])
        v = self.wind_embed(wind_c).unsqueeze(-2)
        v = F.pad(v, (0, 0, 0, self.max_m - 1))

        phases, basis = _edge_geometry(positions, self.max_m)
        basis = self.radial(basis)
        for block in self.blocks:
            s, v = block(s, v, phases, basis)
        s, v = self.norm_out(s, v)
        return _invariants(s, v)


def _readout_dim(cfg: So2ModelConfig) -> int:
    return cfg.embed_dim + cfg.max_m * cfg.embed_dim


class _So2GaussianParams(nn.Module):
    def __init__(self, encoder: _So2Encoder, readout_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(readout_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    @override
    def forward(self, features: Tensor) -> Tensor:
        loc: Tensor = self.head(self.encoder(features))
        return torch.cat([loc, torch.ones_like(loc) * self.log_std], dim=-1)


class _So2Critic(nn.Module):
    def __init__(self, encoder: _So2Encoder, readout_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(readout_dim, 1)

    @override
    def forward(self, features: Tensor) -> Tensor:
        invariants: Tensor = self.encoder(features)
        pooled = invariants.mean(dim=-2)
        value: Tensor = self.head(pooled)
        n = features.shape[-2]
        return value.unsqueeze(-2).expand(*value.shape[:-1], n, 1)


def build_so2_actor_critic(
    env: EnvBase,
    scenario: ScenarioConfig,
    cfg: So2ModelConfig,
    device: str | torch.device,
) -> tuple[ProbabilisticActor, TensorDictSequential]:
    """Build the (policy, critic) pair for ``env``, initialised on a reset tensordict."""
    feature_module = _feature_module(scenario, env)
    action_dim = env.full_action_spec[env.action_key].shape[-1]
    readout_dim = _readout_dim(cfg)

    def make_encoder() -> _So2Encoder:
        return _So2Encoder(
            max_m=cfg.max_m,
            embed_dim=cfg.embed_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
        )

    policy_head = nn.Sequential(
        _So2GaussianParams(make_encoder(), readout_dim, action_dim),
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
            _So2Critic(make_encoder(), readout_dim),
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
