"""Flow-map layout prior: a velocity field over flattened ``(2N)`` layout vectors
conforming to mfm's ``BaseModel.v(s, u, x, t_cond, x_cond, ...)`` interface, its
consistency/flow-matching trainer, and a few-step sampler.

v0 is the *pure flow-matching* regime: only the diagonal ``s == u`` term is
trained (mfm's off-diagonal consistency term is gated behind
``step > num_warmup_steps`` and parked beyond any run length), so the model
learns the instantaneous FM velocity and the sampler is a few-step Euler
integration of the probability-flow ODE. Coordinates are normalised to
``[-1, 1]`` per axis for training and sampling; the sampler returns map metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import override

import numpy as np
import torch
from mfm.losses.losses import get_consistency_loss_fn
from mfm.models.base_model import BaseModel
from mfm.SI import Linear
from numpy.typing import NDArray
from torch import Tensor, nn

from wind_rl.design.geometry import sample_feasible_layout
from wind_rl.scenario import ScenarioConfig

# Off-diagonal consistency warmup parked beyond any run length => pure flow
# matching (diagonal FM only), mirroring physics-informed-flow-map's "pure FM".
_DISABLED = 10**12


@dataclass(frozen=True)
class FlowMapArch:
    n_turbines: int
    map_x_length: float
    map_y_length: float
    width: int = 256
    depth: int = 4
    time_dim: int = 128


class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @override
    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half, 1)
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class FlowMapModel(BaseModel):  # type: ignore[misc]  # mfm BaseModel is untyped
    """Time-conditioned MLP velocity field over flattened, axis-normalised layouts.

    ``v`` takes flattened ``(B, 2N)`` layout vectors in ``[-1, 1]`` per axis and
    ignores ``t_cond``/``x_cond`` (unconditional). ``encode``/``decode`` convert
    between ``(B, N, 2)`` map-metre layouts and the flattened normalised space.
    """

    def __init__(self, arch: FlowMapArch) -> None:
        super().__init__()
        self.arch = arch
        self.dim = 2 * arch.n_turbines
        self.time_embed = _TimeEmbedding(arch.time_dim)
        # Buffers so scale persists in the state dict; sampling reads them back.
        self.register_buffer(
            "_scale",
            torch.tensor([arch.map_x_length, arch.map_y_length], dtype=torch.float32),
        )
        layers: list[nn.Module] = []
        in_dim = self.dim + arch.time_dim
        for _ in range(arch.depth):
            layers += [nn.Linear(in_dim, arch.width), nn.SiLU()]
            in_dim = arch.width
        layers += [nn.Linear(in_dim, self.dim)]
        self.net = nn.Sequential(*layers)

    def v(
        self,
        s: Tensor,
        t: Tensor,
        x: Tensor,
        t_cond: Tensor,
        x_cond: Tensor,
        **kwargs: object,
    ) -> Tensor:
        temb = self.time_embed(s)
        out: Tensor = self.net(torch.cat([x, temb], dim=-1))
        return out

    def encode(self, layout: Tensor) -> Tensor:
        scale: Tensor = self._scale.to(layout.device)
        normed = layout / scale * 2.0 - 1.0
        return normed.reshape(layout.shape[0], self.dim)

    def decode(self, flat: Tensor) -> Tensor:
        scale: Tensor = self._scale.to(flat.device)
        layout = flat.reshape(flat.shape[0], self.arch.n_turbines, 2)
        return (layout + 1.0) * 0.5 * scale


def _loss_cfg() -> SimpleNamespace:
    # Attribute bag mirroring the Hydra DictConfig mfm's consistency loss reads.
    # Pure-FM subset: unconditional (t_cond==0 always), l2 weighting, off-diagonal
    # term disabled. `loss` needs a `.get` (mfm reads adaptive_* via `.get`).
    loss = SimpleNamespace(
        data_fm=True,
        distill_fm=False,
        distillation_type="mf",
        model_guidance=False,
        model_guidance_base_prob=0.5,
        fm_loss_type="l2",
        distillation_loss_type="l2",
        distill_fm_loss_type="l2",
        distill_teacher_stop_grad=True,
    )
    loss.get = lambda key, default=None: getattr(loss, key, default)
    return SimpleNamespace(
        SI=SimpleNamespace(t_max=1.0),
        trainer=SimpleNamespace(
            t_cond_warmup_steps=0,
            t_cond_0_rate=1.0,  # fully unconditional
            t_cond_power=1.0,
            num_warmup_steps=_DISABLED,
            anneal_end_step=_DISABLED,
            class_dropout_prob=0.0,
        ),
        model=SimpleNamespace(
            label_dim=0,
            learn_loss_weighting=False,
            model_guidance_class_ws=[],
            model_guidance_x_cond_ws=[],
            init="dmf",
        ),
        loss=loss,
    )


def _feasible_dataset(
    scenario: ScenarioConfig, n_samples: int, rng: np.random.Generator
) -> Tensor:
    layouts = np.stack(
        [sample_feasible_layout(scenario, rng) for _ in range(n_samples)]
    )
    return torch.as_tensor(layouts, dtype=torch.float32)


def train_flowmap_prior(
    scenario: ScenarioConfig,
    *,
    n_samples: int = 2048,
    n_iters: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    arch: FlowMapArch | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[FlowMapModel, list[float]]:
    """Train a pure-FM layout prior on procedurally sampled feasible layouts.

    Returns the trained model and the per-iteration total loss history. The loss
    is mfm's SI consistency loss with the off-diagonal term disabled, i.e. plain
    flow-matching velocity regression on axis-normalised ``[-1, 1]`` coordinates.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if arch is None:
        arch = FlowMapArch(
            n_turbines=scenario.n_turbines,
            map_x_length=scenario.map_x_length,
            map_y_length=scenario.map_y_length,
        )
    dev = torch.device(device)
    model = FlowMapModel(arch)
    model.to(dev)
    data = _feasible_dataset(scenario, n_samples, rng).to(dev)
    x1_all = model.encode(data)

    loss_fn = get_consistency_loss_fn(_loss_cfg(), Linear(t_max=1.0))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []
    model.train()
    for step in range(n_iters):
        idx = torch.randint(0, x1_all.shape[0], (batch_size,), device=dev)
        x1 = x1_all[idx]
        opt_losses, _ = loss_fn(model, None, x1, None, step=step, teacher_model=None)
        loss = sum(opt_losses.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return model, history


@torch.no_grad()
def sample_layouts(
    model: FlowMapModel, n: int, steps: int, *, seed: int | None = None
) -> NDArray[np.float64]:
    """Few-step Euler sampler; returns ``(n, N, 2)`` layouts in map metres.

    ``steps`` in ``[1, 8]``: probability-flow ODE integrated from Gaussian noise
    with the trained instantaneous velocity ``v(t, t, x)``.
    """
    model.eval()
    device = model._scale.device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, model.dim, device=device, generator=generator)
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    zeros = torch.zeros(n, device=device)
    for i in range(steps):
        s = ts[i].expand(n)
        v = model.v(s, s, x, zeros, torch.zeros_like(x))
        x = x + (ts[i + 1] - ts[i]) * v
    layout = model.decode(x)
    return layout.cpu().numpy().astype(np.float64)


def save_flowmap(model: FlowMapModel, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "arch": model.arch}, path)


def load_flowmap(path: str, device: str = "cpu") -> FlowMapModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    if not isinstance(arch, FlowMapArch):
        raise TypeError(f"checkpoint {path!r} has no FlowMapArch")
    model = FlowMapModel(arch)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model
