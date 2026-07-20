"""Supervised value-regression proxy: fit each critic to empirical returns.

Every architecture's critic (built through the shared :func:`build_actor_critic`
union entry point) is trained under an IDENTICAL optimiser/step budget on the
cached random-policy dataset, then scored by validation MSE and explained
variance against a predict-the-mean baseline. ``EV > 0`` is the functional gate.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from dataset import TARGET_KEY, generate_or_load, split_standardized
from numpy.typing import NDArray
from tensordict import TensorDict
from torch.nn.functional import mse_loss

from wind_rl.env.factory import make_env
from wind_rl.models import ModelConfig, build_actor_critic
from wind_rl.rl.logging import explained_variance
from wind_rl.scenario import ScenarioConfig
from wind_rl.static import GROUP_NAME

_VALUE_KEY = (GROUP_NAME, "state_value")


class CriticResult(NamedTuple):
    name: str
    val_mse: float
    explained_variance: float
    params: int


def _train_critic(
    critic: torch.nn.Module,
    train: TensorDict,
    val: TensorDict,
    n_steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    optimiser = torch.optim.Adam(critic.parameters(), lr=lr)
    size = train.batch_size[0]

    critic.train()
    for _ in range(n_steps):
        idx = torch.randint(0, size, (batch_size,), generator=generator)
        batch = train[idx]
        prediction = critic(batch)[_VALUE_KEY]
        loss = mse_loss(prediction, batch[TARGET_KEY])
        optimiser.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimiser.step()

    critic.eval()
    with torch.no_grad():
        prediction = critic(val)[_VALUE_KEY]
        target = val[TARGET_KEY]
        return float(mse_loss(prediction, target)), explained_variance(
            target, prediction
        )


class CriticProxyConfig(NamedTuple):
    n_rollouts: int
    gamma: float
    val_fraction: float
    n_steps: int
    batch_size: int
    lr: float
    ev_gate: float


def run_critic_proxy(
    wdir_root: object,
    scenario: ScenarioConfig,
    layout: NDArray[np.float64],
    variants: list[tuple[str, ModelConfig]],
    cfg: CriticProxyConfig,
    seed: int,
    device: str,
) -> list[CriticResult]:
    from pathlib import Path

    wdir = Path(str(wdir_root))
    data = generate_or_load(
        wdir, scenario, layout, cfg.n_rollouts, cfg.gamma, seed, device
    )
    train, val = split_standardized(data, cfg.val_fraction, seed)

    results: list[CriticResult] = []
    for name, model in variants:
        env = make_env("eval", scenario, layout=layout, device=device)
        try:
            policy, critic = build_actor_critic(env, scenario, model, device)
        finally:
            env.close()
        params = sum(p.numel() for p in policy.parameters()) + sum(
            p.numel() for p in critic.parameters()
        )
        val_mse, ev = _train_critic(
            critic, train, val, cfg.n_steps, cfg.batch_size, cfg.lr, seed
        )
        print(
            f"[critic:{name}] val_mse {val_mse:.4f}  EV {ev:+.4f}  "
            f"params {params}  -> {'FUNCTIONAL' if ev > cfg.ev_gate else 'FAIL'}"
        )
        results.append(CriticResult(name, val_mse, ev, params))
    return results
