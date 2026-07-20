"""Critic-proxy dataset: random-policy returns on a fixed FLORIS layout.

The supervised value-regression proxy (research doc S5, policy/critic harness A)
fits each architecture's critic to empirical discounted returns collected under a
random policy on a single fixed 8-turbine layout with fixed wind. Wind and layout
are held constant across the whole dataset, so the only variation the critic must
explain is the yaw trajectory -- exactly the geometric-value signal the proxy is
meant to isolate. The dataset is generated once and cached under
``WIND_RL_WDIR`` keyed on its generating parameters.

Kept in the experiment dir (not the library) deliberately: promotion of shared
helpers happens later.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from wind_rl.env.factory import make_env
from wind_rl.scenario import ScenarioConfig
from wind_rl.static import GROUP_NAME

#: Per-agent observation sub-keys the built critic consumes (env contract).
OBS_FEATURES = ("wind_direction", "wind_speed", "yaw", "layout")
TARGET_KEY = "target"


def _discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Monte-Carlo within-episode discounted returns for ``(T, N, 1)`` rewards."""
    returns = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def _cache_path(
    wdir: Path,
    scenario: ScenarioConfig,
    layout: NDArray[np.float64],
    n_rollouts: int,
    gamma: float,
    seed: int,
) -> Path:
    key = repr(
        (
            scenario.model_dump(),
            np.round(layout, 3).tolist(),
            n_rollouts,
            gamma,
            seed,
        )
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return wdir / f"critic_dataset_{digest}.pt"


def _collect(
    scenario: ScenarioConfig,
    layout: NDArray[np.float64],
    n_rollouts: int,
    gamma: float,
    seed: int,
    device: str,
) -> TensorDict:
    env = make_env("eval", scenario, layout=layout, device=device)
    torch.manual_seed(seed)
    obs: dict[str, list[torch.Tensor]] = {name: [] for name in OBS_FEATURES}
    targets: list[torch.Tensor] = []
    try:
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            for _ in range(n_rollouts):
                rollout = env.rollout(scenario.max_steps, break_when_any_done=True)
                for name in OBS_FEATURES:
                    obs[name].append(rollout[GROUP_NAME, "observation", name])
                targets.append(
                    _discounted_returns(rollout["next", GROUP_NAME, "reward"], gamma)
                )
    finally:
        env.close()

    data = TensorDict(
        {
            GROUP_NAME: {
                "observation": {
                    name: torch.cat(obs[name], dim=0) for name in OBS_FEATURES
                }
            },
            TARGET_KEY: torch.cat(targets, dim=0),
        },
        batch_size=[torch.cat(targets, dim=0).shape[0]],
    )
    return data


def generate_or_load(
    wdir: Path,
    scenario: ScenarioConfig,
    layout: NDArray[np.float64],
    n_rollouts: int,
    gamma: float,
    seed: int,
    device: str,
) -> TensorDict:
    """Return the cached critic dataset, generating and caching it on a miss."""
    wdir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(wdir, scenario, layout, n_rollouts, gamma, seed)
    if path.exists():
        loaded = torch.load(path, weights_only=False)
        assert isinstance(loaded, TensorDict)
        return loaded.to(device)
    data = _collect(scenario, layout, n_rollouts, gamma, seed, device)
    torch.save(data.cpu(), path)
    return data


def split_standardized(
    data: TensorDict, val_fraction: float, seed: int
) -> tuple[TensorDict, TensorDict]:
    """Train/val split with the target standardized by TRAIN statistics.

    Standardizing to zero-mean/unit-variance makes the predict-the-mean baseline
    exactly ``MSE = 1`` / ``EV = 0``, so a critic clears the functional gate iff
    validation ``EV > 0``.
    """
    size = data.batch_size[0]
    perm = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(size * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train, val = data[train_idx].clone(), data[val_idx].clone()

    mean = train[TARGET_KEY].mean()
    std = train[TARGET_KEY].std().clamp_min(1e-8)
    train.set(TARGET_KEY, (train[TARGET_KEY] - mean) / std)
    val.set(TARGET_KEY, (val[TARGET_KEY] - mean) / std)
    return train, val
