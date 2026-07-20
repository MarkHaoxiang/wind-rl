"""Numerical contracts of the critic-proxy dataset reductions (no FLORIS needed).

The env-backed collection path is exercised by the 0003 benchmark run; here we
pin the two pure pieces a run trusts blindly: the discounted-return recursion and
the train-standardized split.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from wind_rl.experiment.arch_bench.dataset import (
    TARGET_KEY,
    _discounted_returns,
    split_standardized,
)


def test_discounted_returns_match_geometric_closed_form() -> None:
    gamma, horizon, reward = 0.9, 5, 2.0
    rewards = torch.full((horizon, 3, 1), reward)
    returns = _discounted_returns(rewards, gamma)

    # Constant reward => G_t = r * (1 - gamma^(T-t)) / (1 - gamma); G_last = r.
    expected = torch.tensor(
        [reward * (1 - gamma ** (horizon - t)) / (1 - gamma) for t in range(horizon)]
    )
    assert torch.allclose(returns[:, 0, 0], expected, atol=1e-5)
    assert torch.allclose(returns[-1], torch.full_like(returns[-1], reward))


def test_split_standardized_uses_train_statistics_on_disjoint_partition() -> None:
    size, seed, val_fraction = 40, 7, 0.25
    target = torch.arange(size, dtype=torch.float32).unsqueeze(-1)
    data = TensorDict({TARGET_KEY: target}, batch_size=[size])

    train, val = split_standardized(data, val_fraction, seed)

    perm = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    n_val = int(size * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    assert train.batch_size[0] == size - n_val
    assert val.batch_size[0] == n_val
    assert set(val_idx.tolist()).isdisjoint(train_idx.tolist())

    mean = target[train_idx].mean()
    std = target[train_idx].std().clamp_min(1e-8)
    assert torch.allclose(train[TARGET_KEY], (target[train_idx] - mean) / std)
    assert torch.allclose(val[TARGET_KEY], (target[val_idx] - mean) / std)
    # Train target is standardized; val, scored by train stats, generally is not.
    assert abs(float(train[TARGET_KEY].mean())) < 1e-5
    assert abs(float(train[TARGET_KEY].std()) - 1.0) < 1e-5
