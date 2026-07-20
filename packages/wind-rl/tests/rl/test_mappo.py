from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torchrl.objectives import ClipPPOLoss

from wind_rl.rl.mappo import PPOConfig, batch_normalise_reward, build_optimiser


def test_batch_normalise_reward_zero_mean_unit_population_std() -> None:
    reward = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = batch_normalise_reward(reward)
    assert float(out.mean()) == 0.0
    torch.testing.assert_close(
        out.std(unbiased=False), torch.tensor(1.0), atol=1e-5, rtol=0
    )


def test_batch_normalise_reward_matches_manual_standardisation() -> None:
    reward = torch.tensor([[0.0, 10.0], [20.0, 30.0]])
    expected = (reward - reward.mean()) / (reward.std(unbiased=False) + 1e-8)
    torch.testing.assert_close(batch_normalise_reward(reward), expected)


def _optimiser(anneal: str, n_iters: int, lr: float = 0.1) -> tuple[object, object]:
    module = cast(ClipPPOLoss, nn.Linear(2, 2))
    return build_optimiser(module, PPOConfig(lr=lr, lr_anneal=anneal), n_iters)  # type: ignore[arg-type]


def test_linear_anneal_reaches_zero_linearly() -> None:
    optimiser, scheduler = _optimiser("linear", n_iters=4)
    assert scheduler is not None
    lrs = [optimiser.param_groups[0]["lr"]]  # type: ignore[attr-defined]
    for _ in range(4):
        scheduler.step()  # type: ignore[attr-defined]
        lrs.append(optimiser.param_groups[0]["lr"])  # type: ignore[attr-defined]
    assert lrs[0] == 0.1
    torch.testing.assert_close(
        torch.tensor(lrs[2]), torch.tensor(0.05), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        torch.tensor(lrs[4]), torch.tensor(0.0), atol=1e-6, rtol=0
    )


def test_no_anneal_returns_no_scheduler() -> None:
    _, scheduler = _optimiser("none", n_iters=4)
    assert scheduler is None


def test_cosine_anneal_returns_scheduler() -> None:
    _, scheduler = _optimiser("cosine", n_iters=4)
    assert scheduler is not None
