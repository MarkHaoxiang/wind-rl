from __future__ import annotations

from typing import cast, override

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.data import ReplayBuffer
from torchrl.objectives import ClipPPOLoss

from wind_rl.rl.mappo import (
    PPOConfig,
    batch_normalise_reward,
    build_optimiser,
    run_ppo_epochs,
)


class _FakeLoss(nn.Module):
    """A minimal stand-in for ``ClipPPOLoss`` that emits scripted ``kl_approx``.

    Each forward returns the next scripted approx-KL (last value repeats), so
    :func:`run_ppo_epochs`'s target-KL early-stop can be exercised on synthetic
    losses with no env. The scalar objective/critic depend on ``theta`` so
    ``backward`` populates a real gradient for ``clip_grad_norm_``.
    """

    def __init__(self, kls: list[float]) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(1))
        self._kls = kls
        self.calls = 0

    @override
    def forward(self, _minibatch: object) -> TensorDict:
        kl = self._kls[min(self.calls, len(self._kls) - 1)]
        self.calls += 1
        return TensorDict(
            {
                "loss_objective": self.theta.sum(),
                "loss_critic": (self.theta**2).sum(),
                "kl_approx": torch.tensor(kl),
            },
            [],
        )


class _FakeBuffer:
    def sample(self) -> None:
        return None


def _run(loss: _FakeLoss, cfg: PPOConfig) -> dict[str, list[float]]:
    optimiser = torch.optim.SGD(loss.parameters(), lr=0.0)
    _, diagnostics = run_ppo_epochs(
        loss, cast(ReplayBuffer, _FakeBuffer()), optimiser, cfg
    )
    return diagnostics


def test_target_kl_halts_after_epoch_boundary_not_mid_epoch() -> None:
    # Epoch 1's mean approx_kl ((0.001+0.001+0.5)/3 ~= 0.167) trips the 0.015
    # guard, but only *after* all 3 of its minibatches ran; epoch 2 is skipped.
    loss = _FakeLoss([0.001, 0.001, 0.5, 0.001, 0.001, 0.001])
    diagnostics = _run(
        loss, PPOConfig(n_epochs=2, num_minibatches=3, target_kl=0.015, entropy_eps=0.0)
    )
    assert loss.calls == 3
    assert diagnostics["optim/epochs_completed"] == [1.0]


def test_no_early_stop_runs_every_epoch() -> None:
    loss = _FakeLoss([0.001])
    diagnostics = _run(
        loss, PPOConfig(n_epochs=2, num_minibatches=3, target_kl=0.015, entropy_eps=0.0)
    )
    assert loss.calls == 6
    assert diagnostics["optim/epochs_completed"] == [2.0]


def test_target_kl_none_disables_the_guard() -> None:
    loss = _FakeLoss([10.0])
    diagnostics = _run(
        loss, PPOConfig(n_epochs=3, num_minibatches=4, target_kl=None, entropy_eps=0.0)
    )
    assert loss.calls == 12
    assert diagnostics["optim/epochs_completed"] == [3.0]


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
