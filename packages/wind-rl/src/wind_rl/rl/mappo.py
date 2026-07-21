"""PPO loss and optimiser construction for MAPPO on the ``turbine`` group."""

from __future__ import annotations

from typing import Literal

import torch
from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torchrl.data import ReplayBuffer
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from wind_rl.config import Config
from wind_rl.static import GROUP_NAME

_VALUE_KEY: NestedKey = (GROUP_NAME, "state_value")
_LOG_PROB_KEY: NestedKey = (GROUP_NAME, "sample_log_prob")
_DONE_KEY: NestedKey = (GROUP_NAME, "done")
_TERMINATED_KEY: NestedKey = (GROUP_NAME, "terminated")


class PPOConfig(Config):
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.9
    entropy_eps: float = 1e-3
    normalize_advantage: bool = True
    #: Value-function loss weight (torchrl ``critic_coeff``); the WFCRL benchmark
    #: uses 0.5.
    vf_coef: float = 1.0
    #: Clip the value prediction by ``clip_epsilon`` (PPO's clipped value loss).
    clip_value_loss: bool = False
    lr: float = 3e-4
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    n_epochs: int = 4
    num_minibatches: int = 4
    #: KL trust-region guard (CleanRL's ``--target-kl``). When set, the
    #: ``n_epochs x num_minibatches`` update halts as soon as a minibatch's
    #: ``approx_kl`` exceeds this, so a single runaway batch cannot keep pushing
    #: the policy across the whole update. ``None`` disables the guard (the paper's
    #: Table 5 default). See :func:`wind_rl.rl.trainer._ppo_epochs`.
    target_kl: float | None = None
    #: Standardise each rollout's rewards (over the whole collected batch) before
    #: GAE -- recomputed fresh per update, not a running normaliser.
    reward_batch_norm: bool = False
    lr_anneal: Literal["none", "cosine", "linear"] = "none"
    #: Floor for the cosine schedule (``eta_min``); ignored for linear/none.
    min_lr: float = 0.0


def batch_normalise_reward(reward: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """``(reward - batch_mean) / (batch_std + eps)`` over the whole tensor.

    Matches the WFCRL benchmark's per-rollout reward standardisation: the mean
    and std are recomputed from the batch each update, so this is a pure function
    of one rollout, not a running statistic.
    """
    return (reward - reward.mean()) / (reward.std(unbiased=False) + eps)


def build_loss_module(
    policy: torch.nn.Module,
    critic: torch.nn.Module,
    cfg: PPOConfig,
    action_key: NestedKey,
    reward_key: NestedKey,
) -> ClipPPOLoss:
    """A :class:`ClipPPOLoss` wired to the group's keys with a GAE value estimator."""
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=cfg.clip_epsilon,
        entropy_bonus=cfg.entropy_eps > 0,
        entropy_coeff=cfg.entropy_eps,
        critic_coeff=cfg.vf_coef,
        normalize_advantage=cfg.normalize_advantage,
        # Advantages are normalised over time/batch but not across agents.
        normalize_advantage_exclude_dims=(-2,),
        clip_value=cfg.clip_epsilon if cfg.clip_value_loss else None,
    )
    loss_module.set_keys(
        reward=reward_key,
        action=action_key,
        sample_log_prob=_LOG_PROB_KEY,
        value=_VALUE_KEY,
        terminated=_TERMINATED_KEY,
        done=_DONE_KEY,
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.gamma, lmbda=cfg.lmbda
    )
    return loss_module


def run_ppo_epochs(
    loss_module: torch.nn.Module,
    replay_buffer: ReplayBuffer,
    optimiser: optim.Optimizer,
    cfg: PPOConfig,
) -> tuple[list[float], dict[str, list[float]]]:
    """Run PPO's ``n_epochs x num_minibatches`` update over the sampled batch.

    When ``cfg.target_kl`` is set, the whole update halts (both loops) the moment a
    minibatch's ``approx_kl`` exceeds it -- CleanRL's ``--target-kl`` trust-region
    guard, checked *per minibatch* (finer than CleanRL's per-epoch check) so a
    single runaway batch cannot keep pushing the policy across the remaining
    epochs. ``optim/kl_early_stop`` records whether the guard fired this update.
    """
    grad_norms: list[float] = []
    diagnostics: dict[str, list[float]] = {}
    stopped = False
    for _ in range(cfg.n_epochs):
        for _ in range(cfg.num_minibatches):
            minibatch = replay_buffer.sample()
            loss_vals = loss_module(minibatch)
            loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"]
            if cfg.entropy_eps > 0:
                loss_value = loss_value + loss_vals["loss_entropy"]
            loss_value.backward()
            grad_norms.append(
                float(clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm))
            )
            optimiser.step()
            optimiser.zero_grad()
            _accumulate_diagnostics(diagnostics, loss_vals, loss_value)
            if (
                cfg.target_kl is not None
                and "kl_approx" in loss_vals.keys()  # noqa: SIM118 - TensorDict keys view
                and float(loss_vals["kl_approx"].mean()) > cfg.target_kl
            ):
                stopped = True
                break
        if stopped:
            break
    diagnostics.setdefault("optim/kl_early_stop", []).append(1.0 if stopped else 0.0)
    return grad_norms, diagnostics


def _accumulate_diagnostics(
    diagnostics: dict[str, list[float]],
    loss_vals: TensorDictBase,
    total: torch.Tensor,
) -> None:
    diagnostics.setdefault("loss/total", []).append(float(total.mean()))
    keys = {
        "loss/objective": "loss_objective",
        "loss/critic": "loss_critic",
        "loss/clip_fraction": "clip_fraction",
        "loss/approx_kl": "kl_approx",
        "loss/entropy": "loss_entropy",
        "loss/explained_variance": "explained_variance",
    }
    for metric, key in keys.items():
        if key in loss_vals.keys():  # noqa: SIM118 - TensorDict keys view
            diagnostics.setdefault(metric, []).append(float(loss_vals[key].mean()))


def build_optimiser(
    loss_module: ClipPPOLoss, cfg: PPOConfig, n_iters: int
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler | None]:
    optimiser = optim.Adam(loss_module.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    scheduler: optim.lr_scheduler.LRScheduler | None
    if cfg.lr_anneal == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=n_iters, eta_min=cfg.min_lr
        )
    elif cfg.lr_anneal == "linear":
        # Anneal linearly to 0 over the run (WFCRL's ``lr * (1 - iter/n_iters)``);
        # scheduler.step() is called once per iteration.
        scheduler = optim.lr_scheduler.LinearLR(
            optimiser, start_factor=1.0, end_factor=0.0, total_iters=n_iters
        )
    else:
        scheduler = None
    return optimiser, scheduler
