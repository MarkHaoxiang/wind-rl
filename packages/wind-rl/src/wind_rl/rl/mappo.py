"""PPO loss and optimiser construction for MAPPO on the ``turbine`` group."""

from __future__ import annotations

import torch
from tensordict.utils import NestedKey
from torch import optim
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
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    n_epochs: int = 4
    num_minibatches: int = 4
    lr_scheduler: bool = False
    min_lr: float = 0.0


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
        normalize_advantage=cfg.normalize_advantage,
        # Advantages are normalised over time/batch but not across agents.
        normalize_advantage_exclude_dims=(-2,),
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


def build_optimiser(
    loss_module: ClipPPOLoss, cfg: PPOConfig, n_iters: int
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler | None]:
    optimiser = optim.Adam(loss_module.parameters(), lr=cfg.lr)
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=n_iters, eta_min=cfg.min_lr
        )
        if cfg.lr_scheduler
        else None
    )
    return optimiser, scheduler
