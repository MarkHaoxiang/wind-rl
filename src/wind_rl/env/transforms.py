"""Reward-normalisation TorchRL transform.

DiCoDe normalised rewards in a standalone script computed from random-policy
rollouts; here it becomes a proper env :class:`~torchrl.envs.transforms.Transform`
so normalisation travels with the environment. Only the *precomputed* mode
(fixed mean/std) is implemented for now, but the class is structured so a
*running* (Welford-style) mode can be added behind the same interface.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.envs.transforms import Transform

from wind_rl.env.windfarm import GROUP_NAME

_DEFAULT_REWARD_KEY: NestedKey = (GROUP_NAME, "reward")


class RewardNormalisation(Transform):  # type: ignore[misc]
    """Normalise a reward by a (precomputed) mean and standard deviation.

    The transform applies ``(reward - mean) / std`` in-place on the configured
    reward key. When either ``mean`` or ``std`` is ``None`` it is an identity
    transform (useful as a no-op placeholder in a fixed :class:`Compose`
    pipeline before statistics have been measured).

    Parameters
    ----------
    mean, std:
        Precomputed reward statistics. Pass both to enable normalisation, or
        leave either ``None`` for an identity transform.
    eps:
        Floor applied to ``std`` to avoid division by zero.
    """

    def __init__(
        self,
        mean: float | None = None,
        std: float | None = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        *,
        eps: float = 1e-6,
    ) -> None:
        if in_keys is None:
            in_keys = [_DEFAULT_REWARD_KEY]
        if out_keys is None:
            out_keys = list(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)

        self.enabled = mean is not None and std is not None
        loc = 0.0 if mean is None else float(mean)
        scale = 1.0 if std is None else max(float(std), eps)
        self.register_buffer("loc", torch.tensor(loc))
        self.register_buffer("scale", torch.tensor(scale))

    def _apply_transform(self, reward: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return reward
        normalised: torch.Tensor = (
            reward - self.loc.to(reward.device)
        ) / self.scale.to(reward.device)
        return normalised

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # No per-episode state to reset in precomputed mode; a running mode would
        # (optionally) reset its accumulators here.
        return tensordict_reset
