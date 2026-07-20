from __future__ import annotations

import math

import torch
from torch import nn

from wind_rl.rl.logging import checkpoint_aliases, explained_variance, param_norms


def test_explained_variance_ranges_from_perfect_to_negative() -> None:
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(target, target.clone()) == 1.0
    # Predicting the mean explains none of the variance.
    mean_pred = torch.full_like(target, float(target.mean()))
    assert explained_variance(target, mean_pred) == 0.0
    # Anti-correlated predictions do worse than the mean -> negative.
    assert explained_variance(target, target.flip(0)) < 0.0
    # A constant target has zero variance -> guarded to 0.0, not NaN/inf.
    assert explained_variance(torch.ones(4), torch.zeros(4)) == 0.0


def test_param_norms_matches_manual_l2_and_sums_to_total() -> None:
    first, second = nn.Linear(2, 3, bias=False), nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        first.weight.fill_(2.0)  # sqrt(2*2*2*3) = sqrt(24)
        second.weight.fill_(1.0)  # sqrt(1*1*3) = sqrt(3)
    module = nn.Sequential(first, second)

    norms = param_norms(module)

    assert norms["0"] == math.sqrt(24.0)
    assert norms["1"] == math.sqrt(3.0)
    assert norms["total"] == math.sqrt(27.0)


def test_checkpoint_aliases_tags_iteration_and_final() -> None:
    # Default cadence (interval=1): every iteration gets an iter-<N> alias.
    assert checkpoint_aliases(0, 1, is_final=False) == ["iter-0"]
    assert checkpoint_aliases(3, 1, is_final=False) == ["iter-3"]
    # Coarser cadence skips iterations not divisible by the interval.
    assert checkpoint_aliases(1, 2, is_final=False) == []
    assert checkpoint_aliases(2, 2, is_final=False) == ["iter-2"]
    # Final always gets tagged, alongside iter-<N> when the cadence also hits.
    assert checkpoint_aliases(5, None, is_final=True) == ["final"]
    assert checkpoint_aliases(4, 2, is_final=True) == ["iter-4", "final"]
    # Periodic uploads disabled and not final -> nothing to upload.
    assert checkpoint_aliases(4, None, is_final=False) == []
