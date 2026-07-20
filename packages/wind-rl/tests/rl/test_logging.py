from __future__ import annotations

import math

import torch

from wind_rl.rl.logging import clip_fraction, explained_variance


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


def test_clip_fraction_counts_ratios_outside_band() -> None:
    eps = 0.2
    # ratios 0.5 and 1.6 fall outside [0.8, 1.2]; 1.0 and 1.1 stay inside.
    log_ratio = torch.log(torch.tensor([0.5, 1.0, 1.1, 1.6]))
    assert clip_fraction(log_ratio, eps) == 0.5
    assert clip_fraction(torch.zeros(8), eps) == 0.0
    # Exactly on the band edge is not clipped (clamp is a no-op there).
    assert clip_fraction(torch.tensor([math.log1p(eps)]), eps) == 0.0
