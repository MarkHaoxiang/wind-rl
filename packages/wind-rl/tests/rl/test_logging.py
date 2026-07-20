from __future__ import annotations

import torch

from wind_rl.rl.logging import explained_variance


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
