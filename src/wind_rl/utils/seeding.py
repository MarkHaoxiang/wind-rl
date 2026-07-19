"""Global seeding for reproducible experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed python's ``random``, ``numpy``, and ``torch`` (including all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
