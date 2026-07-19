"""Torch device resolution."""

from __future__ import annotations

import torch


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve a torch device: explicit ``device`` > CUDA if available > CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
