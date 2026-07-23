"""Feasible-layout geometry: min-distance + in-bounds checks and sampling."""

from wind_rl.design.geometry import (
    is_feasible,
    pairwise_min_distance,
    sample_feasible_layout,
    within_bounds,
)

__all__ = [
    "is_feasible",
    "pairwise_min_distance",
    "sample_feasible_layout",
    "within_bounds",
]
