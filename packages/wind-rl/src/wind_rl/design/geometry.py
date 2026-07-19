"""Layout feasibility (min-distance + in-bounds) and feasible-layout sampling.

Single source of truth for what makes a turbine layout valid in a scenario's
map, shared by the designers and their tests.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wind_rl.scenario import ScenarioConfig


def pairwise_min_distance(layout: NDArray[np.float64]) -> float:
    if layout.shape[0] < 2:
        return float("inf")
    dist = np.linalg.norm(layout[:, None, :] - layout[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def within_bounds(
    layout: NDArray[np.float64], map_x_length: float, map_y_length: float
) -> bool:
    x, y = layout[:, 0], layout[:, 1]
    return bool(
        np.all(x >= 0.0)
        and np.all(x <= map_x_length)
        and np.all(y >= 0.0)
        and np.all(y <= map_y_length)
    )


def is_feasible(layout: NDArray[np.float64], scenario: ScenarioConfig) -> bool:
    return (
        within_bounds(layout, scenario.map_x_length, scenario.map_y_length)
        and pairwise_min_distance(layout) >= scenario.min_distance_between_turbines
    )


def sample_feasible_layout(
    scenario: ScenarioConfig,
    rng: np.random.Generator,
    max_attempts_per_turbine: int = 1000,
) -> NDArray[np.float64]:
    """Sample a feasible ``(n_turbines, 2)`` layout by min-distance rejection.

    Turbines are placed one at a time, rejecting candidates that violate the
    min-distance constraint. Raises if a turbine cannot be placed within
    ``max_attempts_per_turbine`` tries, so an infeasible scenario (min distance
    too large for the map) fails fast rather than looping forever.
    """
    n = scenario.n_turbines
    min_distance = scenario.min_distance_between_turbines
    high = np.array([scenario.map_x_length, scenario.map_y_length])
    coords = np.empty((n, 2), dtype=np.float64)

    placed = 0
    attempts = 0
    while placed < n:
        candidate = rng.uniform(low=0.0, high=high)
        if placed == 0 or bool(
            np.all(np.linalg.norm(coords[:placed] - candidate, axis=-1) >= min_distance)
        ):
            coords[placed] = candidate
            placed += 1
            attempts = 0
        else:
            attempts += 1
            if attempts >= max_attempts_per_turbine:
                raise RuntimeError(
                    f"Could not place turbine {placed + 1}/{n} within "
                    f"{max_attempts_per_turbine} attempts; scenario "
                    f"{scenario.name!r} may be infeasible (min distance too large "
                    "for the map)."
                )
    return coords
