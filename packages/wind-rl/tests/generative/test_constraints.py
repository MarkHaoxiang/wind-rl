from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wind_rl.design.geometry import pairwise_min_distance
from wind_rl.generative.constraints import project_soft
from wind_rl.scenario import ScenarioConfig


def _scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="gen3",
        n_turbines=3,
        max_steps=10,
        map_x_length=2000.0,
        map_y_length=2000.0,
        min_distance_between_turbines=400.0,
    )


def _violation(layout: NDArray[np.float64], min_d: float) -> float:
    return max(0.0, min_d - pairwise_min_distance(layout))


def test_project_soft_reduces_violation_and_is_idempotent_on_feasible() -> None:
    scenario = _scenario()
    min_d = scenario.min_distance_between_turbines

    infeasible = np.array(
        [[[500.0, 500.0], [520.0, 500.0], [1500.0, 1500.0]]], dtype=np.float64
    )
    before = _violation(infeasible[0], min_d)
    assert before > 0.0

    projected = project_soft(infeasible, scenario)
    after = _violation(projected[0], min_d)
    assert after < before

    reprojected = project_soft(projected, scenario)
    assert _violation(projected[0], min_d) == 0.0
    assert np.allclose(projected, reprojected, atol=1e-2)
