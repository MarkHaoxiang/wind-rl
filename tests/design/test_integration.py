"""End-to-end: a designer's reset policy drives layout at every env reset."""

from __future__ import annotations

import numpy as np
import pytest

from wind_rl.design import RandomDesigner, is_feasible
from wind_rl.env import make_env
from wind_rl.scenario import ScenarioConfig

N_TURBINES = 3


@pytest.fixture
def scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="codesign3",
        n_turbines=N_TURBINES,
        max_steps=10,
        map_x_length=2000.0,
        map_y_length=2000.0,
        min_distance_between_turbines=400.0,
    )


def test_random_designer_reset_policy_closes_codesign_loop(
    scenario: ScenarioConfig,
) -> None:
    designer = RandomDesigner(scenario, seed=0)
    env = make_env("train", scenario, reset_policy=designer.to_td_module())

    layout_a = env.reset()["state", "layout"].numpy()
    layout_b = env.reset()["state", "layout"].numpy()

    assert layout_a.shape == (N_TURBINES, 2)
    assert not np.allclose(layout_a, layout_b)
    for layout in (layout_a, layout_b):
        assert is_feasible(layout.astype(np.float64), scenario)
