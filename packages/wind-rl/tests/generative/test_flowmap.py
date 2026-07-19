from __future__ import annotations

import numpy as np

from wind_rl.design.geometry import is_feasible
from wind_rl.generative.constraints import project_slsqp
from wind_rl.generative.flowmap import sample_layouts, train_flowmap_prior
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


def test_sampled_layouts_project_feasible_and_projection_is_idempotent() -> None:
    scenario = _scenario()
    model, _ = train_flowmap_prior(
        scenario, n_samples=256, n_iters=200, batch_size=128, seed=0
    )
    raw = sample_layouts(model, 32, 4, seed=1)
    assert raw.shape == (32, scenario.n_turbines, 2)

    projected = project_slsqp(raw, scenario)
    assert projected.shape == raw.shape
    for layout in projected:
        assert is_feasible(layout, scenario)

    # A feasible layout is a fixed point of the projection.
    reprojected = project_slsqp(projected, scenario)
    assert np.allclose(projected, reprojected, atol=1e-3)


def test_consistency_loss_strictly_decreases_over_training() -> None:
    scenario = _scenario()
    _, history = train_flowmap_prior(
        scenario, n_samples=512, n_iters=600, batch_size=128, seed=0
    )
    window = len(history) // 5
    first = float(np.mean(history[:window]))
    last = float(np.mean(history[-window:]))
    assert last < 0.9 * first
