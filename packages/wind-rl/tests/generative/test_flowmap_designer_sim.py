from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("wfcrl")

from wind_rl.design.designers import FlowMapDesigner
from wind_rl.design.geometry import is_feasible
from wind_rl.env import make_env
from wind_rl.generative.flowmap import save_flowmap, train_flowmap_prior
from wind_rl.scenario import ScenarioConfig

pytestmark = pytest.mark.sim

N_TURBINES = 3


def test_flowmap_designer_drives_feasible_varying_env_resets(tmp_path: Path) -> None:
    scenario = ScenarioConfig(
        name="codesign_fm3",
        n_turbines=N_TURBINES,
        max_steps=10,
        map_x_length=2000.0,
        map_y_length=2000.0,
        min_distance_between_turbines=400.0,
    )
    model, _ = train_flowmap_prior(
        scenario, n_samples=256, n_iters=300, batch_size=128, seed=0
    )
    ckpt = tmp_path / "prior.pt"
    save_flowmap(model, str(ckpt))

    designer = FlowMapDesigner(scenario, ckpt, sampling_steps=4)
    env = make_env("train", scenario, reset_policy=designer.to_td_module())

    layout_a = env.reset()["state", "layout"].numpy()
    layout_b = env.reset()["state", "layout"].numpy()

    assert layout_a.shape == (N_TURBINES, 2)
    assert not np.allclose(layout_a, layout_b)
    for layout in (layout_a, layout_b):
        assert is_feasible(layout.astype(np.float64), scenario)
    assert designer.get_logs()["total_nfe"] > 0
