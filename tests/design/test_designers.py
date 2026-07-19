from __future__ import annotations

import numpy as np
import pytest
from pydantic import TypeAdapter

from wind_rl.design import (
    Designer,
    DesignerConfig,
    FixedDesigner,
    FixedDesignerConfig,
    ManualDesigner,
    ManualDesignerConfig,
    RandomDesigner,
    RandomDesignerConfig,
    create_designer,
    is_feasible,
    sample_feasible_layout,
)
from wind_rl.scenario import ScenarioConfig, real_farm_layout

N_TURBINES = 6
BATCH_SIZE = 4


@pytest.fixture
def scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="test",
        n_turbines=N_TURBINES,
        max_steps=10,
        map_x_length=2000.0,
        map_y_length=2000.0,
        min_distance_between_turbines=300.0,
    )


def _assert_feasible_batch(batch: np.ndarray, scenario: ScenarioConfig) -> None:
    assert batch.shape == (BATCH_SIZE, scenario.n_turbines, 2)
    for layout in batch:
        assert is_feasible(layout, scenario)


def test_random_designer_produces_feasible_batch(scenario: ScenarioConfig) -> None:
    batch = RandomDesigner(scenario, seed=0).generate_layout_batch(BATCH_SIZE)
    _assert_feasible_batch(batch, scenario)


def test_random_designer_layouts_differ(scenario: ScenarioConfig) -> None:
    batch = RandomDesigner(scenario, seed=0).generate_layout_batch(BATCH_SIZE)
    assert not np.allclose(batch[0], batch[1])


def test_fixed_designer_is_constant_and_feasible(scenario: ScenarioConfig) -> None:
    layout = sample_feasible_layout(scenario, np.random.default_rng(1))
    batch = FixedDesigner(layout).generate_layout_batch(BATCH_SIZE)
    _assert_feasible_batch(batch, scenario)
    for one in batch:
        np.testing.assert_array_equal(one, layout)


def test_manual_designer_matches_horns_rev1() -> None:
    expected = real_farm_layout("HornsRev1")
    batch = ManualDesigner("HornsRev1").generate_layout_batch(BATCH_SIZE)
    assert batch.shape == (BATCH_SIZE, expected.shape[0], 2)
    for layout in batch:
        np.testing.assert_array_equal(layout, expected)


def test_sample_feasible_layout_raises_when_infeasible() -> None:
    infeasible = ScenarioConfig(
        name="tight",
        n_turbines=50,
        max_steps=10,
        map_x_length=100.0,
        map_y_length=100.0,
        min_distance_between_turbines=90.0,
    )
    with pytest.raises(RuntimeError):
        sample_feasible_layout(
            infeasible, np.random.default_rng(0), max_attempts_per_turbine=50
        )


@pytest.mark.parametrize(
    "cfg,expected",
    [
        (RandomDesignerConfig(seed=0), RandomDesigner),
        (FixedDesignerConfig(seed=0), FixedDesigner),
        (ManualDesignerConfig(farm="HornsRev1"), ManualDesigner),
    ],
)
def test_create_designer_dispatches(
    cfg: DesignerConfig, expected: type[Designer], scenario: ScenarioConfig
) -> None:
    assert isinstance(create_designer(cfg, scenario), expected)


def test_designer_config_discriminated_union_parses() -> None:
    adapter: TypeAdapter[DesignerConfig] = TypeAdapter(DesignerConfig)
    assert isinstance(adapter.validate_python({"kind": "random"}), RandomDesignerConfig)
    assert isinstance(adapter.validate_python({"kind": "fixed"}), FixedDesignerConfig)
    manual = adapter.validate_python({"kind": "manual", "farm": "HornsRev1"})
    assert isinstance(manual, ManualDesignerConfig)
