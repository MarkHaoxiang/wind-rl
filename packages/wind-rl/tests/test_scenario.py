from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from wind_rl.scenario import (
    RealFarmConfig,
    ScenarioConfig,
    list_real_farms,
    real_farm_layout,
    resolve_real_farm,
)


def _template() -> ScenarioConfig:
    return ScenarioConfig(
        name="template",
        n_turbines=1,
        max_steps=150,
        map_x_length=1.0,
        map_y_length=1.0,
        min_distance_between_turbines=150.0,
        fixed_wind_direction=270.0,
        load_coef=1.0,
    )


# dict[str, Any]: values span str/int/float for **-unpacking into ScenarioConfig
# (and later corrupted with an invalid type/key by the tests below); no narrower
# union satisfies mypy's per-parameter **kwargs check.
def _base_kwargs() -> dict[str, Any]:  # type: ignore[explicit-any]
    return {
        "name": "s",
        "n_turbines": 4,
        "max_steps": 10,
        "map_x_length": 10.0,
        "map_y_length": 10.0,
        "min_distance_between_turbines": 1.0,
    }


def test_scenario_config_valid() -> None:
    cfg = ScenarioConfig(**_base_kwargs())
    assert cfg.n_turbines == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_turbines", 0),
        ("max_steps", 0),
        ("map_x_length", 0.0),
        ("map_y_length", -1.0),
        ("min_distance_between_turbines", 0.0),
    ],
)
def test_scenario_config_rejects_non_positive(field: str, value: float) -> None:
    kwargs = _base_kwargs()
    kwargs[field] = value
    with pytest.raises(ValidationError):
        ScenarioConfig(**kwargs)


def test_scenario_config_extra_field_raises() -> None:
    kwargs = _base_kwargs()
    kwargs["not_a_field"] = 1
    with pytest.raises(ValidationError):
        ScenarioConfig(**kwargs)


@pytest.mark.sim
def test_list_real_farms_nonempty() -> None:
    farms = list_real_farms()
    assert len(farms) > 0
    assert "Ablaincourt" in farms
    assert "HornsRev1" in farms


@pytest.mark.sim
def test_real_farm_layout_shape_and_finite() -> None:
    layout = real_farm_layout("Ablaincourt")
    assert layout.shape == (7, 2)
    assert layout.dtype == np.float64
    assert np.all(np.isfinite(layout))


@pytest.mark.sim
def test_real_farm_layout_accepts_trailing_underscore() -> None:
    with_underscore = real_farm_layout("Ablaincourt_")
    without = real_farm_layout("Ablaincourt")
    assert np.array_equal(with_underscore, without)


@pytest.mark.sim
def test_real_farm_layout_unknown_raises() -> None:
    with pytest.raises(KeyError):
        real_farm_layout("NotARealFarm")


@pytest.mark.sim
def test_resolve_turb3_row1_is_a_4d_aligned_row() -> None:
    scenario, layout = resolve_real_farm(RealFarmConfig(name="Turb3_Row1"), _template())
    assert scenario.n_turbines == 3
    assert layout.shape == (3, 2)
    # Template (non-geometry) fields are preserved through the resolve.
    assert scenario.max_steps == 150
    assert scenario.load_coef == 1.0
    assert scenario.fixed_wind_direction == 270.0
    # Aligned row (constant y) at 4D = 504 m spacing (D = 126 m).
    assert np.allclose(layout[:, 1], layout[0, 1])
    spacings = np.diff(layout[:, 0])
    assert np.allclose(spacings, 504.0)


@pytest.mark.sim
def test_resolve_ablaincourt_shape_and_in_bounds() -> None:
    scenario, layout = resolve_real_farm(
        RealFarmConfig(name="Ablaincourt"), _template()
    )
    assert scenario.n_turbines == 7
    assert layout.shape == (7, 2)
    # Translated fully in-map (every coordinate positive and within map bounds).
    assert np.all(layout > 0.0)
    assert np.all(layout[:, 0] <= scenario.map_x_length)
    assert np.all(layout[:, 1] <= scenario.map_y_length)
