from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from wind_rl.scenario import ScenarioConfig, list_real_farms, real_farm_layout


def _base_kwargs() -> dict[str, Any]:
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


def test_list_real_farms_nonempty() -> None:
    farms = list_real_farms()
    assert len(farms) > 0
    assert "Ablaincourt" in farms
    assert "HornsRev1" in farms


def test_real_farm_layout_shape_and_finite() -> None:
    layout = real_farm_layout("Ablaincourt")
    assert layout.shape == (7, 2)
    assert layout.dtype == np.float64
    assert np.all(np.isfinite(layout))


def test_real_farm_layout_accepts_trailing_underscore() -> None:
    with_underscore = real_farm_layout("Ablaincourt_")
    without = real_farm_layout("Ablaincourt")
    assert np.array_equal(with_underscore, without)


def test_real_farm_layout_unknown_raises() -> None:
    with pytest.raises(KeyError):
        real_farm_layout("NotARealFarm")
