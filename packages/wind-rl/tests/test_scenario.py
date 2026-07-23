from typing import Any

import pytest
from pydantic import ValidationError

from wind_rl.scenario import ScenarioConfig


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
