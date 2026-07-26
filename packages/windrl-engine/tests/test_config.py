"""WindFarmEnvConfig validation: what pydantic rejects before an env is ever built."""

from typing import Any

import pytest
from pydantic import ValidationError

from windrl_engine.env.config import WindFarmEnvConfig

# The config is the only guard between a user's yaml/CLI and jitted code, where a
# bad value surfaces as a shape error or a silently wrong rollout. Each entry is
# a field the type alone does not constrain, or a name that must not typo through.
REJECTED: list[tuple[str, dict[str, Any]]] = [
    ("unknown_field", {"yaw_stpe": 5.0}),
    ("yaw_step_zero", {"yaw_step": 0.0}),
    ("yaw_step_negative", {"yaw_step": -5.0}),
    ("horizon_zero", {"horizon": 0}),
    ("n_envs_zero", {"n_envs": 0}),
    ("load_coef_negative", {"load_coef": -0.1}),
    ("unknown_layout", {"layout": "hornsrev2"}),
    ("unknown_control_mode", {"control_mode": "bang_bang"}),
    ("unknown_fidelity", {"fidelity": "exact"}),
    ("unknown_turbine", {"turbine": "v80"}),
]


@pytest.mark.parametrize(
    ("case_id", "kwargs"), [pytest.param(*c, id=c[0]) for c in REJECTED]
)
def test_invalid_config_raises_instead_of_silently_defaulting(
    case_id: str, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        WindFarmEnvConfig(**kwargs)


def test_an_explicit_layout_becomes_the_coordinates_it_was_given() -> None:
    layout = WindFarmEnvConfig(layout=[(0.0, 10.0), (504.0, -10.0)]).build_layout()
    assert [float(x) for x in layout.x] == [0.0, 504.0]
    assert [float(y) for y in layout.y] == [10.0, -10.0]
