from importlib.resources import files
from typing import Any

import pytest
import yaml

from windrl_engine.physics import deficit, deflection, flow, frame, turbulence


@pytest.fixture(scope="module")
def floris_defaults() -> dict[str, Any]:
    text = (files("floris") / "default_inputs.yaml").read_text()
    return dict(yaml.safe_load(text))


def test_deflection_parameters_match_floris_wake_deflection_gauss(
    floris_defaults: dict[str, Any],
) -> None:
    gauss = floris_defaults["wake"]["wake_deflection_parameters"]["gauss"]
    assert gauss["alpha"] == deflection.ALPHA
    assert gauss["beta"] == deflection.BETA
    assert gauss["ka"] == deflection.KA
    assert gauss["kb"] == deflection.KB
    assert gauss["dm"] == deflection.DM


def test_deficit_parameters_match_floris_wake_velocity_gauss(
    floris_defaults: dict[str, Any],
) -> None:
    # Asserted against wake_velocity_parameters, never against the deflection block:
    # the two sets are independently overridable and equal only by shipped default.
    gauss = floris_defaults["wake"]["wake_velocity_parameters"]["gauss"]
    assert gauss["alpha"] == deficit.ALPHA
    assert gauss["beta"] == deficit.BETA
    assert gauss["ka"] == deficit.KA
    assert gauss["kb"] == deficit.KB


def test_crespo_hernandez_parameters_match_floris_wake_turbulence(
    floris_defaults: dict[str, Any],
) -> None:
    # The yaml is authoritative over the attrs class defaults (which say constant=0.9)
    # because FlorisModel("defaults") -- our reference -- loads this file.
    crespo = floris_defaults["wake"]["wake_turbulence_parameters"]["crespo_hernandez"]
    assert crespo["initial"] == turbulence.CRESPO_INITIAL
    assert crespo["constant"] == turbulence.CRESPO_CONSTANT
    assert crespo["ai"] == turbulence.CRESPO_AI
    assert crespo["downstream"] == turbulence.CRESPO_DOWNSTREAM


def test_flow_parameters_match_floris_flow_field(
    floris_defaults: dict[str, Any],
) -> None:
    flow_field = floris_defaults["flow_field"]
    assert flow_field["wind_shear"] == flow.SHEAR
    assert flow_field["air_density"] == flow.AIR_DENSITY


def test_rotor_grid_resolution_matches_floris_solver(
    floris_defaults: dict[str, Any],
) -> None:
    assert floris_defaults["solver"]["turbine_grid_points"] == frame.GRID
