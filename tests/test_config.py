from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from wind_rl.scenario import ScenarioConfig

SCENARIO_YAML = """
name: test_scenario
n_turbines: 4
max_steps: 100
map_x_length: 1000.0
map_y_length: 1000.0
min_distance_between_turbines: 200.0
"""


def _write_scenario_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(SCENARIO_YAML)
    return path


def test_from_file_round_trip(tmp_path: Path) -> None:
    cfg = ScenarioConfig.from_file(_write_scenario_yaml(tmp_path))
    assert cfg.name == "test_scenario"
    assert cfg.n_turbines == 4
    assert cfg.max_steps == 100
    assert cfg.min_distance_between_turbines == 200.0


def test_from_file_dotlist_override(tmp_path: Path) -> None:
    path = _write_scenario_yaml(tmp_path)
    cfg = ScenarioConfig.from_file(path, overrides=["n_turbines=8"])
    assert cfg.n_turbines == 8
    # unrelated fields are untouched
    assert cfg.name == "test_scenario"


def test_from_raw_round_trip() -> None:
    cfg = ScenarioConfig.from_raw(
        OmegaConf.create(
            {
                "name": "raw",
                "n_turbines": 3,
                "max_steps": 50,
                "map_x_length": 500.0,
                "map_y_length": 500.0,
                "min_distance_between_turbines": 100.0,
            }
        )
    )
    assert cfg.name == "raw"
    assert cfg.n_turbines == 3


def test_from_raw_extra_field_raises() -> None:
    raw = OmegaConf.create(
        {
            "name": "s",
            "n_turbines": 4,
            "max_steps": 10,
            "map_x_length": 10.0,
            "map_y_length": 10.0,
            "min_distance_between_turbines": 1.0,
            "unexpected": 1,
        }
    )
    with pytest.raises(ValidationError):
        ScenarioConfig.from_raw(raw)
