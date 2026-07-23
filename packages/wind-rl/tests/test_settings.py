from pathlib import Path

import pytest

from wind_rl.experiment.settings import WindRlSettings


def test_default_wdir() -> None:
    settings = WindRlSettings()
    assert settings.wdir == Path("outputs")
    assert settings.wandb_mode == "online"


def test_env_var_overrides_wdir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIND_RL_WDIR", "/tmp/custom_wind_rl")
    settings = WindRlSettings()
    assert settings.wdir == Path("/tmp/custom_wind_rl")


def test_env_var_overrides_wandb_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIND_RL_WANDB_MODE", "disabled")
    settings = WindRlSettings()
    assert settings.wandb_mode == "disabled"


def test_resolved_wdir_expands_user() -> None:
    settings = WindRlSettings(wdir=Path("~/.wind_rl"))
    assert "~" not in str(settings.resolved_wdir)
    assert settings.resolved_wdir.is_absolute()
