from __future__ import annotations

import math

import pytest
import torch
import wandb
from torch import nn

from wind_rl.experiment.settings import WindRlSettings
from wind_rl.rl.logging import (
    RunLogger,
    checkpoint_aliases,
    explained_variance,
    param_norms,
)
from wind_rl.rl.trainer import LoggingConfig, TrainingConfig
from wind_rl.scenario import ScenarioConfig


def test_explained_variance_ranges_from_perfect_to_negative() -> None:
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(target, target.clone()) == 1.0
    # Predicting the mean explains none of the variance.
    mean_pred = torch.full_like(target, float(target.mean()))
    assert explained_variance(target, mean_pred) == 0.0
    # Anti-correlated predictions do worse than the mean -> negative.
    assert explained_variance(target, target.flip(0)) < 0.0
    # A constant target has zero variance -> guarded to 0.0, not NaN/inf.
    assert explained_variance(torch.ones(4), torch.zeros(4)) == 0.0


def test_param_norms_matches_manual_l2_and_sums_to_total() -> None:
    first, second = nn.Linear(2, 3, bias=False), nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        first.weight.fill_(2.0)  # sqrt(2*2*2*3) = sqrt(24)
        second.weight.fill_(1.0)  # sqrt(1*1*3) = sqrt(3)
    module = nn.Sequential(first, second)

    norms = param_norms(module)

    assert norms["0"] == math.sqrt(24.0)
    assert norms["1"] == math.sqrt(3.0)
    assert norms["total"] == math.sqrt(27.0)


def test_checkpoint_aliases_tags_iteration_and_final() -> None:
    # Default cadence (interval=1): every iteration gets an iter-<N> alias.
    assert checkpoint_aliases(0, 1, is_final=False) == ["iter-0"]
    assert checkpoint_aliases(3, 1, is_final=False) == ["iter-3"]
    # Coarser cadence skips iterations not divisible by the interval.
    assert checkpoint_aliases(1, 2, is_final=False) == []
    assert checkpoint_aliases(2, 2, is_final=False) == ["iter-2"]
    # Final always gets tagged, alongside iter-<N> when the cadence also hits.
    assert checkpoint_aliases(5, None, is_final=True) == ["final"]
    assert checkpoint_aliases(4, 2, is_final=True) == ["iter-4", "final"]
    # Periodic uploads disabled and not final -> nothing to upload.
    assert checkpoint_aliases(4, None, is_final=False) == []


def _training_config(logging: LoggingConfig) -> TrainingConfig:
    return TrainingConfig(
        experiment_name="test_run_logger",
        scenario=ScenarioConfig(
            name="smoke3",
            n_turbines=3,
            max_steps=8,
            map_x_length=2000.0,
            map_y_length=2000.0,
            min_distance_between_turbines=400.0,
        ),
        logging=logging,
    )


def test_run_logger_forwards_group_job_type_name_and_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeRun:
        url = "http://fake"

        def define_metric(self, *args: object, **kwargs: object) -> None:
            pass

    def _fake_init(**kwargs: object) -> _FakeRun:
        captured.update(kwargs)
        return _FakeRun()

    monkeypatch.setattr(wandb, "init", _fake_init)
    cfg = _training_config(
        LoggingConfig(
            use_wandb=True,
            run_name="ablaincourt-mlp-s0",
            group="0001_wfcrl_baseline",
            job_type="ablaincourt",
            tags=["ablaincourt", "mlp", "seed0"],
        )
    )

    RunLogger(cfg, WindRlSettings(wandb_mode="offline"))

    assert captured["name"] == "ablaincourt-mlp-s0"
    assert captured["group"] == "0001_wfcrl_baseline"
    assert captured["job_type"] == "ablaincourt"
    assert captured["tags"] == ["ablaincourt", "mlp", "seed0"]
    config = captured["config"]
    assert isinstance(config, dict)
    assert config["seed"] == 0
    scenario = config["scenario"]
    assert isinstance(scenario, dict)
    assert scenario["n_turbines"] == 3


def test_run_logger_name_falls_back_to_experiment_name() -> None:
    cfg = _training_config(LoggingConfig(use_wandb=True))
    assert cfg.logging.run_name is None
    assert cfg.logging.group is None
    assert cfg.logging.tags == []
