"""Fixed-budget MAPPO policy proxy: a tier's variants over the shared sweep.

A thin adapter that turns a tier's ``(name, ModelConfig)`` variants into
:class:`~wind_rl.experiment.sweep.Variant` model overrides and runs them under one
identical budget via :func:`~wind_rl.experiment.sweep.run_sweep` -- which owns the
MappoTrainer loop, per-run timing, the between-run wandb ``teardown()`` group fix,
and the windowed deterministic-eval delta.
"""

from __future__ import annotations

from wind_rl.experiment.sweep import SweepResult, Variant, run_sweep
from wind_rl.models import ModelConfig
from wind_rl.rl.trainer import TrainingConfig


def run_policy_proxy(
    base: TrainingConfig,
    variants: list[tuple[str, ModelConfig]],
    seeds: list[int],
) -> SweepResult:
    return run_sweep(
        base, [Variant(name, {"model": model}) for name, model in variants], seeds
    )
