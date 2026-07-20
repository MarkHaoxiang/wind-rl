"""Fixed-budget MAPPO policy proxy: the tier's variants over the shared sweep.

0003-specific glue only. It turns a tier's ``(name, ModelConfig)`` variants into
:class:`~wind_rl.experiment.sweep.Variant` overrides and runs them under one
identical budget via :func:`~wind_rl.experiment.sweep.run_sweep` -- which already
owns the MappoTrainer loop, per-run timing, the between-run wandb ``teardown()``
group fix, and the windowed deterministic-eval delta. The returned
:class:`~wind_rl.experiment.sweep.SweepResult` feeds the combined critic+policy
comparison table and the joint functional verdict in ``run.py``.
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
