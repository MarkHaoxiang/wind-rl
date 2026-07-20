"""Run one MAPPO training per (variant, seed) and harvest comparable per-run metrics.

This is the shared loop behind the fixed-layout benchmark frameworks: given a base
:class:`~wind_rl.rl.trainer.TrainingConfig` and a list of :class:`Variant` overrides,
it trains every variant across every seed, times each run, and reduces each run's
metric history to a small typed :class:`RunResult` (windowed learning delta, eval
AUC, wall-clock, finiteness). Aggregation into a comparison table and pass/fail
gating live in :mod:`~wind_rl.experiment.table` and
:mod:`~wind_rl.experiment.verdict`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from statistics import fmean
from typing import NamedTuple

from wind_rl.experiment.verdict import windowed_delta
from wind_rl.rl.trainer import MappoTrainer, TrainingConfig

#: The deterministic-eval metric every variant is scored on (total farm power).
DEFAULT_METRIC = "eval/episode_reward_mean"


@dataclass(frozen=True)
class Variant:
    """A named point in the sweep: either partial overrides onto the base config,
    or a full ``TrainingConfig`` that replaces it wholesale."""

    name: str
    overrides: Mapping[str, object] = field(default_factory=dict)
    config: TrainingConfig | None = None


class RunResult(NamedTuple):
    variant: str
    seed: int
    first: float
    last: float
    delta: float
    auc: float
    seconds: float
    finite: bool


@dataclass(frozen=True)
class SweepResult:
    runs: list[RunResult]
    metric: str = DEFAULT_METRIC


def _experiment_name(
    base: TrainingConfig, variant: str, seed: int, seeded: bool
) -> str:
    # Per-variant so checkpoints/wandb runs never collide; the seed suffix is only
    # added when a variant spans >1 seed (single-seed runs keep the plain name).
    suffix = f"_s{seed}" if seeded else ""
    return f"{base.experiment_name}_{variant}{suffix}"


def _build_config(
    base: TrainingConfig, variant: Variant, seed: int, name: str
) -> TrainingConfig:
    if variant.config is not None:
        return variant.config.model_copy(update={"seed": seed, "experiment_name": name})
    return base.model_copy(
        update={**variant.overrides, "seed": seed, "experiment_name": name}
    )


def _set_wandb_env(group: str, tags: Sequence[str]) -> None:
    # RunLogger reads group/tags from wandb's env vars (it does not take them as
    # args); set them per run so the seeds of one variant share a wandb group.
    os.environ["WANDB_RUN_GROUP"] = group
    os.environ["WANDB_TAGS"] = ",".join(tags)


def _teardown_wandb() -> None:
    # wandb caches WANDB_RUN_GROUP / WANDB_TAGS in its process-global setup on the
    # first init, so per-run env changes are otherwise ignored; tearing the setup
    # down forces the next init to re-read them.
    try:
        import wandb

        wandb.teardown()
    except Exception:  # pragma: no cover - wandb absent or disabled
        pass


def _harvest(
    variant: str,
    seed: int,
    history: list[dict[str, float]],
    metric: str,
    seconds: float,
) -> RunResult:
    evals = [m[metric] for m in history if metric in m]
    first, last, delta = windowed_delta(evals)
    auc = fmean(evals) if evals else float("nan")
    finite = bool(evals) and all(isfinite(v) for m in history for v in m.values())
    return RunResult(variant, seed, first, last, delta, auc, seconds, finite)


def run_sweep(
    base: TrainingConfig,
    variants: Sequence[Variant],
    seeds: Sequence[int],
    metric: str = DEFAULT_METRIC,
) -> SweepResult:
    """Train every ``(variant, seed)`` and return their harvested per-run results."""
    seeded = len(seeds) > 1
    runs: list[RunResult] = []
    for variant in variants:
        for seed in seeds:
            name = _experiment_name(base, variant.name, seed, seeded)
            cfg = _build_config(base, variant, seed, name)
            _set_wandb_env(
                f"{base.experiment_name}_{variant.name}",
                [base.experiment_name, variant.name, f"seed{seed}"],
            )
            start = time.perf_counter()
            history = MappoTrainer(cfg).run()
            seconds = time.perf_counter() - start
            _teardown_wandb()
            result = _harvest(variant.name, seed, history, metric, seconds)
            print(
                f"[{result.variant} s{seed}] {result.first:.4f} -> {result.last:.4f} "
                f"(delta {result.delta:+.4f}, auc {result.auc:.4f}) "
                f"{seconds:.1f}s {'finite' if result.finite else 'NON-FINITE'}"
            )
            runs.append(result)
    return SweepResult(runs=runs, metric=metric)
