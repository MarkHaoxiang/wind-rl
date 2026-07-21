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
    #: First logged eval value, pre-dating any learning; see
    #: :func:`~wind_rl.experiment.verdict.windowed_delta`.
    initial: float
    auc: float
    seconds: float
    finite: bool
    #: Final logged value of each harvested ``extra_metrics`` key (for gates that
    #: threshold a metric other than the windowed score, e.g. power gain).
    extra: Mapping[str, float] = {}


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


def _run_name(variant: str, seed: int, seeded: bool, job_type: str | None) -> str:
    # wandb's run sidebar shows name (often as the only visible column) but not
    # job_type, so bake job_type in when it differs from variant -- otherwise
    # runs from different job_types (e.g. farms) are indistinguishable at a
    # glance. Collapse the two when equal to avoid "turb3_row1-turb3_row1-s0".
    stem = (
        f"{job_type}-{variant}"
        if job_type is not None and job_type != variant
        else variant
    )
    return f"{stem}-s{seed}" if seeded else stem


def _run_tags(variant: str, seed: int, extra: Sequence[str]) -> list[str]:
    return [*extra, variant, f"seed{seed}"]


def _build_config(
    base: TrainingConfig,
    variant: Variant,
    seed: int,
    seeded: bool,
    name: str,
    group: str,
    job_type: str,
    tags: Sequence[str],
) -> TrainingConfig:
    cfg = (
        variant.config.model_copy(update={"seed": seed, "experiment_name": name})
        if variant.config is not None
        else base.model_copy(
            update={**variant.overrides, "seed": seed, "experiment_name": name}
        )
    )
    labels = cfg.logging.model_copy(
        update={
            "run_name": _run_name(variant.name, seed, seeded, job_type),
            "group": group,
            "job_type": job_type,
            "tags": _run_tags(variant.name, seed, tags),
        }
    )
    return cfg.model_copy(update={"logging": labels})


def _final(history: list[dict[str, float]], metric: str) -> float:
    values = [m[metric] for m in history if metric in m]
    return values[-1] if values else float("nan")


def _harvest(
    variant: str,
    seed: int,
    history: list[dict[str, float]],
    metric: str,
    seconds: float,
    extra_metrics: Sequence[str],
) -> RunResult:
    evals = [m[metric] for m in history if metric in m]
    first, last, delta, initial = windowed_delta(evals)
    auc = fmean(evals) if evals else float("nan")
    finite = bool(evals) and all(isfinite(v) for m in history for v in m.values())
    extra = {name: _final(history, name) for name in extra_metrics}
    return RunResult(
        variant, seed, first, last, delta, initial, auc, seconds, finite, extra
    )


def run_sweep(
    base: TrainingConfig,
    variants: Sequence[Variant],
    seeds: Sequence[int],
    metric: str = DEFAULT_METRIC,
    extra_metrics: Sequence[str] = (),
    seed_suffix: bool | None = None,
    group: str | None = None,
    job_type: str | None = None,
    tags: Sequence[str] = (),
) -> SweepResult:
    """Train every ``(variant, seed)`` and return their harvested per-run results.

    ``seed_suffix`` forces the ``_s{seed}`` run-name/checkpoint suffix on or off;
    ``None`` keeps the default (suffix iff a variant spans >1 seed). Set it ``True``
    when one logical multi-seed sweep is split across processes (each running a
    single seed) so the per-seed runs share a group without colliding on names.

    ``group``/``job_type``/``tags`` set the wandb hierarchy for every run in this
    sweep: ``group`` defaults to ``base.experiment_name`` (one framework/scenario
    collapses in the UI regardless of variant), ``job_type`` defaults to each
    run's ``variant.name`` (the axis this sweep actually compares); pass an
    explicit ``job_type`` when the framework's real comparison axis lives outside
    ``variants`` (e.g. a fixed farm/scenario choice), which then also becomes the
    caller's responsibility to include in ``tags`` if it should be filterable.
    """
    seeded = len(seeds) > 1 if seed_suffix is None else seed_suffix
    resolved_group = group if group is not None else base.experiment_name
    runs: list[RunResult] = []
    for variant in variants:
        for seed in seeds:
            name = _experiment_name(base, variant.name, seed, seeded)
            resolved_job_type = job_type if job_type is not None else variant.name
            cfg = _build_config(
                base,
                variant,
                seed,
                seeded,
                name,
                resolved_group,
                resolved_job_type,
                tags,
            )
            start = time.perf_counter()
            history = MappoTrainer(cfg).run()
            seconds = time.perf_counter() - start
            result = _harvest(
                variant.name, seed, history, metric, seconds, extra_metrics
            )
            print(
                f"[{result.variant} s{seed}] {result.first:.4f} -> {result.last:.4f} "
                f"(delta {result.delta:+.4f}, auc {result.auc:.4f}) "
                f"{seconds:.1f}s {'finite' if result.finite else 'NON-FINITE'}"
            )
            runs.append(result)
    return SweepResult(runs=runs, metric=metric)
