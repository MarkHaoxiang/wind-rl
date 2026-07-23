"""Per-run sweep result records and wandb run-naming for the benchmark harness.

A sweep reduces each training run to a small typed :class:`RunResult` (windowed
learning delta, eval AUC, wall-clock, finiteness); aggregation into a comparison
table and pass/fail gating live in :mod:`~wind_rl.experiment.table` and
:mod:`~wind_rl.experiment.verdict`. The training loop that *produces* these
records is owner-provided (the torchrl MAPPO trainer was removed with the RL
stack; training now lives in the ``windrl-train`` package).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

#: The deterministic-eval metric every variant is scored on (total farm power).
DEFAULT_METRIC = "eval/episode_reward_mean"


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


def _run_name(variant: str, job_type: str | None) -> str:
    # wandb's run sidebar shows name (often as the only visible column) but not
    # job_type, so bake job_type in when it differs from variant -- otherwise
    # runs from different job_types (e.g. farms) are indistinguishable at a
    # glance. Collapse the two when equal to avoid "turb3_row1-turb3_row1".
    # Deliberately no seed suffix: identical names across seeds let wandb's
    # "group by name" render a seed distribution (mean/min/max bands) instead
    # of one line per seed. The seed stays queryable via the `seed{N}` tag and
    # `run.config.seed`.
    return (
        f"{job_type}-{variant}"
        if job_type is not None and job_type != variant
        else variant
    )


def _run_tags(variant: str, seed: int, extra: Sequence[str]) -> list[str]:
    return [*extra, variant, f"seed{seed}"]
