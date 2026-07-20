"""Windowed learning delta and the parameterised pass/fail gates a sweep asserts.

A run's deterministic-eval trajectory is scored by comparing the mean of its
first third against the mean of its last third (``windowed_delta``): the window
smooths rollout/SGD stochasticity, and because eval is deterministic under fixed
wind the comparison is a clean per-run learning signal. Gates turn a
:class:`~wind_rl.experiment.sweep.RunResult` into a boolean verdict; thresholds
are caller-supplied, never baked in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import nan
from statistics import fmean
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from wind_rl.experiment.sweep import RunResult

#: A verdict predicate over a single run.
Gate = Callable[["RunResult"], bool]


class WindowedDelta(NamedTuple):
    first: float
    last: float
    delta: float


def windowed_delta(values: Sequence[float]) -> WindowedDelta:
    """Mean of the last third minus mean of the first third (window ``>= 1``)."""
    if not values:
        return WindowedDelta(nan, nan, nan)
    window = max(1, len(values) // 3)
    first = fmean(values[:window])
    last = fmean(values[-window:])
    return WindowedDelta(first, last, last - first)


def improves(margin: float = 0.0) -> Gate:
    """Gate: the run's windowed delta strictly exceeds ``margin`` (own baseline)."""
    return lambda run: run.delta > margin


def improves_ratio(factor: float) -> Gate:
    """Gate: the last-window mean is at least ``factor`` times the first-window mean."""
    return lambda run: run.last >= run.first * factor


def exceeds(metric: str, threshold: float) -> Gate:
    """Gate: the run's final harvested ``metric`` (from ``extra``) is ``>= threshold``."""
    return lambda run: run.extra.get(metric, nan) >= threshold


def all_of(*gates: Gate) -> Gate:
    """Gate: every supplied gate passes."""
    return lambda run: all(gate(run) for gate in gates)


def is_finite() -> Gate:
    """Gate: the run completed with every logged metric finite (capability)."""
    return lambda run: run.finite
