"""Windowed learning delta scored over a run's deterministic-eval trajectory.

A run's eval trajectory is scored by comparing the mean of its first third
against the mean of its last third (``windowed_delta``): the window smooths
rollout/SGD stochasticity, and because eval is deterministic under fixed wind the
comparison is a clean per-run learning signal. ``windowed_delta`` also carries
``initial``, the very first eval point, because fast-converging runs make the
first-third window itself post-convergence -- a window-vs-window ratio then
understates how much the policy actually learned.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import nan
from statistics import fmean
from typing import NamedTuple


class WindowedDelta(NamedTuple):
    first: float
    last: float
    delta: float
    initial: float


def windowed_delta(values: Sequence[float]) -> WindowedDelta:
    """Mean of the last third minus mean of the first third (window ``>= 1``);
    ``initial`` is the first logged value, pre-dating any learning."""
    if not values:
        return WindowedDelta(nan, nan, nan, nan)
    window = max(1, len(values) // 3)
    first = fmean(values[:window])
    last = fmean(values[-window:])
    return WindowedDelta(first, last, last - first, values[0])
