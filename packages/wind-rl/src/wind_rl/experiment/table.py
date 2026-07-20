"""Aggregate per-run sweep results into per-variant summaries and a text table.

Seeds of a variant are reduced to a mean (with the delta's spread as a population
std) so the comparison table reads variant x {first window, last window, delta,
auc, wall-clock, verdict}. Gating is delegated to a caller-supplied
:data:`~wind_rl.experiment.verdict.Gate`: a variant PASSes iff every one of its
runs passes.
"""

from __future__ import annotations

from statistics import fmean, pstdev
from typing import NamedTuple

from wind_rl.experiment.sweep import RunResult, SweepResult
from wind_rl.experiment.verdict import Gate


class VariantSummary(NamedTuple):
    name: str
    n_seeds: int
    first: float
    last: float
    delta_mean: float
    delta_std: float
    auc: float
    seconds: float
    finite: bool
    passed: bool


def _std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def summarize(result: SweepResult, gate: Gate) -> list[VariantSummary]:
    """Reduce each variant's runs (across seeds) to one gated summary row."""
    order: list[str] = []
    groups: dict[str, list[RunResult]] = {}
    for run in result.runs:
        if run.variant not in groups:
            order.append(run.variant)
            groups[run.variant] = []
        groups[run.variant].append(run)

    summaries: list[VariantSummary] = []
    for name in order:
        runs = groups[name]
        deltas = [r.delta for r in runs]
        summaries.append(
            VariantSummary(
                name=name,
                n_seeds=len(runs),
                first=fmean(r.first for r in runs),
                last=fmean(r.last for r in runs),
                delta_mean=fmean(deltas),
                delta_std=_std(deltas),
                auc=fmean(r.auc for r in runs),
                seconds=fmean(r.seconds for r in runs),
                finite=all(r.finite for r in runs),
                passed=all(gate(r) for r in runs),
            )
        )
    return summaries


def format_table(summaries: list[VariantSummary]) -> str:
    """Render the variant comparison as a monospace table."""
    header = (
        f"{'variant':<18}{'seeds':>6}{'first_win':>11}{'last_win':>10}"
        f"{'delta':>10}{'std':>8}{'auc':>9}{'wall_s':>9}  verdict"
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        lines.append(
            f"{s.name:<18}{s.n_seeds:>6}{s.first:>11.4f}{s.last:>10.4f}"
            f"{s.delta_mean:>+10.4f}{s.delta_std:>8.4f}{s.auc:>9.4f}{s.seconds:>9.1f}"
            f"  {'PASS' if s.passed else 'FAIL'}"
        )
    return "\n".join(lines)
