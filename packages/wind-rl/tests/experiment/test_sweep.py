"""Unit tests for the sweep's pure reductions: windowed delta, gates, aggregation.

These are the parts a run.py trusts without re-running training, so they are
tested on hand-built results (no MappoTrainer). The end-to-end plumbing is covered
by a manual smoke run, not here.
"""

from __future__ import annotations

import math

from wind_rl.experiment.sweep import RunResult, SweepResult, _run_name, _run_tags
from wind_rl.experiment.table import format_table, summarize
from wind_rl.experiment.verdict import (
    all_of,
    exceeds,
    improves,
    improves_ratio,
    is_finite,
    windowed_delta,
)


def test_run_name_adds_seed_suffix_only_when_seeded() -> None:
    assert _run_name("mlp", 0, seeded=True, job_type=None) == "mlp-s0"
    assert _run_name("mlp", 0, seeded=False, job_type=None) == "mlp"


def test_run_name_bakes_in_job_type_when_it_differs_from_variant() -> None:
    assert (
        _run_name("mlp", 0, seeded=True, job_type="turb3_row1") == "turb3_row1-mlp-s0"
    )
    assert _run_name("mlp", 0, seeded=False, job_type="turb3_row1") == "turb3_row1-mlp"


def test_run_name_collapses_duplicate_job_type_and_variant() -> None:
    assert _run_name("mlp", 0, seeded=True, job_type="mlp") == "mlp-s0"


def test_run_tags_combines_extra_variant_and_seed() -> None:
    assert _run_tags("mlp", 1, ["ablaincourt"]) == ["ablaincourt", "mlp", "seed1"]
    assert _run_tags("mlp", 1, []) == ["mlp", "seed1"]


def _run(
    variant: str,
    seed: int,
    delta: float,
    finite: bool = True,
    first: float = 1.0,
    initial: float | None = None,
    extra: dict[str, float] | None = None,
) -> RunResult:
    # first/last chosen so last - first == delta; initial defaults to first (no
    # early convergence); auc/seconds are aggregation fodder.
    return RunResult(
        variant=variant,
        seed=seed,
        first=first,
        last=first + delta,
        delta=delta,
        initial=first if initial is None else initial,
        auc=2.0 + seed,
        seconds=10.0 + seed,
        finite=finite,
        extra=extra or {},
    )


def test_windowed_delta_first_and_last_third() -> None:
    # len 6 -> window 2: first mean([1,2])=1.5, last mean([5,6])=5.5.
    win = windowed_delta([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert win.first == 1.5
    assert win.last == 5.5
    assert win.delta == 4.0
    assert win.initial == 1.0


def test_windowed_delta_single_value_has_zero_delta() -> None:
    win = windowed_delta([7.0])
    assert (win.first, win.last, win.delta, win.initial) == (7.0, 7.0, 0.0, 7.0)


def test_windowed_delta_empty_is_nan() -> None:
    win = windowed_delta([])
    assert math.isnan(win.first) and math.isnan(win.last) and math.isnan(win.delta)
    assert math.isnan(win.initial)


def test_improves_gate_uses_margin() -> None:
    gate = improves(margin=0.5)
    assert gate(_run("a", 0, delta=0.6))
    assert not gate(_run("a", 0, delta=0.5))  # strictly greater
    assert not gate(_run("a", 0, delta=-1.0))


def test_improves_ratio_gate_uses_factor() -> None:
    gate = improves_ratio(1.05)
    assert gate(_run("a", 0, delta=0.06, first=1.0))  # last 1.06 >= 1.05
    assert not gate(_run("a", 0, delta=0.04, first=1.0))  # last 1.04 < 1.05


def test_improves_ratio_gate_uses_initial_when_lower_than_first_window() -> None:
    # First-third window is already post-convergence (only rises 1%), but the
    # true starting point (initial) is much lower -- the gate should still
    # credit the run for the learning it did relative to where it started.
    gate = improves_ratio(1.05)
    converged_early = _run("a", 0, delta=0.01, first=1.0, initial=0.5)
    assert gate(converged_early)  # last=1.01 >= min(0.5, 1.0)*1.05 == 0.525


def test_improves_ratio_gate_fails_flat_from_initial() -> None:
    gate = improves_ratio(1.05)
    flat = _run("a", 0, delta=0.0, first=1.0, initial=1.0)
    assert not gate(flat)  # last==first==initial, ratio 1.0 < 1.05


def test_exceeds_gate_thresholds_extra_metric() -> None:
    gate = exceeds("eval/power_gain", 0.10)
    assert gate(_run("a", 0, delta=1.0, extra={"eval/power_gain": 0.10}))
    assert not gate(_run("a", 0, delta=1.0, extra={"eval/power_gain": 0.09}))
    assert not gate(_run("a", 0, delta=1.0, extra={}))  # missing metric -> nan -> fail


def test_all_of_requires_every_gate() -> None:
    gate = all_of(improves_ratio(1.05), exceeds("eval/power_gain", 0.10))
    passing = _run("a", 0, delta=0.10, first=1.0, extra={"eval/power_gain": 0.20})
    steers_but_flat = _run(
        "a", 0, delta=0.0, first=1.0, extra={"eval/power_gain": 0.20}
    )
    learns_no_steer = _run(
        "a", 0, delta=0.10, first=1.0, extra={"eval/power_gain": 0.05}
    )
    assert gate(passing)
    assert not gate(steers_but_flat)
    assert not gate(learns_no_steer)


def test_is_finite_gate() -> None:
    gate = is_finite()
    assert gate(_run("a", 0, delta=1.0, finite=True))
    assert not gate(_run("a", 0, delta=1.0, finite=False))


def test_summarize_aggregates_across_seeds_in_order() -> None:
    result = SweepResult(
        runs=[
            _run("mlp", 0, delta=1.0),
            _run("mlp", 1, delta=3.0),
            _run("gcn", 0, delta=-1.0),
            _run("gcn", 1, delta=0.5, finite=False),
        ]
    )
    summaries = summarize(result, improves())

    assert [s.name for s in summaries] == ["mlp", "gcn"]  # first-seen order

    mlp = summaries[0]
    assert mlp.n_seeds == 2
    assert mlp.delta_mean == 2.0
    assert mlp.delta_std == 1.0  # population std of [1, 3]
    assert mlp.auc == 2.5  # mean([2, 3])
    assert mlp.finite
    assert mlp.passed  # both seeds improve

    gcn = summaries[1]
    assert not gcn.finite  # one seed non-finite
    assert not gcn.passed  # seed 0 delta -1 does not beat baseline


def test_summarize_finite_gate_independent_of_learning() -> None:
    result = SweepResult(runs=[_run("flat", 0, delta=-2.0, finite=True)])
    (summary,) = summarize(result, is_finite())
    assert summary.passed  # non-finite gate: flat/regressing but finite still PASSes


def test_format_table_lists_every_variant_and_verdict() -> None:
    result = SweepResult(runs=[_run("mlp", 0, delta=1.0), _run("gcn", 0, delta=-1.0)])
    table = format_table(summarize(result, improves()))
    assert "mlp" in table and "gcn" in table
    assert "PASS" in table and "FAIL" in table
