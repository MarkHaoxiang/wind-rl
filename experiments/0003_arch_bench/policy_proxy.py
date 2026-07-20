"""Fixed-budget MAPPO proxy: short runs per architecture x seed.

Each architecture is trained by the shared :class:`MappoTrainer` under an
identical tiny budget (same iterations, frames, PPO hyperparameters) on the fixed
8-turbine layout, over several seeds. The score is the deterministic-eval reward
delta between the last and first evaluation window (a learning signal, not a
converged result) plus wall-clock per iteration. A run is functional iff it
completes with all-finite metrics.
"""

from __future__ import annotations

import math
import time
from typing import NamedTuple

from wind_rl.models import ModelConfig
from wind_rl.rl.trainer import MappoTrainer, TrainingConfig

_METRIC = "eval/episode_reward_mean"
_ITER_TIME = "time/iter_s"


class SeedResult(NamedTuple):
    seed: int
    first: float
    last: float
    delta: float
    s_per_iter: float
    finite: bool


class PolicyResult(NamedTuple):
    name: str
    seed_results: list[SeedResult]

    @property
    def mean_delta(self) -> float:
        return sum(s.delta for s in self.seed_results) / len(self.seed_results)

    @property
    def s_per_iter(self) -> float:
        return sum(s.s_per_iter for s in self.seed_results) / len(self.seed_results)

    @property
    def functional(self) -> bool:
        return all(s.finite for s in self.seed_results)


def _windowed(evals: list[float]) -> tuple[float, float]:
    window = max(1, len(evals) // 3)
    first = sum(evals[:window]) / window
    last = sum(evals[-window:]) / window
    return first, last


def _all_finite(history: list[dict[str, float]]) -> bool:
    return all(math.isfinite(v) for metrics in history for v in metrics.values())


def _run_seed(
    base: TrainingConfig, name: str, model: ModelConfig, seed: int
) -> SeedResult:
    cfg = base.model_copy(
        update={
            "model": model,
            "seed": seed,
            "experiment_name": f"{base.experiment_name}_{name}_s{seed}",
        }
    )
    history = MappoTrainer(cfg).run()
    evals = [m[_METRIC] for m in history if _METRIC in m]
    first, last = _windowed(evals)
    iter_times = [m[_ITER_TIME] for m in history if _ITER_TIME in m]
    s_per_iter = sum(iter_times) / len(iter_times)
    finite = _all_finite(history) and math.isfinite(last - first)
    print(
        f"[policy:{name}:s{seed}] {first:.3f} -> {last:.3f} "
        f"(delta {last - first:+.3f})  {s_per_iter:.2f} s/iter  "
        f"-> {'OK' if finite else 'NaN'}"
    )
    return SeedResult(seed, first, last, last - first, s_per_iter, finite)


def run_policy_proxy(
    base: TrainingConfig,
    variants: list[tuple[str, ModelConfig]],
    seeds: list[int],
) -> list[PolicyResult]:
    results: list[PolicyResult] = []
    for name, model in variants:
        start = time.perf_counter()
        seed_results = [_run_seed(base, name, model, seed) for seed in seeds]
        print(
            f"[policy:{name}] {len(seeds)} seeds in {time.perf_counter() - start:.1f}s"
        )
        results.append(PolicyResult(name, seed_results))
    return results
