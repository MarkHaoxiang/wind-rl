"""0001_fixed_layout_marl: fixed-layout MARL benchmark.

Trains a set of MARL agent architectures (``mlp``, ``gcn``, ...) under an
identical PPO budget on the SAME fixed wind-farm layout, then reports a
per-variant learning verdict and a cross-variant comparison table.

Verdict (asserted in code, PER variant): training completes and the mean of the
last third of the variant's deterministic evals -- total farm power under wake
steering -- strictly exceeds the mean of its first third (its OWN baseline).
Wind is fixed for every reset (direction 270, speed 8), so eval is
deterministic; the windowed comparison smooths training stochasticity (rollout
sampling, SGD). Each variant PASSes or FAILs independently. The benchmark does
NOT crown a winner at smoke scale -- that comparison needs a larger budget.
Exits nonzero iff any variant FAILs its own baseline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import NamedTuple

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pydantic import Field

from wind_rl.config import Config
from wind_rl.models import ModelConfig
from wind_rl.rl.trainer import MappoTrainer, TrainingConfig

_METRIC = "eval/episode_reward_mean"


class VariantConfig(Config):
    name: str
    model: ModelConfig


class ExperimentConfig(Config):
    base: TrainingConfig
    variants: list[VariantConfig] = Field(min_length=1)

    def training_configs(self) -> list[tuple[str, TrainingConfig]]:
        return [
            (
                variant.name,
                self.base.model_copy(
                    update={
                        "model": variant.model,
                        "experiment_name": f"{self.base.experiment_name}_{variant.name}",
                    }
                ),
            )
            for variant in self.variants
        ]


class VariantResult(NamedTuple):
    name: str
    first: float
    last: float
    delta: float
    seconds: float
    passed: bool


def _compose(overrides: list[str]) -> DictConfig:
    conf_dir = str(Path(__file__).parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        return compose(config_name="config", overrides=overrides)


def _verdict(history: list[dict[str, float]]) -> tuple[bool, float, float]:
    evals = [m[_METRIC] for m in history if _METRIC in m]
    window = max(1, len(evals) // 3)
    first = sum(evals[:window]) / window
    last = sum(evals[-window:]) / window
    return last > first, first, last


def _run_variant(name: str, cfg: TrainingConfig) -> VariantResult:
    start = time.perf_counter()
    history = MappoTrainer(cfg).run()
    seconds = time.perf_counter() - start
    passed, first, last = _verdict(history)
    evals = [m[_METRIC] for m in history if _METRIC in m]
    trajectory = " ".join(f"{v:.2f}" for v in evals)
    print(f"[{name}] {_METRIC} trajectory: {trajectory}")
    print(
        f"[{name}] first-window {first:.4f} -> last-window {last:.4f} "
        f"(delta {last - first:+.4f}) in {seconds:.1f}s "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    return VariantResult(name, first, last, last - first, seconds, passed)


def _print_table(results: list[VariantResult]) -> None:
    header = (
        f"{'variant':<10} {'first_win':>10} {'last_win':>10} "
        f"{'delta':>9} {'wall_s':>8}  verdict"
    )
    print("\ncomparison")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<10} {r.first:>10.4f} {r.last:>10.4f} "
            f"{r.delta:>+9.4f} {r.seconds:>8.1f}  {'PASS' if r.passed else 'FAIL'}"
        )


def main() -> int:
    cfg = ExperimentConfig.from_raw(_compose(sys.argv[1:]))
    results = [_run_variant(name, tc) for name, tc in cfg.training_configs()]
    _print_table(results)
    all_pass = all(r.passed for r in results)
    print("\nBENCHMARK PASS" if all_pass else "\nBENCHMARK FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
