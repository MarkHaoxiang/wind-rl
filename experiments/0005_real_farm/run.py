"""0005_real_farm: first MARL training on a REAL wind-farm layout.

Trains MARL agent variants on a named wfcrl real-farm layout (Ablaincourt 7t,
Ormonde 30t, HornsRev1 80t, ...) under the fixed-layout path, and reports a
capability + learning verdict at production farm scale. This is also the first
production use of the parallel FLORIS collector (``TrainingConfig.n_envs``).

Real farms carry their own metre-scale coordinate frames (HornsRev1 has y down
to -1947 m). Wake physics depend only on RELATIVE turbine positions, so the raw
layout is translated so its bounding-box corner sits at ``(margin, margin)`` and
the scenario map bounds are the bbox plus a margin on every side. Translation
keeps every coordinate positive and in-map, which is what the mlp position
normalisation, the renderer's axes, and the ``layout`` observation Box (low=0)
all assume. Wind is fixed (deterministic eval, large steering headroom, matches
0001's regime); varied wind is a follow-up.

Verdict (asserted in code):
  * CAPABILITY (hard gate, sets exit code): training completes for every variant
    and every logged metric is finite. This is the claim the framework exists to
    settle -- RL runs at real-farm scale.
  * LEARNING (reported, not gated): windowed deterministic-eval delta (mean of
    the last third of evals minus the first third). At 30+ turbines under a short
    budget partial or flat learning is plausible and reported honestly.
Exits nonzero iff any variant fails the capability gate (non-finite / crash).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from wind_rl.config import Config
from wind_rl.experiment.cli import compose_experiment
from wind_rl.models import ModelConfig
from wind_rl.rl.mappo import PPOConfig
from wind_rl.rl.trainer import LoggingConfig, MappoTrainer, TrainingConfig
from wind_rl.scenario import ScenarioConfig, real_farm_layout

_METRIC = "eval/episode_reward_mean"
#: Mirrors wind_rl.rl.trainer._AUTO_ENV_CAP for reporting the auto-resolved value.
_AUTO_ENV_CAP = 20


class RealFarmConfig(Config):
    name: str
    #: Metres of empty map padding added on every side of the layout bounding box.
    margin: float = Field(default=500.0, gt=0)
    max_steps: int = Field(default=20, gt=0)
    min_distance_between_turbines: float = Field(default=300.0, gt=0)
    fixed_wind_direction: float | None = 270.0
    fixed_wind_speed: float = Field(default=8.0, gt=0)


class TrainingKnobs(Config):
    experiment_name: str
    seed: int = 0
    device: str | None = "cpu"
    n_iters: int = 25
    frames_per_batch: int = 1000
    n_envs: int | None = None
    eval_interval: int = 1
    eval_episodes: int = 1
    checkpoint_interval: int = 10
    ppo: PPOConfig = PPOConfig()
    logging: LoggingConfig = LoggingConfig()


class VariantConfig(Config):
    name: str
    model: ModelConfig


class ExperimentConfig(Config):
    farm: RealFarmConfig
    training: TrainingKnobs
    variants: list[VariantConfig] = Field(min_length=1)

    def resolve_scenario(self) -> tuple[NDArray[np.float64], ScenarioConfig]:
        return _resolve_farm(self.farm)

    def training_configs(
        self, layout: NDArray[np.float64], scenario: ScenarioConfig
    ) -> list[tuple[str, TrainingConfig]]:
        shared = self.training.model_dump(exclude={"experiment_name"})
        return [
            (
                variant.name,
                TrainingConfig(
                    experiment_name=f"{self.training.experiment_name}_{variant.name}",
                    scenario=scenario,
                    layout=layout.tolist(),
                    model=variant.model,
                    **shared,
                ),
            )
            for variant in self.variants
        ]


def _resolve_farm(
    farm: RealFarmConfig,
) -> tuple[NDArray[np.float64], ScenarioConfig]:
    raw = real_farm_layout(farm.name)
    # Physics-preserving translation: min corner -> (margin, margin), so all
    # coords are positive and in-map for normalisation / rendering / obs Box.
    layout = raw - raw.min(axis=0) + farm.margin
    map_x = float(layout[:, 0].max() + farm.margin)
    map_y = float(layout[:, 1].max() + farm.margin)
    scenario = ScenarioConfig(
        name=f"real_{farm.name}",
        n_turbines=len(layout),
        max_steps=farm.max_steps,
        map_x_length=map_x,
        map_y_length=map_y,
        min_distance_between_turbines=farm.min_distance_between_turbines,
        fixed_wind_direction=farm.fixed_wind_direction,
        fixed_wind_speed=farm.fixed_wind_speed,
    )
    return layout, scenario


def auto_n_envs(frames_per_batch: int, max_steps: int) -> int:
    """The value ``n_envs=None`` resolves to (mirrors MappoTrainer._resolve_n_envs)."""
    target = min(frames_per_batch // max_steps, _AUTO_ENV_CAP)
    return next(d for d in range(max(1, target), 0, -1) if frames_per_batch % d == 0)


class VariantResult(NamedTuple):
    name: str
    first: float
    last: float
    delta: float
    seconds: float
    mean_collect_s: float
    mean_iter_s: float
    finite: bool
    learned: bool


def _all_finite(history: list[dict[str, float]]) -> bool:
    return all(math.isfinite(v) for m in history for v in m.values())


def _windowed(history: list[dict[str, float]]) -> tuple[float, float]:
    evals = [m[_METRIC] for m in history if _METRIC in m]
    window = max(1, len(evals) // 3)
    return sum(evals[:window]) / window, sum(evals[-window:]) / window


def _mean_of(history: list[dict[str, float]], key: str) -> float:
    vals = [m[key] for m in history if key in m]
    return sum(vals) / len(vals) if vals else float("nan")


def _run_variant(name: str, cfg: TrainingConfig) -> VariantResult:
    start = time.perf_counter()
    history = MappoTrainer(cfg).run()
    seconds = time.perf_counter() - start

    finite = _all_finite(history)
    first, last = _windowed(history)
    evals = [m[_METRIC] for m in history if _METRIC in m]
    trajectory = " ".join(f"{v:.2f}" for v in evals)
    mean_collect = _mean_of(history, "time/collect_s")
    mean_iter = _mean_of(history, "time/iter_s")
    print(f"[{name}] {_METRIC} trajectory: {trajectory}")
    print(
        f"[{name}] first-window {first:.4f} -> last-window {last:.4f} "
        f"(delta {last - first:+.4f}) | finite={finite} | "
        f"collect {mean_collect:.2f}s/iter, wall {mean_iter:.2f}s/iter | "
        f"{seconds:.1f}s total -> {'CAPABLE' if finite else 'FAILED'}"
    )
    return VariantResult(
        name,
        first,
        last,
        last - first,
        seconds,
        mean_collect,
        mean_iter,
        finite,
        last > first,
    )


def _print_table(results: list[VariantResult], n_envs: int) -> None:
    header = (
        f"{'variant':<16} {'first_win':>10} {'last_win':>10} {'delta':>9} "
        f"{'collect_s':>10} {'iter_s':>8} {'wall_s':>8}  capable  learned"
    )
    print(f"\ncomparison (n_envs={n_envs})")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<16} {r.first:>10.4f} {r.last:>10.4f} {r.delta:>+9.4f} "
            f"{r.mean_collect_s:>10.2f} {r.mean_iter_s:>8.2f} {r.seconds:>8.1f}  "
            f"{'yes' if r.finite else 'NO ':>7}  {'yes' if r.learned else 'flat':>7}"
        )


def main() -> int:
    cfg = compose_experiment(
        Path(__file__).parent / "conf", ExperimentConfig, sys.argv[1:]
    )
    layout, scenario = cfg.resolve_scenario()
    n_envs = (
        cfg.training.n_envs
        if cfg.training.n_envs is not None
        else auto_n_envs(cfg.training.frames_per_batch, scenario.max_steps)
    )
    print(
        f"Farm {cfg.farm.name!r}: N={scenario.n_turbines} turbines, "
        f"map {scenario.map_x_length:.0f}x{scenario.map_y_length:.0f} m, "
        f"max_steps={scenario.max_steps}, n_envs={n_envs} "
        f"(auto={cfg.training.n_envs is None})"
    )
    results = [
        _run_variant(name, tc) for name, tc in cfg.training_configs(layout, scenario)
    ]
    _print_table(results, n_envs)
    all_capable = all(r.finite for r in results)
    print("\nCAPABILITY PASS" if all_capable else "\nCAPABILITY FAIL")
    return 0 if all_capable else 1


if __name__ == "__main__":
    raise SystemExit(main())
