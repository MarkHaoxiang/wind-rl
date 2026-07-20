"""0002_flowmap_prior: does a pure-FM flow-map prior, trained on the 3-turbine
procedural feasible distribution, produce few-step samples that are (a) mostly
feasible raw and (b) feasible after projection?

Verdict (asserted in code): unconditional 4-step samples projected with hard
SLSQP reach ``projected_feasibility_threshold``; raw (pre-projection) feasibility
stays above ``raw_feasibility_floor``. Prints PASS/FAIL and exits nonzero on FAIL.
The trained prior is checkpointed under ``WIND_RL_WDIR/0002_flowmap_prior/`` so
the FlowMapDesigner can reuse it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wind_rl.config import Config
from wind_rl.design.geometry import is_feasible
from wind_rl.experiment.cli import compose_experiment
from wind_rl.experiment.settings import WindRlSettings
from wind_rl.generative.constraints import project_slsqp
from wind_rl.generative.flowmap import (
    FlowMapArch,
    sample_layouts,
    save_flowmap,
    train_flowmap_prior,
)
from wind_rl.scenario import ScenarioConfig


class TrainCfg(Config):
    n_samples: int
    n_iters: int
    batch_size: int
    lr: float
    width: int
    depth: int


class SamplingCfg(Config):
    n_eval: int
    steps: int


class VerdictCfg(Config):
    projected_feasibility_threshold: float
    raw_feasibility_floor: float


class ExperimentConfig(Config):
    experiment_name: str
    seed: int
    scenario: ScenarioConfig
    train: TrainCfg
    sampling: SamplingCfg
    verdict: VerdictCfg


def _feasibility_rate(layouts: NDArray[np.float64], scenario: ScenarioConfig) -> float:
    return float(np.mean([is_feasible(layout, scenario) for layout in layouts]))


def main() -> int:
    cfg = compose_experiment(
        Path(__file__).parent / "conf", ExperimentConfig, sys.argv[1:]
    )
    scenario = cfg.scenario

    arch = FlowMapArch(
        n_turbines=scenario.n_turbines,
        map_x_length=scenario.map_x_length,
        map_y_length=scenario.map_y_length,
        width=cfg.train.width,
        depth=cfg.train.depth,
    )

    t0 = time.time()
    model, history = train_flowmap_prior(
        scenario,
        n_samples=cfg.train.n_samples,
        n_iters=cfg.train.n_iters,
        batch_size=cfg.train.batch_size,
        lr=cfg.train.lr,
        arch=arch,
        seed=cfg.seed,
    )
    train_s = time.time() - t0

    ckpt_dir = WindRlSettings().resolved_wdir / cfg.experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "prior.pt"
    save_flowmap(model, str(ckpt))

    raw = sample_layouts(model, cfg.sampling.n_eval, cfg.sampling.steps, seed=cfg.seed)
    raw_rate = _feasibility_rate(raw, scenario)
    projected = project_slsqp(raw, scenario)
    proj_rate = _feasibility_rate(projected, scenario)

    window = max(1, len(history) // 10)
    print(f"train: {train_s:.1f}s, {cfg.train.n_iters} iters")
    print(
        f"loss: first-window {np.mean(history[:window]):.4f} -> "
        f"last-window {np.mean(history[-window:]):.4f}"
    )
    print(f"raw feasibility ({cfg.sampling.steps}-step): {raw_rate:.4f}")
    print(f"projected feasibility (SLSQP): {proj_rate:.4f}")
    print(f"checkpoint: {ckpt}")

    passed = (
        proj_rate >= cfg.verdict.projected_feasibility_threshold
        and raw_rate >= cfg.verdict.raw_feasibility_floor
    )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
