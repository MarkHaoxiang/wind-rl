"""0001_wfcrl_baseline: reproduce the WFCRL NeurIPS-2024 MAPPO benchmark.

Recreates the paper's Scenario I (constant wind, yaw-only control) inside our
torchrl MAPPO stack, matching the official setup (episode 150, 2048-step
updates, 32 minibatches x 10 epochs, linear-annealed lr, per-rollout reward
standardisation, clipped value loss) wherever it is config-reachable. Two config
entry points select the farm (``config=<name>``):

  * ``turb3_row1``  -- 3 aligned turbines, 4D spacing (``Turb3_Row1`` wfcrl case).
  * ``ablaincourt`` -- the 7-turbine real French farm (``Ablaincourt`` wfcrl case).

Both farms flow through ``resolve_real_farm`` (fetch the wfcrl coordinates,
translate them in-map) so the layout is never hardcoded. The sweep loop, metric
harvest, wandb grouping, table, and gates live in ``wind_rl.experiment``; this
script only composes config and asserts the benchmark verdict:

  (a) final eval episode power >= +10% over the zero-yaw greedy baseline, and
  (b) final-third mean eval score >= run's baseline eval score x 1.05, where
      baseline is min(first eval point, first-third mean) -- guards against a
      first-third window that is already post-convergence (see
      ``wind_rl.experiment.verdict.improves_ratio``).

Exits nonzero iff any (variant, all seeds) FAILs either gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field

from wind_rl.config import Config
from wind_rl.experiment.cli import compose_experiment
from wind_rl.experiment.sweep import Variant, run_sweep
from wind_rl.experiment.table import format_table, summarize
from wind_rl.experiment.verdict import all_of, exceeds, improves_ratio
from wind_rl.models import ModelConfig
from wind_rl.rl.trainer import TrainingConfig
from wind_rl.scenario import RealFarmConfig, resolve_real_farm

_SCORE_METRIC = "eval/episode_reward_mean"
_POWER_GAIN_METRIC = "eval/power_gain"
#: The benchmark verdict: learn (score rises >=5% over the run's own baseline,
#: robust to early convergence) AND steer (final eval power >=10% over the
#: zero-yaw baseline).
_GATE = all_of(improves_ratio(1.05), exceeds(_POWER_GAIN_METRIC, 0.10))


class VariantSpec(Config):
    name: str
    model: ModelConfig


class ExperimentConfig(Config):
    base: TrainingConfig
    farm: RealFarmConfig
    seeds: list[int] = Field(default=[0, 1], min_length=1)
    variants: list[VariantSpec] = Field(min_length=1)

    def resolved_base(self) -> TrainingConfig:
        scenario, layout = resolve_real_farm(self.farm, self.base.scenario)
        return self.base.model_copy(
            update={"scenario": scenario, "layout": layout.tolist()}
        )


def _parse_args(argv: list[str]) -> tuple[str, list[str]]:
    config_name, overrides = "turb3_row1", []
    for arg in argv:
        if arg.startswith("config="):
            config_name = arg.split("=", 1)[1]
        else:
            overrides.append(arg)
    return config_name, overrides


def main() -> int:
    config_name, overrides = _parse_args(sys.argv[1:])
    cfg = compose_experiment(
        Path(__file__).parent / "conf", ExperimentConfig, overrides, config_name
    )
    result = run_sweep(
        cfg.resolved_base(),
        [Variant(name=v.name, overrides={"model": v.model}) for v in cfg.variants],
        cfg.seeds,
        metric=_SCORE_METRIC,
        extra_metrics=[_POWER_GAIN_METRIC],
    )
    summaries = summarize(result, _GATE)
    print(f"\nWFCRL baseline ({config_name}) -- score windows + power gain")
    print(format_table(summaries))
    for run in result.runs:
        print(
            f"  {run.variant} s{run.seed}: "
            f"final power gain {run.extra[_POWER_GAIN_METRIC]:+.1%}"
        )
    passed = all(s.passed for s in summaries)
    print("\nBENCHMARK PASS" if passed else "\nBENCHMARK FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
