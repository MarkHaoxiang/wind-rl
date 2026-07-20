"""0001_fixed_layout_marl: the fixed-layout MARL benchmark framework.

Trains a list of config variants under an IDENTICAL PPO budget on the same fixed
layout(s), then reports a per-variant verdict and a cross-variant comparison
table. One framework, three config entry points (pick with ``config=<name>``):

  * ``config``      -- architecture benchmark (mlp/gcn/... on the 3-turbine row).
  * ``ppo_sweep``   -- PPO tuning levers as variants (lr x max_grad_norm + entropy).
  * ``real_farms``  -- named wfcrl farms (Ormonde/HornsRev1) x architectures, with
                       the real coordinate frame translated in-map by the library.

The sweep loop, metric harvest, wandb grouping, table, and gates all live in
``wind_rl.experiment`` (sweep/table/verdict); this script only composes config and
selects the verdict gate. ``gate=improves`` asserts each variant beats its own
first-window baseline (learning); ``gate=finite`` asserts each run completes with
finite metrics (capability at scale). Exits nonzero iff any variant FAILs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import Field

from wind_rl.config import Config
from wind_rl.experiment.cli import compose_experiment
from wind_rl.experiment.sweep import Variant, run_sweep
from wind_rl.experiment.table import format_table, summarize
from wind_rl.experiment.verdict import Gate, improves, is_finite
from wind_rl.models import ModelConfig
from wind_rl.rl.mappo import PPOConfig
from wind_rl.rl.trainer import TrainingConfig
from wind_rl.scenario import RealFarmConfig, resolve_real_farm

_GATES: dict[str, Gate] = {"improves": improves(), "finite": is_finite()}


class VariantSpec(Config):
    name: str
    model: ModelConfig | None = None
    ppo: PPOConfig | None = None

    def to_variant(self) -> Variant:
        overrides: dict[str, object] = {}
        if self.model is not None:
            overrides["model"] = self.model
        if self.ppo is not None:
            overrides["ppo"] = self.ppo
        return Variant(name=self.name, overrides=overrides)


class ExperimentConfig(Config):
    base: TrainingConfig
    seeds: list[int] = Field(default=[0], min_length=1)
    gate: Literal["improves", "finite"] = "improves"
    #: When set, replaces base's scenario+layout with a translated real wfcrl farm.
    farm: RealFarmConfig | None = None
    variants: list[VariantSpec] = Field(min_length=1)

    def resolved_base(self) -> TrainingConfig:
        if self.farm is None:
            return self.base
        scenario, layout = resolve_real_farm(self.farm, self.base.scenario)
        return self.base.model_copy(
            update={"scenario": scenario, "layout": layout.tolist()}
        )


def _parse_args(argv: list[str]) -> tuple[str, list[str]]:
    config_name, overrides = "config", []
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
        cfg.resolved_base(), [v.to_variant() for v in cfg.variants], cfg.seeds
    )
    summaries = summarize(result, _GATES[cfg.gate])
    print(f"\ncomparison (gate={cfg.gate})")
    print(format_table(summaries))
    passed = all(s.passed for s in summaries)
    print("\nBENCHMARK PASS" if passed else "\nBENCHMARK FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
