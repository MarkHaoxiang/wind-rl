"""0003_arch_bench: an independent architecture-benchmark suite.

Ranks the policy/critic architectures in the :data:`ModelConfig` union
(``mlp`` / ``gcn`` / ``set_transformer``) via two fast proxy tasks from
``docs/research/2026-07-19-geometric-architectures.md`` S5:

1. **Critic proxy** -- supervised value regression
   (:mod:`wind_rl.experiment.arch_bench.critic`). A random-policy dataset of
   per-agent observations + empirical discounted returns on a fixed FLORIS layout
   (cached under ``WIND_RL_WDIR``) is regressed by each architecture's critic under
   an identical optimiser budget; scored by validation MSE and explained variance
   vs a predict-the-mean baseline.
2. **Policy proxy** -- fixed-budget MAPPO
   (:mod:`wind_rl.experiment.arch_bench.policy`), which delegates to
   the shared :func:`~wind_rl.experiment.sweep.run_sweep` loop: identical-budget
   PPO runs per architecture over several seeds on the same layout, scored by the
   windowed deterministic-eval reward delta and wall-clock.

Three profiles select via ``--config-name``: ``config`` (quick, fixed wind, 8t),
``decisive`` (varied wind, 8+16t, 20 iters), and ``tiebreak`` (varied wind,
gcn vs set_transformer to convergence). The sweep loop, wandb grouping, windowed
delta, and finiteness gate live in ``wind_rl.experiment`` (sweep/table/verdict);
this script composes config, joins the two proxies into one comparison table, and
asserts the verdict IN CODE: every architecture is FUNCTIONAL -- critic EV clears
the gate AND every PPO seed completes with finite metrics. Exits nonzero iff any
architecture is non-functional.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from wind_rl.config import Config
from wind_rl.design.geometry import is_feasible, sample_feasible_layout
from wind_rl.experiment.arch_bench.critic import (
    CriticProxyConfig,
    CriticResult,
    run_critic_proxy,
)
from wind_rl.experiment.arch_bench.policy import run_policy_proxy
from wind_rl.experiment.cli import compose_experiment
from wind_rl.experiment.settings import WindRlSettings
from wind_rl.experiment.sweep import RunResult, SweepResult
from wind_rl.experiment.table import VariantSummary, summarize
from wind_rl.experiment.verdict import is_finite
from wind_rl.models import ModelConfig
from wind_rl.rl.mappo import PPOConfig
from wind_rl.rl.trainer import LoggingConfig, TrainingConfig
from wind_rl.scenario import ScenarioConfig


class VariantConfig(Config):
    name: str
    model: ModelConfig


class TierConfig(Config):
    name: str
    scenario: ScenarioConfig
    variants: list[VariantConfig] = Field(min_length=1)

    def model_variants(self) -> list[tuple[str, ModelConfig]]:
        return [(v.name, v.model) for v in self.variants]


class CriticConfig(Config):
    n_rollouts: int = Field(gt=0)
    gamma: float = Field(gt=0, le=1)
    val_fraction: float = Field(gt=0, lt=1)
    n_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    lr: float = Field(gt=0)
    ev_gate: float = 0.0


class PolicyConfig(Config):
    seeds: list[int] = Field(min_length=2)
    n_iters: int = Field(gt=0)
    frames_per_batch: int = Field(gt=0)
    n_envs: int | None = 1
    eval_episodes: int = Field(gt=0)
    ppo: PPOConfig = PPOConfig()


class ExperimentConfig(Config):
    experiment_name: str
    seed: int
    layout_seed: int
    device: str = "cpu"
    critic: CriticConfig
    policy: PolicyConfig
    logging: LoggingConfig = LoggingConfig()
    tiers: list[TierConfig] = Field(min_length=1)


def _parse_args(argv: list[str]) -> tuple[str, list[str]]:
    config_name, overrides = "config", []
    it = iter(argv)
    for arg in it:
        if arg == "--config-name":
            config_name = next(it)
        elif arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
        else:
            overrides.append(arg)
    return config_name, overrides


def _base_training_config(
    cfg: ExperimentConfig,
    tier: TierConfig,
    layout: NDArray[np.float64],
) -> TrainingConfig:
    return TrainingConfig(
        experiment_name=f"{cfg.experiment_name}_{tier.name}",
        seed=cfg.seed,
        device=cfg.device,
        n_iters=cfg.policy.n_iters,
        frames_per_batch=cfg.policy.frames_per_batch,
        n_envs=cfg.policy.n_envs,
        eval_interval=1,
        eval_episodes=cfg.policy.eval_episodes,
        checkpoint_interval=cfg.policy.n_iters,
        layout=layout.tolist(),
        scenario=tier.scenario,
        ppo=cfg.policy.ppo,
        logging=cfg.logging,
    )


def _print_table(
    tier: str,
    n_iters: int,
    critic: list[CriticResult],
    policy: SweepResult,
    summaries: list[VariantSummary],
) -> None:
    summary_by_name = {s.name: s for s in summaries}
    seeds_by_name: dict[str, list[RunResult]] = {}
    for run in policy.runs:
        seeds_by_name.setdefault(run.variant, []).append(run)

    header = (
        f"{'arch':<16}{'critic_EV':>10}{'critic_MSE':>11}"
        f"{'d_mean':>9}{'d_std':>8}{'s/iter':>8}{'params':>9}  verdict"
    )
    print(f"\ncomparison [{tier}]")
    print(header)
    print("-" * len(header))
    for c in critic:
        s = summary_by_name[c.name]
        functional = c.explained_variance > 0.0 and s.passed
        print(
            f"{c.name:<16}{c.explained_variance:>+10.4f}{c.val_mse:>11.4f}"
            f"{s.delta_mean:>+9.3f}{s.delta_std:>8.3f}{s.seconds / n_iters:>8.2f}"
            f"{c.params:>9}  {'FUNCTIONAL' if functional else 'NON-FUNCTIONAL'}"
        )
    per_seed = "  ".join(
        f"{c.name}: "
        + ",".join(f"s{r.seed}{r.delta:+.3f}" for r in seeds_by_name[c.name])
        for c in critic
    )
    print(f"per-seed deltas [{tier}]  {per_seed}")


def main() -> int:
    config_name, overrides = _parse_args(sys.argv[1:])
    cfg = compose_experiment(
        Path(__file__).parent / "conf", ExperimentConfig, overrides, config_name
    )
    wdir = WindRlSettings().resolved_wdir / cfg.experiment_name

    critic_cfg = CriticProxyConfig(
        n_rollouts=cfg.critic.n_rollouts,
        gamma=cfg.critic.gamma,
        val_fraction=cfg.critic.val_fraction,
        n_steps=cfg.critic.n_steps,
        batch_size=cfg.critic.batch_size,
        lr=cfg.critic.lr,
        ev_gate=cfg.critic.ev_gate,
    )

    non_functional: list[str] = []
    for tier in cfg.tiers:
        layout = sample_feasible_layout(
            tier.scenario, np.random.default_rng(cfg.layout_seed)
        )
        if not is_feasible(layout, tier.scenario):  # pragma: no cover - defensive
            raise RuntimeError(f"sampled layout for tier {tier.name!r} is infeasible")

        variants = tier.model_variants()
        critic_results = run_critic_proxy(
            wdir, tier.scenario, layout, variants, critic_cfg, cfg.seed, cfg.device
        )

        base = _base_training_config(cfg, tier, layout)
        policy = run_policy_proxy(base, variants, cfg.policy.seeds)
        summaries = summarize(policy, is_finite())

        _print_table(tier.name, cfg.policy.n_iters, critic_results, policy, summaries)

        critic_by_name = {c.name: c for c in critic_results}
        summary_by_name = {s.name: s for s in summaries}
        non_functional += [
            f"{tier.name}/{name}"
            for name, _ in variants
            if not (
                critic_by_name[name].explained_variance > cfg.critic.ev_gate
                and summary_by_name[name].passed
            )
        ]

    if non_functional:
        print(f"\nBENCHMARK FAIL: non-functional architectures {non_functional}")
        return 1
    print("\nBENCHMARK PASS: every architecture is functional")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
