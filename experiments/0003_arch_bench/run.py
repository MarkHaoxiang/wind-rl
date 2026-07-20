"""0003_arch_bench: an independent architecture-benchmark suite.

Ranks the policy/critic architectures in the :data:`ModelConfig` union
(``mlp`` / ``gcn`` / ``set_transformer``) in minutes, NOT full training runs,
via the two fast proxy tasks from ``docs/research/2026-07-19-geometric-architectures.md``
S5:

1. **Critic proxy** -- supervised value regression. A random-policy dataset of
   per-agent observations + empirical discounted returns on a fixed 8-turbine
   FLORIS layout (cached under ``WIND_RL_WDIR``) is regressed by each
   architecture's critic under an identical optimiser budget; scored by
   validation MSE and explained variance vs a predict-the-mean baseline.
2. **Policy proxy** -- fixed-budget MAPPO. Short identical-budget PPO runs per
   architecture over several seeds on the same layout; scored by the windowed
   deterministic-eval reward delta and wall-clock per iteration.

Deliverable: one comparison table over architectures x {critic EV, policy delta,
s/iter, params}. Verdict asserted IN CODE: every architecture is FUNCTIONAL --
critic EV clears the gate AND every PPO seed completes with finite metrics. The
benchmark does NOT crown a winner at this scale (see ``report.md`` Decision).
Exits nonzero iff any architecture is non-functional.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from critic_proxy import CriticProxyConfig, CriticResult, run_critic_proxy
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray
from omegaconf import DictConfig
from policy_proxy import PolicyResult, run_policy_proxy
from pydantic import Field

from wind_rl.config import Config
from wind_rl.design.geometry import is_feasible, sample_feasible_layout
from wind_rl.experiment.settings import WindRlSettings
from wind_rl.models import ModelConfig
from wind_rl.rl.mappo import PPOConfig
from wind_rl.rl.trainer import LoggingConfig, TrainingConfig
from wind_rl.scenario import ScenarioConfig


class VariantConfig(Config):
    name: str
    model: ModelConfig


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
    eval_episodes: int = Field(gt=0)
    ppo: PPOConfig = PPOConfig()


class ExperimentConfig(Config):
    experiment_name: str
    seed: int
    layout_seed: int
    device: str = "cpu"
    scenario: ScenarioConfig
    critic: CriticConfig
    policy: PolicyConfig
    logging: LoggingConfig = LoggingConfig()
    variants: list[VariantConfig] = Field(min_length=1)

    def model_variants(self) -> list[tuple[str, ModelConfig]]:
        return [(v.name, v.model) for v in self.variants]


def _compose(overrides: list[str]) -> DictConfig:
    conf_dir = str(Path(__file__).parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        return compose(config_name="config", overrides=overrides)


def _base_training_config(
    cfg: ExperimentConfig, layout: NDArray[np.float64]
) -> TrainingConfig:
    return TrainingConfig(
        experiment_name=cfg.experiment_name,
        seed=cfg.seed,
        device=cfg.device,
        n_iters=cfg.policy.n_iters,
        frames_per_batch=cfg.policy.frames_per_batch,
        eval_interval=1,
        eval_episodes=cfg.policy.eval_episodes,
        checkpoint_interval=cfg.policy.n_iters,
        layout=layout.tolist(),
        scenario=cfg.scenario,
        ppo=cfg.policy.ppo,
        logging=cfg.logging,
    )


def _print_table(
    critic: list[CriticResult],
    policy: list[PolicyResult],
    seeds: list[int],
) -> None:
    critic_by_name = {c.name: c for c in critic}
    policy_by_name = {p.name: p for p in policy}
    names = [c.name for c in critic]

    seed_cols = "".join(f"{f'd(s{s})':>9}" for s in seeds)
    header = (
        f"{'arch':<16}{'critic_EV':>10}{'critic_MSE':>11}"
        f"{seed_cols}{'d_mean':>9}{'s/iter':>8}{'params':>9}  verdict"
    )
    print("\ncomparison")
    print(header)
    print("-" * len(header))
    for name in names:
        c = critic_by_name[name]
        p = policy_by_name[name]
        deltas = {sr.seed: sr.delta for sr in p.seed_results}
        seed_vals = "".join(f"{deltas[s]:>+9.3f}" for s in seeds)
        functional = c.explained_variance > 0.0 and p.functional
        print(
            f"{name:<16}{c.explained_variance:>+10.4f}{c.val_mse:>11.4f}"
            f"{seed_vals}{p.mean_delta:>+9.3f}{p.s_per_iter:>8.2f}"
            f"{c.params:>9}  {'FUNCTIONAL' if functional else 'NON-FUNCTIONAL'}"
        )


def main() -> int:
    cfg = ExperimentConfig.from_raw(_compose(sys.argv[1:]))
    wdir = WindRlSettings().resolved_wdir / cfg.experiment_name

    layout = sample_feasible_layout(
        cfg.scenario, np.random.default_rng(cfg.layout_seed)
    )
    if not is_feasible(layout, cfg.scenario):  # pragma: no cover - defensive
        raise RuntimeError("sampled layout is infeasible")

    variants = cfg.model_variants()
    critic_results = run_critic_proxy(
        wdir,
        cfg.scenario,
        layout,
        variants,
        CriticProxyConfig(
            n_rollouts=cfg.critic.n_rollouts,
            gamma=cfg.critic.gamma,
            val_fraction=cfg.critic.val_fraction,
            n_steps=cfg.critic.n_steps,
            batch_size=cfg.critic.batch_size,
            lr=cfg.critic.lr,
            ev_gate=cfg.critic.ev_gate,
        ),
        cfg.seed,
        cfg.device,
    )

    base = _base_training_config(cfg, layout)
    policy_results = run_policy_proxy(base, variants, cfg.policy.seeds)

    _print_table(critic_results, policy_results, cfg.policy.seeds)

    critic_by_name = {c.name: c for c in critic_results}
    policy_by_name = {p.name: p for p in policy_results}
    non_functional = [
        name
        for name, _ in variants
        if not (
            critic_by_name[name].explained_variance > cfg.critic.ev_gate
            and policy_by_name[name].functional
        )
    ]

    if non_functional:
        print(f"\nBENCHMARK FAIL: non-functional architectures {non_functional}")
        return 1
    print("\nBENCHMARK PASS: every architecture is functional")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
