"""0004_ppo_tuning: PPO tuning-lever sweep exposed by 0001's rich telemetry.

0001 found PPO stable but conservative on the fixed 3-turbine row: pre-clip grad
norm ~8x above ``max_grad_norm=1.0`` (grad clip saturates every step, so
effective LR is ~8x below nominal), clip fraction <2%, approx-KL <0.005. This
sweep tests whether loosening the grad clip and/or raising LR speeds convergence
without instability, plus one entropy-bonus arm -- a small hypothesis-driven
2x3 factorial (lr x max_grad_norm) + entropy, each over >=2 seeds.

Convergence speed is scored by the mean of the deterministic-eval trajectory
(eval AUC): under a fixed budget every arm saturates to the same wake-steering
optimum, so a higher mean = it climbed there sooner. Stability is read from the
final-iteration clip fraction / approx-KL / grad norm. The single best arm (max
mean AUC, gated on a trust-region KL cap and on having learned) is then
validated on a 6-turbine row.

Verdict (asserted in code): every run completes with all metrics finite. Exits
nonzero iff any run is non-finite. The report's Decision recommends the PPO
default from the AUC ranking + stability evidence (or reports no improvement).
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

from pydantic import Field

from wind_rl.config import Config
from wind_rl.experiment.cli import compose_experiment
from wind_rl.models import ModelConfig
from wind_rl.rl.mappo import PPOConfig
from wind_rl.rl.trainer import LoggingConfig, MappoTrainer, TrainingConfig
from wind_rl.scenario import ScenarioConfig

_EVAL = "eval/episode_reward_mean"
_CLIP = "loss/clip_fraction"
_KL = "loss/approx_kl"
_GRAD = "optim/grad_norm"
# Trust-region cap: an arm whose mean final approx-KL exceeds this is treated as
# too aggressive to crown, however good its AUC. 0001 sat at ~0.003; 0.05 is
# ~10x PPO's usual early-stop KL and far above the conservative baseline.
_KL_CAP = 0.05


class ArmConfig(Config):
    name: str
    max_grad_norm: float | None = None
    lr: float | None = None
    entropy_eps: float | None = None

    def ppo(self, base: PPOConfig) -> PPOConfig:
        updates: dict[str, float] = {
            key: value
            for key, value in {
                "max_grad_norm": self.max_grad_norm,
                "lr": self.lr,
                "entropy_eps": self.entropy_eps,
            }.items()
            if value is not None
        }
        return base.model_copy(update=updates)


class ValidationConfig(Config):
    n_iters: int = Field(gt=0)
    seeds: list[int] = Field(min_length=1)
    scenario: ScenarioConfig
    layout: list[list[float]] = Field(min_length=1)


class ExperimentConfig(Config):
    experiment_name: str
    seeds: list[int] = Field(min_length=2)
    device: str = "cpu"
    n_iters: int = Field(gt=0)
    frames_per_batch: int = Field(gt=0)
    eval_episodes: int = Field(gt=0)
    scenario: ScenarioConfig
    layout: list[list[float]] = Field(min_length=1)
    model: ModelConfig
    ppo: PPOConfig
    logging: LoggingConfig = LoggingConfig()
    validation: ValidationConfig
    arms: list[ArmConfig] = Field(min_length=1)


class RunResult(NamedTuple):
    arm: str
    seed: int
    first: float
    last: float
    delta: float
    auc: float
    clip_fraction: float
    approx_kl: float
    grad_norm: float
    seconds: float
    finite: bool


class ArmResult(NamedTuple):
    name: str
    delta_mean: float
    delta_std: float
    last_mean: float
    auc_mean: float
    clip_fraction: float
    approx_kl: float
    grad_norm: float
    finite: bool

    def stable(self) -> bool:
        return self.finite and self.approx_kl < _KL_CAP

    def learned(self) -> bool:
        return self.delta_mean > 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _final(history: list[dict[str, float]], key: str) -> float:
    for metrics in reversed(history):
        if key in metrics:
            return metrics[key]
    return math.nan


def _reset_wandb_setup() -> None:
    # wandb caches WANDB_RUN_GROUP / WANDB_TAGS in its process-global setup on the
    # first init, so per-run env changes are otherwise ignored; tearing the setup
    # down forces the next init to re-read them.
    try:
        import wandb

        wandb.teardown()
    except Exception:  # pragma: no cover - wandb absent or disabled
        pass


def _run(
    tc: TrainingConfig, group: str, tags: list[str]
) -> tuple[list[dict[str, float]], float]:
    # RunLogger reads group/tags from wandb's env vars (it does not take them as
    # args); set them per run so seeds of an arm share a wandb group.
    os.environ["WANDB_RUN_GROUP"] = group
    os.environ["WANDB_TAGS"] = ",".join(tags)
    start = time.perf_counter()
    history = MappoTrainer(tc).run()
    _reset_wandb_setup()
    return history, time.perf_counter() - start


def _result(
    arm: str, seed: int, history: list[dict[str, float]], seconds: float
) -> RunResult:
    evals = [m[_EVAL] for m in history if _EVAL in m]
    window = max(1, len(evals) // 3)
    first = _mean(evals[:window])
    last = _mean(evals[-window:])
    auc = _mean(evals)
    clip = _final(history, _CLIP)
    kl = _final(history, _KL)
    grad = _final(history, _GRAD)
    finite = all(all(math.isfinite(v) for v in m.values()) for m in history) and bool(
        evals
    )
    return RunResult(
        arm, seed, first, last, last - first, auc, clip, kl, grad, seconds, finite
    )


def _training_config(
    cfg: ExperimentConfig,
    arm: ArmConfig,
    seed: int,
    n_iters: int,
    scenario: ScenarioConfig,
    layout: list[list[float]],
    suffix: str,
) -> TrainingConfig:
    return TrainingConfig(
        experiment_name=f"{cfg.experiment_name}_{arm.name}{suffix}_s{seed}",
        seed=seed,
        device=cfg.device,
        n_iters=n_iters,
        frames_per_batch=cfg.frames_per_batch,
        eval_interval=1,
        eval_episodes=cfg.eval_episodes,
        checkpoint_interval=n_iters,
        layout=layout,
        scenario=scenario,
        model=cfg.model,
        ppo=arm.ppo(cfg.ppo),
        logging=cfg.logging,
    )


def _aggregate(name: str, runs: list[RunResult]) -> ArmResult:
    return ArmResult(
        name=name,
        delta_mean=_mean([r.delta for r in runs]),
        delta_std=_std([r.delta for r in runs]),
        last_mean=_mean([r.last for r in runs]),
        auc_mean=_mean([r.auc for r in runs]),
        clip_fraction=_mean([r.clip_fraction for r in runs]),
        approx_kl=_mean([r.approx_kl for r in runs]),
        grad_norm=_mean([r.grad_norm for r in runs]),
        finite=all(r.finite for r in runs),
    )


def _print_sweep(arms: list[ArmResult]) -> None:
    header = (
        f"{'arm':<18}{'delta_mean':>11}{'delta_std':>10}{'last_mean':>10}"
        f"{'auc':>9}{'clip':>8}{'kl':>9}{'grad':>8}  status"
    )
    print("\nsweep (3-turbine)")
    print(header)
    print("-" * len(header))
    for a in arms:
        status = "finite" if a.finite else "NON-FINITE"
        if a.finite and not a.stable():
            status = "unstable"
        print(
            f"{a.name:<18}{a.delta_mean:>+11.4f}{a.delta_std:>10.4f}{a.last_mean:>10.4f}"
            f"{a.auc_mean:>9.4f}{a.clip_fraction:>8.3f}{a.approx_kl:>9.4f}"
            f"{a.grad_norm:>8.3f}  {status}"
        )


def _winner(arms: list[ArmResult], baseline: ArmResult) -> ArmResult:
    eligible = [a for a in arms if a.stable() and a.learned()]
    pool = eligible or [baseline]
    return max(pool, key=lambda a: a.auc_mean)


def main() -> int:
    cfg = compose_experiment(
        Path(__file__).parent / "conf", ExperimentConfig, sys.argv[1:]
    )

    arm_results: list[ArmResult] = []
    all_runs: list[RunResult] = []
    for arm in cfg.arms:
        runs: list[RunResult] = []
        for seed in cfg.seeds:
            tc = _training_config(
                cfg, arm, seed, cfg.n_iters, cfg.scenario, cfg.layout, ""
            )
            group = f"{cfg.experiment_name}_{arm.name}"
            history, seconds = _run(
                tc, group, [cfg.experiment_name, arm.name, f"seed{seed}"]
            )
            r = _result(arm.name, seed, history, seconds)
            print(
                f"[{arm.name} s{seed}] {r.first:.4f} -> {r.last:.4f} "
                f"(delta {r.delta:+.4f}, auc {r.auc:.4f}, kl {r.approx_kl:.4f}, "
                f"grad {r.grad_norm:.2f}) {seconds:.1f}s "
                f"{'finite' if r.finite else 'NON-FINITE'}"
            )
            runs.append(r)
            all_runs.append(r)
        arm_results.append(_aggregate(arm.name, runs))

    _print_sweep(arm_results)

    baseline = next(a for a in arm_results if a.name == cfg.arms[0].name)
    winner = _winner(arm_results, baseline)
    print(
        f"\nwinner: {winner.name} (auc {winner.auc_mean:.4f} vs baseline "
        f"{baseline.auc_mean:.4f}, delta {winner.delta_mean:+.4f}+-{winner.delta_std:.4f})"
    )

    winner_arm = next(a for a in cfg.arms if a.name == winner.name)
    print(f"\nvalidation (6-turbine) on winner {winner.name}")
    val_runs: list[RunResult] = []
    for seed in cfg.validation.seeds:
        tc = _training_config(
            cfg,
            winner_arm,
            seed,
            cfg.validation.n_iters,
            cfg.validation.scenario,
            cfg.validation.layout,
            "_val6",
        )
        group = f"{cfg.experiment_name}_val6_{winner.name}"
        history, seconds = _run(
            tc, group, [cfg.experiment_name, "val6", winner.name, f"seed{seed}"]
        )
        r = _result(winner.name, seed, history, seconds)
        print(
            f"[val6 {winner.name} s{seed}] {r.first:.4f} -> {r.last:.4f} "
            f"(delta {r.delta:+.4f}, kl {r.approx_kl:.4f}) {seconds:.1f}s "
            f"{'finite' if r.finite else 'NON-FINITE'}"
        )
        val_runs.append(r)
    val = _aggregate(f"val6_{winner.name}", val_runs)
    print(
        f"val6 {winner.name}: delta {val.delta_mean:+.4f}+-{val.delta_std:.4f}, "
        f"last {val.last_mean:.4f}, kl {val.approx_kl:.4f}"
    )

    non_finite = [r for r in (*all_runs, *val_runs) if not r.finite]
    if non_finite:
        offenders = ", ".join(f"{r.arm} s{r.seed}" for r in non_finite)
        print(f"\nSWEEP FAIL: non-finite runs [{offenders}]")
        return 1
    print("\nSWEEP PASS: every run completed with finite metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
