"""M1 MAPPO baseline framework: Mava FF-MAPPO on a fixed windrl-engine layout.

Each seed trains in a fresh ``windrl_train.train`` subprocess (Hydra/global-state
isolation per run); the eval trajectory is read back from a metrics JSON the
trainer writes (``WINDRL_TRAIN_METRICS_PATH``). The zero-yaw baseline is computed
here directly with a ``windrl_engine`` rollout.

Verdict (asserted below, per seed, every seed must clear both gates):
  * learning   -- trained best-policy return >= early-window return * learning_ratio
  * power gain -- trained best-policy return >= zero-yaw return     * power_gain_ratio

Run:  uv run python experiments/0002_mappo_baseline/run.py [config=turb3_row1] [k=v ...]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

from windrl_engine.env.config import WindFarmEnvConfig
from windrl_engine.env.env import reset as core_reset
from windrl_engine.env.env import step as core_step
from windrl_engine.farm.wind import WindCondition
from windrl_train.config import Config
from windrl_train.settings import WindRlSettings
from windrl_train.verdict import windowed_delta

FRAMEWORK = "0002_mappo_baseline"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = Path(__file__).resolve().parent / "conf"


class FixedWind(Config):
    speed: float
    direction: float


class EnvConf(Config):
    layout: str
    horizon: int
    load_coef: float
    yaw_step: float
    fixed_wind: FixedWind


class TrainConf(Config):
    num_envs: int
    rollout_length: int
    update_batch_size: int
    num_minibatches: int
    ppo_epochs: int
    ent_coef: float
    actor_lr: float
    critic_lr: float
    num_updates: int
    num_evaluation: int
    num_eval_episodes: int
    num_absolute_metric_eval_episodes: int
    evaluation_greedy: bool


class VerdictConf(Config):
    learning_ratio: float
    power_gain_ratio: float


class ExperimentConf(Config):
    variant: str
    seeds: list[int]
    env: EnvConf
    train: TrainConf
    verdict: VerdictConf


class EvalPoint(Config):
    timestep: float
    episode_return: float


class MetricsFile(Config):
    final_eval: float
    absolute_return: float | None
    eval_series: list[EvalPoint]


class SeedVerdict(NamedTuple):
    seed: int
    early: float
    trained: float
    zero: float
    learning_ratio: float
    power_ratio: float

    @property
    def learning_pass(self) -> bool:
        return self.trained >= self.early * self.learning_ratio

    @property
    def power_pass(self) -> bool:
        return self.trained >= self.zero * self.power_ratio

    @property
    def passed(self) -> bool:
        return self.learning_pass and self.power_pass


def zero_policy_return(env: EnvConf) -> float:
    """Deterministic episode return of the all-zero-action (zero-yaw) policy under
    the fixed aligned wind, computed on the same engine core the trainer wraps."""
    core = WindFarmEnvConfig(
        control_mode="continuous",
        layout=env.layout,  # type: ignore[arg-type]
        horizon=env.horizon,
        yaw_step=env.yaw_step,
        load_coef=env.load_coef,
    )
    layout = core.build_layout()
    n_turbines = int(layout.x.shape[0])
    wind = WindCondition(
        speed=jnp.asarray(env.fixed_wind.speed),
        direction=jnp.asarray(env.fixed_wind.direction),
    )
    state, _ = core_reset(layout, jax.random.PRNGKey(0), wind=wind)
    zero_action = jnp.zeros(n_turbines)
    total = 0.0
    for _ in range(env.horizon):
        state, _obs, reward, _truncated = core_step(
            layout,
            state,
            zero_action,
            yaw_step=env.yaw_step,
            load_coef=env.load_coef,
            horizon=env.horizon,
        )
        total += float(reward)
    return total


def _overrides(conf: ExperimentConf, seed: int, run_dir: Path) -> list[str]:
    e, t = conf.env, conf.train
    return [
        # Keep Hydra's run artifacts under WIND_RL_WDIR: runs never write into the repo.
        f"hydra.run.dir={run_dir}",
        "hydra.output_subdir=null",
        f"env.kwargs.layout={e.layout}",
        f"env.kwargs.horizon={e.horizon}",
        f"env.kwargs.load_coef={e.load_coef}",
        f"env.kwargs.yaw_step={e.yaw_step}",
        f"+env.fixed_wind.speed={e.fixed_wind.speed}",
        f"+env.fixed_wind.direction={e.fixed_wind.direction}",
        f"arch.num_envs={t.num_envs}",
        f"system.rollout_length={t.rollout_length}",
        f"system.update_batch_size={t.update_batch_size}",
        f"system.num_minibatches={t.num_minibatches}",
        f"system.ppo_epochs={t.ppo_epochs}",
        f"system.ent_coef={t.ent_coef}",
        f"system.actor_lr={t.actor_lr}",
        f"system.critic_lr={t.critic_lr}",
        f"system.num_updates={t.num_updates}",
        f"arch.num_evaluation={t.num_evaluation}",
        f"arch.num_eval_episodes={t.num_eval_episodes}",
        f"arch.num_absolute_metric_eval_episodes={t.num_absolute_metric_eval_episodes}",
        "arch.absolute_metric=True",
        f"arch.evaluation_greedy={t.evaluation_greedy}",
        f"system.seed={seed}",
    ]


def train_seed(conf: ExperimentConf, seed: int, metrics_path: Path) -> MetricsFile:
    child_env = dict(os.environ)
    child_env["WINDRL_TRAIN_METRICS_PATH"] = str(metrics_path)
    child_env.setdefault("JAX_PLATFORMS", "cpu")
    run_dir = metrics_path.parent / f"hydra_{metrics_path.stem}"
    # Subprocess-per-seed (not in-process) isolates Hydra + Mava global state
    # between runs; one repo venv now, so the current interpreter is the trainer.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "windrl_train.train",
            *_overrides(conf, seed, run_dir),
        ],
        check=True,
        cwd=REPO_ROOT,
        env=child_env,
    )
    return MetricsFile.model_validate_json(metrics_path.read_text())


def evaluate_seed(
    conf: ExperimentConf, metrics: MetricsFile, seed: int, zero: float
) -> SeedVerdict:
    series = [p.episode_return for p in metrics.eval_series]
    if metrics.absolute_return is None:
        raise ValueError(
            "trainer did not report an absolute-metric return (absolute_metric off?)"
        )
    return SeedVerdict(
        seed=seed,
        early=windowed_delta(series).first,
        trained=metrics.absolute_return,
        zero=zero,
        learning_ratio=conf.verdict.learning_ratio,
        power_ratio=conf.verdict.power_gain_ratio,
    )


def _log_wandb(
    conf: ExperimentConf,
    settings: WindRlSettings,
    seed: int,
    metrics: MetricsFile,
    verdict: SeedVerdict,
) -> None:
    os.environ["WANDB_MODE"] = settings.wandb_mode
    import wandb

    # Deliberately no seed in the run name: seeds share a name so wandb's group-by-name
    # renders a seed distribution rather than one line per seed (repo convention).
    run = wandb.init(
        project="wind-rl",
        name=conf.variant,
        group=FRAMEWORK,
        job_type=conf.variant,
        tags=[FRAMEWORK, conf.variant, f"seed{seed}"],
        config={"seed": seed, **conf.model_dump()},
        reinit=True,
    )
    for point in metrics.eval_series:
        run.log({"eval/episode_return": point.episode_return}, step=int(point.timestep))
    run.summary["eval/trained_return"] = verdict.trained
    run.summary["eval/zero_policy_return"] = verdict.zero
    run.summary["eval/power_gain"] = verdict.trained / verdict.zero - 1.0
    run.summary["verdict/learning_pass"] = verdict.learning_pass
    run.summary["verdict/power_pass"] = verdict.power_pass
    run.summary["verdict/passed"] = verdict.passed
    run.finish()


def _print_table(zero: float, verdicts: list[SeedVerdict]) -> None:
    print(f"\n{FRAMEWORK}: zero-yaw baseline return = {zero:.2f}\n")
    header = f"{'seed':>4} {'early':>8} {'trained':>8} {'learn x':>8} {'power x':>8} {'verdict':>8}"
    print(header)
    print("-" * len(header))
    for v in verdicts:
        status = "PASS" if v.passed else "FAIL"
        print(
            f"{v.seed:>4} {v.early:>8.2f} {v.trained:>8.2f} "
            f"{v.trained / v.early:>8.3f} {v.trained / v.zero:>8.3f} {status:>8}"
        )
    print()


def _parse_args(argv: list[str]) -> tuple[str, list[str]]:
    variant = "turb3_row1"
    overrides: list[str] = []
    for arg in argv:
        if arg.startswith("config="):
            variant = arg.split("=", 1)[1]
        else:
            overrides.append(arg)
    return variant, overrides


def main(argv: list[str]) -> int:
    variant, overrides = _parse_args(argv)
    conf = ExperimentConf.from_file(CONF_DIR / f"{variant}.yaml", overrides)

    settings = WindRlSettings()
    workdir = settings.resolved_wdir / FRAMEWORK
    workdir.mkdir(parents=True, exist_ok=True)

    zero = zero_policy_return(conf.env)

    verdicts: list[SeedVerdict] = []
    for seed in conf.seeds:
        metrics_path = workdir / f"{conf.variant}_seed{seed}_metrics.json"
        metrics = train_seed(conf, seed, metrics_path)
        verdict = evaluate_seed(conf, metrics, seed, zero)
        verdicts.append(verdict)
        if settings.wandb_mode != "disabled":
            _log_wandb(conf, settings, seed, metrics, verdict)

    _print_table(zero, verdicts)

    failed = [v.seed for v in verdicts if not v.passed]
    if failed:
        print(f"VERDICT: FAIL -- seeds {failed} did not clear both gates.")
        return 1
    print(
        f"VERDICT: PASS -- all {len(verdicts)} seeds cleared learning and power gates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
