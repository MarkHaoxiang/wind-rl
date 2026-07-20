"""Concrete MAPPO training loop for the FLORIS ``turbine`` co-design env.

Deliberately de-generic'd from DiCoDe's ``MAPPOCoDesign`` ABC: there is a single
consumer (the FLORIS MLP smoke experiment), so this is one small class over
:func:`~wind_rl.env.factory.make_env` rather than an abstraction over domains.
"""

from __future__ import annotations

import contextlib
import functools
import math
import shutil
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import model_validator
from tensordict import TensorDictBase
from torch.nn.utils import clip_grad_norm_
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs import EnvBase, ParallelEnv, SerialEnv, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from wind_rl.config import Config
from wind_rl.design import (
    Designer,
    DesignerConfig,
    LayoutProducer,
    create_designer,
    create_layout_buffer,
    run_buffer_dir,
)
from wind_rl.env.factory import make_env
from wind_rl.env.render import render_farm
from wind_rl.experiment.settings import WindRlSettings
from wind_rl.models import ModelConfig, build_actor_critic
from wind_rl.models.mlp import MlpModelConfig
from wind_rl.rl.logging import (
    RunLogger,
    checkpoint_aliases,
    explained_variance,
    param_norms,
)
from wind_rl.rl.mappo import (
    PPOConfig,
    batch_normalise_reward,
    build_loss_module,
    build_optimiser,
)
from wind_rl.scenario import ScenarioConfig
from wind_rl.static import GROUP_NAME
from wind_rl.utils import pinned_worker_threads, resolve_device, seed_all
from wind_rl.viz import build_replay_html, record_episode

_REWARD_KEY = (GROUP_NAME, "reward")
_EPISODE_REWARD_KEY = (GROUP_NAME, "episode_reward")
_DONE_KEY = (GROUP_NAME, "done")
_STATE_VALUE_KEY = (GROUP_NAME, "state_value")
_SCALE_KEY = (GROUP_NAME, "scale")
_ACTION_KEY = (GROUP_NAME, "action", "yaw")
_POWER_KEY = ("power",)
_GAUSSIAN_ENTROPY_CONST = 0.5 * math.log(2.0 * math.pi * math.e)
#: Upper bound on auto-resolved parallel env workers (DiCoDe's ParallelEnv cap).
_AUTO_ENV_CAP = 20


class LoggingConfig(Config):
    project: str = "wind-rl-mappo"
    use_wandb: bool = False
    #: Log an interactive HTML replay of an eval episode (only when wandb is live).
    replay: bool = True
    #: Upload every checkpoint saved on iterations divisible by this to wandb as
    #: a versioned artifact (aliased ``iter-<N>``), so a killed run still has
    #: recent weights in wandb rather than only local disk. ``None`` disables
    #: periodic uploads (final checkpoint only, as before). Only evaluated on
    #: iterations where ``checkpoint_interval`` actually saved a checkpoint, so
    #: values not a multiple of it round up to the next saved iteration. The
    #: default of 1 uploads every checkpoint -- fine at the default
    #: ``checkpoint_interval=1``/short smoke runs, but chatty for long runs
    #: with frequent checkpointing; raise it alongside ``checkpoint_interval``.
    checkpoint_upload_interval: int | None = 1


class TrainingConfig(Config):
    experiment_name: str
    seed: int = 0
    device: str | None = None
    n_iters: int = 5
    frames_per_batch: int = 400
    #: Parallel FLORIS collection workers. ``1`` (default) keeps the original
    #: serial single-env path exactly. ``None`` auto-resolves to the largest
    #: divisor of ``frames_per_batch`` that is ``<= min(frames_per_batch //
    #: scenario.max_steps, _AUTO_ENV_CAP)``. An explicit ``>1`` must divide
    #: ``frames_per_batch``.
    n_envs: int | None = 1
    #: Debug fallback: build the batched env as an in-process ``SerialEnv``
    #: instead of a subprocess ``ParallelEnv`` (no fork; easier to trace).
    serial_envs: bool = False
    #: Pin each ``ParallelEnv`` worker's numexpr/OpenBLAS/MKL/torch thread
    #: pools to 1 (see :func:`wind_rl.utils.pinned_worker_threads`), so
    #: ``n_envs`` FLORIS worker processes don't each oversubscribe every
    #: core. Only takes effect via ``mp_start_method="spawn"``, so this also
    #: switches the parallel start method from ``fork`` to ``spawn`` --
    #: measured ~6x aggregate throughput at 16 workers vs unpinned ``fork``.
    #: No effect for ``n_envs=1`` or ``serial_envs=True``.
    pin_worker_threads: bool = True
    eval_interval: int = 1
    eval_episodes: int = 4
    checkpoint_interval: int = 1
    layout: list[list[float]] | None = None
    designer: DesignerConfig | None = None
    scenario: ScenarioConfig
    model: ModelConfig = MlpModelConfig()
    ppo: PPOConfig = PPOConfig()
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def _layout_xor_designer(self) -> TrainingConfig:
        if self.layout is not None and self.designer is not None:
            raise ValueError(
                "set at most one of `layout` (fixed) or `designer` "
                "(per-episode); a fixed layout is the fixed designer's equivalent"
            )
        return self

    @model_validator(mode="after")
    def _n_envs_divides_batch(self) -> TrainingConfig:
        if self.n_envs is not None:
            if self.n_envs < 1:
                raise ValueError("n_envs must be >= 1 (or None for auto)")
            if self.frames_per_batch % self.n_envs != 0:
                raise ValueError(
                    f"frames_per_batch ({self.frames_per_batch}) must be divisible "
                    f"by n_envs ({self.n_envs}) so batches stay exactly sized"
                )
        return self


class _EvalResult(NamedTuple):
    reward_mean: float
    reward_std: float
    power_mean: float
    image: NDArray[np.uint8] | None
    replay_html: str | None


def _layout_array(cfg: TrainingConfig) -> NDArray[np.float64] | None:
    if cfg.layout is None:
        return None
    return np.asarray(cfg.layout, dtype=np.float64)


def _worker_env(
    *,
    scenario: ScenarioConfig,
    layout: NDArray[np.float64] | None,
    layout_consumer: object | None,
    device: str,
    pin_threads: bool,
) -> TransformedEnv:
    if pin_threads:
        torch.set_num_threads(1)
    return make_env(
        "train",
        scenario,
        layout=layout,
        layout_consumer=layout_consumer,  # type: ignore[arg-type]
        device=device,
    )


def _rollout_metrics(data: TensorDictBase) -> dict[str, float]:
    done = data["next", *_DONE_KEY]
    ep_reward = data["next", *_EPISODE_REWARD_KEY]
    env_done = done[..., 0].any(dim=-1)
    farm_ep_reward = ep_reward[..., 0].mean(dim=-1)
    completed = farm_ep_reward[env_done]
    if completed.numel() == 0:
        completed = data["next", *_REWARD_KEY].mean().reshape(1)

    advantage = data["advantage"]
    value_target = data["value_target"]
    action = data[_ACTION_KEY]
    return {
        "train/episode_reward_mean": float(completed.mean()),
        "train/episode_reward_min": float(completed.min()),
        "train/episode_reward_max": float(completed.max()),
        "train/episode_reward_std": float(completed.std(unbiased=False)),
        "train/step_reward_mean": float(data["next", *_REWARD_KEY].mean()),
        "train/episodes": float(int(env_done.sum())),
        "train/advantage_mean": float(advantage.mean()),
        "train/advantage_std": float(advantage.std(unbiased=False)),
        "train/value_target_mean": float(value_target.mean()),
        "train/value_target_std": float(value_target.std(unbiased=False)),
        "train/explained_variance": explained_variance(
            value_target, data[_STATE_VALUE_KEY]
        ),
        "train/action_yaw_mean": float(action.mean()),
        "train/action_yaw_std": float(action.std(unbiased=False)),
        "train/action_yaw_min": float(action.min()),
        "train/action_yaw_max": float(action.max()),
        # |dyaw| tracks the actuation budget (wfcrl zeroes yaw once cumulative
        # actuation exceeds ~10% of the horizon), so it must stay observable.
        "train/action_yaw_abs_mean": float(action.abs().mean()),
        "train/step_power_mw": float(data["next", *_POWER_KEY].mean()),
        "train/policy_entropy": float(
            (_GAUSSIAN_ENTROPY_CONST + data[_SCALE_KEY].log()).mean()
        ),
    }


class MappoTrainer:
    """End-to-end MAPPO trainer; :meth:`run` returns per-iteration metrics."""

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.settings = WindRlSettings()
        self.device = resolve_device(cfg.device)
        self.checkpoint_dir = self.settings.resolved_wdir / cfg.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.designer: Designer | None = (
            None
            if cfg.designer is None
            else create_designer(cfg.designer, cfg.scenario)
        )

    def _make_env(self, mode: str) -> TransformedEnv:
        # The designer drives the train env per episode; eval stays on a fixed
        # layout so runs are comparable (evaluating on designed-per-episode
        # layouts is a later decision).
        reset_policy = (
            self.designer.to_td_module()
            if self.designer is not None and mode == "train"
            else None
        )
        return make_env(
            mode,  # type: ignore[arg-type]
            self.cfg.scenario,
            layout=_layout_array(self.cfg),
            reset_policy=reset_policy,
            device=str(self.device),
        )

    def _resolve_n_envs(self) -> int:
        cfg = self.cfg
        if cfg.n_envs is not None:
            return cfg.n_envs
        target = min(cfg.frames_per_batch // cfg.scenario.max_steps, _AUTO_ENV_CAP)
        # Largest divisor of frames_per_batch <= target keeps batches exact.
        return next(
            d for d in range(max(1, target), 0, -1) if cfg.frames_per_batch % d == 0
        )

    def _build_train_env(self, n_envs: int, consumer: object | None) -> EnvBase:
        if n_envs == 1:
            return self._make_env("train")
        # reset_policy (a live TensorDictModule) cannot cross a process boundary,
        # so the designer path in parallel mode flows through ``consumer``.
        worker = functools.partial(
            _worker_env,
            scenario=self.cfg.scenario,
            layout=_layout_array(self.cfg),
            layout_consumer=consumer,
            device=str(self.device),
            pin_threads=self.cfg.pin_worker_threads,
        )
        env: EnvBase
        if self.cfg.serial_envs:
            env = SerialEnv(n_envs, worker, device=self.device)
        else:
            # spawn (not fork) is required for pin_worker_threads to take
            # effect: numexpr/OpenBLAS/MKL read their thread-count env var
            # once, at first import in the *process* -- a forked worker
            # inherits the parent's already-imported, already-configured
            # copies verbatim, so env vars set around a fork are a no-op.
            mp_start_method = "spawn" if self.cfg.pin_worker_threads else "fork"
            penv = ParallelEnv(
                n_envs, worker, mp_start_method=mp_start_method, device=self.device
            )
            # Spawn workers now, inside the pin: worker startup is otherwise
            # lazy (deferred to first reset/spec access), which could land
            # after this method returns and the env vars are restored.
            with pinned_worker_threads(self.cfg.pin_worker_threads):
                penv.start()
            env = penv
        # Distinct per-worker seeds derived from the run seed (worker i -> seed+i).
        env.set_seed(self.cfg.seed)
        return env

    def _layouts_per_iter(self, n_envs: int) -> int:
        # One layout per reset: ~frames_per_batch/max_steps episodes per iter,
        # plus an ``n_envs`` margin for reset-boundary jitter. A momentary
        # shortfall degrades gracefully (worker keeps its current layout).
        cfg = self.cfg
        return cfg.frames_per_batch // cfg.scenario.max_steps + n_envs

    def _eval(self, policy: torch.nn.Module, record_replay: bool) -> _EvalResult:
        env = self._make_env("eval")
        max_steps = self.cfg.scenario.max_steps
        rewards: list[float] = []
        powers: list[float] = []
        replay_html: str | None = None
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for episode in range(self.cfg.eval_episodes):
                is_last = episode == self.cfg.eval_episodes - 1
                rollout = env.rollout(max_steps, policy, break_when_any_done=True)
                rewards.append(float(rollout["next", *_EPISODE_REWARD_KEY][-1].mean()))
                powers.append(float(rollout["next", *_POWER_KEY].mean()))
                # Instrument the final eval episode for the replay instead of
                # spending a whole extra FLORIS-heavy episode on it.
                if record_replay and is_last:
                    replay_html = _replay_html(env, policy)
        # Render the live wake-resolved flow field before the env is torn down.
        render = _render_eval(env)
        env.close()
        return _EvalResult(
            float(np.mean(rewards)),
            float(np.std(rewards)),
            float(np.mean(powers)),
            render,
            replay_html,
        )

    def _greedy_baseline_power(self) -> float:
        """Mean farm power (MW) over one episode of zero-yaw (greedy) control.

        Computed once at startup so the eval power gain is readable as a
        percentage over the do-nothing baseline the paper compares against.
        """
        env = self._make_env("eval")
        max_steps = self.cfg.scenario.max_steps
        action_key = env.action_key
        zero_action = env.full_action_spec[action_key].zero()
        powers: list[float] = []
        td = env.reset()
        for _ in range(max_steps):
            td.set(action_key, zero_action)
            td = env.step(td)
            powers.append(float(td["next", *_POWER_KEY].mean()))
            if bool(td["next", *_DONE_KEY].any()):
                break
            td = step_mdp(td)
        env.close()
        return float(np.mean(powers))

    def _save_checkpoint(
        self, policy: torch.nn.Module, critic: torch.nn.Module, tag: str
    ) -> Path:
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "policy": policy.state_dict(),
                "critic": critic.state_dict(),
                "config": self.cfg.model_dump(),
            },
            path,
        )
        return path

    def run(self) -> list[dict[str, float]]:
        cfg = self.cfg
        seed_all(cfg.seed)

        ref_env = self._make_env("train")
        policy, critic = build_actor_critic(
            ref_env, cfg.scenario, cfg.model, self.device
        )
        loss_module = build_loss_module(
            policy, critic, cfg.ppo, ref_env.action_key, ref_env.reward_key
        )
        optimiser, scheduler = build_optimiser(loss_module, cfg.ppo, cfg.n_iters)

        n_envs = self._resolve_n_envs()
        designer = self.designer
        # Single-env mode drives the designer as a live reset_policy inside the
        # env; parallel workers cannot receive that module, so their designer
        # layouts flow through the shared on-disk buffer instead.
        producer: LayoutProducer | None = None
        buffer_dir: Path | None = None
        consumer: object | None = None
        if designer is not None and n_envs > 1:
            buffer_dir = run_buffer_dir(cfg.experiment_name, self.settings)
            producer, consumer = create_layout_buffer(buffer_dir)
            producer.push(
                designer.generate_layout_batch(self._layouts_per_iter(n_envs))
            )

        collector = SyncDataCollector(
            self._build_train_env(n_envs, consumer),
            policy,
            frames_per_batch=cfg.frames_per_batch,
            total_frames=cfg.frames_per_batch * cfg.n_iters,
            exploration_type=ExplorationType.RANDOM,
            device=self.device,
        )
        minibatch_size = cfg.frames_per_batch // cfg.ppo.num_minibatches
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(cfg.frames_per_batch, device=self.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=minibatch_size,
        )

        baseline_power = self._greedy_baseline_power()

        logger = RunLogger(cfg, self.settings)
        history: list[dict[str, float]] = []
        t_prev = time.perf_counter()
        latest_checkpoint_path: Path | None = None
        latest_checkpoint_iteration: int | None = None
        latest_checkpoint_uploaded = False
        try:
            for iteration, data in enumerate(collector):
                collect_s = time.perf_counter() - t_prev
                t0 = time.perf_counter()
                # Standardise rewards over the rollout before GAE, then restore
                # the raw reward: the loss consumes the stored advantage/value
                # target, so restoring keeps the logged reward/power honest
                # without changing the update.
                raw_reward: torch.Tensor | None = None
                if cfg.ppo.reward_batch_norm:
                    raw_reward = data["next", *_REWARD_KEY].clone()
                    data["next", *_REWARD_KEY] = batch_normalise_reward(raw_reward)
                with torch.no_grad():
                    loss_module.value_estimator(
                        data,
                        params=loss_module.critic_network_params,
                        target_params=loss_module.target_critic_network_params,
                    )
                if raw_reward is not None:
                    data["next", *_REWARD_KEY] = raw_reward
                replay_buffer.empty()
                replay_buffer.extend(data.reshape(-1))

                policy.train()
                critic.train()
                grad_norms: list[float] = []
                diagnostics: dict[str, list[float]] = {}
                for _ in range(cfg.ppo.n_epochs):
                    for _ in range(cfg.ppo.num_minibatches):
                        minibatch = replay_buffer.sample()
                        loss_vals = loss_module(minibatch)
                        loss_value = (
                            loss_vals["loss_objective"] + loss_vals["loss_critic"]
                        )
                        if cfg.ppo.entropy_eps > 0:
                            loss_value = loss_value + loss_vals["loss_entropy"]
                        loss_value.backward()
                        grad_norms.append(
                            float(
                                clip_grad_norm_(
                                    loss_module.parameters(), cfg.ppo.max_grad_norm
                                )
                            )
                        )
                        optimiser.step()
                        optimiser.zero_grad()
                        _accumulate_diagnostics(diagnostics, loss_vals, loss_value)
                if scheduler is not None:
                    scheduler.step()
                collector.update_policy_weights_()

                policy.eval()
                critic.eval()
                update_s = time.perf_counter() - t0

                metrics: dict[str, float] = {
                    "iteration": float(iteration),
                    "train/total_frames": float((iteration + 1) * cfg.frames_per_batch),
                    "optim/grad_norm": float(np.mean(grad_norms)),
                    "optim/lr": float(optimiser.param_groups[0]["lr"]),
                }
                metrics.update(_rollout_metrics(data))
                metrics.update({k: float(np.mean(v)) for k, v in diagnostics.items()})
                if designer is not None:
                    designer.update(data)
                    metrics.update(
                        {f"designer/{k}": v for k, v in designer.get_logs().items()}
                    )
                    # Refill the buffer for the workers' next iteration; a live
                    # reset_policy (single-env) needs no push.
                    if producer is not None:
                        producer.push(
                            designer.generate_layout_batch(
                                self._layouts_per_iter(n_envs)
                            )
                        )

                images: dict[str, NDArray[np.uint8]] = {}
                html: dict[str, str] = {}
                eval_s = 0.0
                if iteration % cfg.eval_interval == 0:
                    t0 = time.perf_counter()
                    result = self._eval(policy, cfg.logging.replay and logger.enabled)
                    eval_s = time.perf_counter() - t0
                    metrics["eval/episode_reward_mean"] = result.reward_mean
                    metrics["eval/episode_reward_std"] = result.reward_std
                    metrics["eval/episode_power_mw"] = result.power_mean
                    metrics["eval/baseline_power_mw"] = baseline_power
                    metrics["eval/power_gain"] = (
                        result.power_mean / baseline_power - 1.0
                        if baseline_power > 0.0
                        else float("nan")
                    )
                    if result.image is not None:
                        images["eval/layout"] = result.image
                    if result.replay_html is not None:
                        html["eval/replay"] = result.replay_html
                    # Cheap (no forward pass) weight telemetry at eval cadence.
                    metrics.update(
                        {
                            f"weights/policy/{k}": v
                            for k, v in param_norms(policy).items()
                        }
                    )
                    metrics.update(
                        {
                            f"weights/critic/{k}": v
                            for k, v in param_norms(critic).items()
                        }
                    )

                is_final = iteration == cfg.n_iters - 1
                checkpoint_path: Path | None = None
                if iteration % cfg.checkpoint_interval == 0 or is_final:
                    checkpoint_path = self._save_checkpoint(
                        policy, critic, str(iteration)
                    )
                if is_final:
                    checkpoint_path = self._save_checkpoint(policy, critic, "final")
                if checkpoint_path is not None:
                    latest_checkpoint_path = checkpoint_path
                    latest_checkpoint_iteration = iteration
                    latest_checkpoint_uploaded = False

                metrics["time/collect_s"] = collect_s
                metrics["time/update_s"] = update_s
                metrics["time/eval_s"] = eval_s
                metrics["time/iter_s"] = time.perf_counter() - t_prev

                logger.log(metrics, images=images or None, html=html or None)
                aliases = checkpoint_aliases(
                    iteration, cfg.logging.checkpoint_upload_interval, is_final=is_final
                )
                if checkpoint_path is not None and aliases:
                    logger.log_artifact(
                        checkpoint_path,
                        name=f"{cfg.experiment_name}-checkpoint",
                        aliases=aliases,
                    )
                    latest_checkpoint_uploaded = True
                history.append(metrics)
                t_prev = time.perf_counter()
        finally:
            collector.shutdown()
            ref_env.close()
            if buffer_dir is not None:
                shutil.rmtree(buffer_dir, ignore_errors=True)
            if (
                logger.enabled
                and latest_checkpoint_path is not None
                and not latest_checkpoint_uploaded
            ):
                # Best-effort: a crash or kill between periodic uploads should
                # not leave wandb with metrics but zero weights.
                with contextlib.suppress(Exception):  # pragma: no cover - teardown-only
                    logger.log_artifact(
                        latest_checkpoint_path,
                        name=f"{cfg.experiment_name}-checkpoint",
                        aliases=[f"iter-{latest_checkpoint_iteration}", "interrupted"],
                    )
            logger.finish()

        return history


def _accumulate_diagnostics(
    diagnostics: dict[str, list[float]],
    loss_vals: TensorDictBase,
    total: torch.Tensor,
) -> None:
    diagnostics.setdefault("loss/total", []).append(float(total.mean()))
    keys = {
        "loss/objective": "loss_objective",
        "loss/critic": "loss_critic",
        "loss/clip_fraction": "clip_fraction",
        "loss/approx_kl": "kl_approx",
        "loss/entropy": "loss_entropy",
        "loss/explained_variance": "explained_variance",
    }
    for metric, key in keys.items():
        if key in loss_vals.keys():  # noqa: SIM118 - TensorDict keys view
            diagnostics.setdefault(metric, []).append(float(loss_vals[key].mean()))


def _render_eval(env: object) -> NDArray[np.uint8] | None:
    try:
        return render_farm(env.base_env.designable_env)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - render is best-effort telemetry
        return None


def _replay_html(env: TransformedEnv, policy: torch.nn.Module) -> str | None:
    """Best-effort interactive HTML replay of one deterministic episode."""
    try:
        return build_replay_html(record_episode(env, policy))
    except Exception:  # pragma: no cover - replay is best-effort telemetry
        return None
