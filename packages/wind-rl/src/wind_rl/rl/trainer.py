"""Concrete MAPPO training loop for the FLORIS ``turbine`` co-design env.

Deliberately de-generic'd from DiCoDe's ``MAPPOCoDesign`` ABC: there is a single
consumer (the FLORIS MLP smoke experiment), so this is one small class over
:func:`~wind_rl.env.factory.make_env` rather than an abstraction over domains.
"""

from __future__ import annotations

import math
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
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from wind_rl.config import Config
from wind_rl.design import Designer, DesignerConfig, create_designer
from wind_rl.env.factory import make_env
from wind_rl.env.render import render_farm
from wind_rl.experiment.settings import WindRlSettings
from wind_rl.models import ModelConfig, build_actor_critic
from wind_rl.models.mlp import MlpModelConfig
from wind_rl.rl.logging import RunLogger, explained_variance
from wind_rl.rl.mappo import PPOConfig, build_loss_module, build_optimiser
from wind_rl.scenario import ScenarioConfig
from wind_rl.static import GROUP_NAME
from wind_rl.utils import resolve_device, seed_all
from wind_rl.viz import build_replay_html, record_episode

_REWARD_KEY = (GROUP_NAME, "reward")
_EPISODE_REWARD_KEY = (GROUP_NAME, "episode_reward")
_DONE_KEY = (GROUP_NAME, "done")
_STATE_VALUE_KEY = (GROUP_NAME, "state_value")
_SCALE_KEY = (GROUP_NAME, "scale")
_ACTION_KEY = (GROUP_NAME, "action", "yaw")
_GAUSSIAN_ENTROPY_CONST = 0.5 * math.log(2.0 * math.pi * math.e)


class LoggingConfig(Config):
    project: str = "wind-rl-mappo"
    use_wandb: bool = False
    #: Log an interactive HTML replay of an eval episode (only when wandb is live).
    replay: bool = True


class TrainingConfig(Config):
    experiment_name: str
    seed: int = 0
    device: str | None = None
    n_iters: int = 5
    frames_per_batch: int = 400
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


class _EvalResult(NamedTuple):
    reward_mean: float
    reward_std: float
    image: NDArray[np.uint8] | None
    replay_html: str | None


def _layout_array(cfg: TrainingConfig) -> NDArray[np.float64] | None:
    if cfg.layout is None:
        return None
    return np.asarray(cfg.layout, dtype=np.float64)


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

    def _eval(self, policy: torch.nn.Module, record_replay: bool) -> _EvalResult:
        env = self._make_env("eval")
        max_steps = self.cfg.scenario.max_steps
        rewards = []
        replay_html: str | None = None
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for episode in range(self.cfg.eval_episodes):
                is_last = episode == self.cfg.eval_episodes - 1
                # Instrument the final eval episode for the replay instead of
                # spending a whole extra FLORIS-heavy episode on it.
                if record_replay and is_last:
                    reward, replay_html = _record_last_eval_episode(
                        env, policy, max_steps
                    )
                    rewards.append(reward)
                else:
                    rollout = env.rollout(max_steps, policy, break_when_any_done=True)
                    rewards.append(
                        float(rollout["next", *_EPISODE_REWARD_KEY][-1].mean())
                    )
        # Render the live wake-resolved flow field before the env is torn down.
        render = _render_eval(env)
        env.close()
        return _EvalResult(
            float(np.mean(rewards)),
            float(np.std(rewards)),
            render,
            replay_html,
        )

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

        collector = SyncDataCollector(
            self._make_env("train"),
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

        logger = RunLogger(cfg, self.settings)
        history: list[dict[str, float]] = []
        t_prev = time.perf_counter()
        try:
            for iteration, data in enumerate(collector):
                collect_s = time.perf_counter() - t_prev
                t0 = time.perf_counter()
                with torch.no_grad():
                    loss_module.value_estimator(
                        data,
                        params=loss_module.critic_network_params,
                        target_params=loss_module.target_critic_network_params,
                    )
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
                if self.designer is not None:
                    self.designer.update(data)
                    metrics.update(
                        {
                            f"designer/{k}": v
                            for k, v in self.designer.get_logs().items()
                        }
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
                    if result.image is not None:
                        images["eval/layout"] = result.image
                    if result.replay_html is not None:
                        html["eval/replay"] = result.replay_html

                is_final = iteration == cfg.n_iters - 1
                checkpoint_path: Path | None = None
                if iteration % cfg.checkpoint_interval == 0 or is_final:
                    checkpoint_path = self._save_checkpoint(
                        policy, critic, str(iteration)
                    )
                if is_final:
                    checkpoint_path = self._save_checkpoint(policy, critic, "final")

                metrics["time/collect_s"] = collect_s
                metrics["time/update_s"] = update_s
                metrics["time/eval_s"] = eval_s
                metrics["time/iter_s"] = time.perf_counter() - t_prev

                logger.log(metrics, images=images or None, html=html or None)
                if is_final and checkpoint_path is not None:
                    logger.log_artifact(
                        checkpoint_path, name=f"{cfg.experiment_name}-checkpoint"
                    )
                history.append(metrics)
                t_prev = time.perf_counter()
        finally:
            collector.shutdown()
            ref_env.close()
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


def _record_last_eval_episode(
    env: TransformedEnv, policy: torch.nn.Module, max_steps: int
) -> tuple[float, str | None]:
    """Run the final eval episode as an instrumented replay: ``(reward, html)``.

    ``record_episode`` manually steps the env (reading live FLORIS per step), so
    its terminal mean episode reward is the same metric a plain ``env.rollout``
    would report -- no separate scoring episode is needed. Replay is best-effort
    telemetry, so on failure it falls back to a plain rollout for the reward.
    """
    try:
        traj = record_episode(env, policy)
        return traj.cumulative_reward[-1], build_replay_html(traj)
    except Exception:  # pragma: no cover - replay is best-effort telemetry
        rollout = env.rollout(max_steps, policy, break_when_any_done=True)
        return float(rollout["next", *_EPISODE_REWARD_KEY][-1].mean()), None
