"""Concrete MAPPO training loop for the FLORIS ``turbine`` co-design env.

Deliberately de-generic'd from DiCoDe's ``MAPPOCoDesign`` ABC: there is a single
consumer (the FLORIS MLP smoke experiment), so this is one small class over
:func:`~wind_rl.env.factory.make_env` rather than an abstraction over domains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import model_validator
from tensordict import TensorDictBase
from torch.nn.utils import clip_grad_norm_
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType, set_exploration_type

from wind_rl.config import Config
from wind_rl.design import Designer, DesignerConfig, create_designer
from wind_rl.env.factory import make_env
from wind_rl.env.windfarm import GROUP_NAME
from wind_rl.experiment.settings import WindRlSettings
from wind_rl.models.mlp import MlpModelConfig, build_mlp_actor_critic
from wind_rl.rl.mappo import PPOConfig, build_loss_module, build_optimiser
from wind_rl.scenario import ScenarioConfig
from wind_rl.utils import resolve_device, seed_all

_REWARD_KEY = (GROUP_NAME, "reward")
_EPISODE_REWARD_KEY = (GROUP_NAME, "episode_reward")
_DONE_KEY = (GROUP_NAME, "done")


class LoggingConfig(Config):
    project: str = "wind-rl-mappo"
    use_wandb: bool = False


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
    model: MlpModelConfig = MlpModelConfig()
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


def _layout_array(cfg: TrainingConfig) -> NDArray[np.float64] | None:
    if cfg.layout is None:
        return None
    return np.asarray(cfg.layout, dtype=np.float64)


def _mean_episode_reward(td: TensorDictBase) -> float:
    done = td["next", *_DONE_KEY]
    episode_reward = td["next", *_EPISODE_REWARD_KEY]
    if done.any():
        return float(episode_reward[done].mean())
    return float(td["next", *_REWARD_KEY].mean())


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

    def _make_env(self, mode: str) -> Any:
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

    def _eval_reward(self, policy: torch.nn.Module) -> float:
        env = self._make_env("eval")
        rewards = []
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for _ in range(self.cfg.eval_episodes):
                rollout = env.rollout(
                    self.cfg.scenario.max_steps, policy, break_when_any_done=True
                )
                rewards.append(float(rollout["next", *_EPISODE_REWARD_KEY][-1].mean()))
        env.close()
        return float(np.mean(rewards))

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
        policy, critic = build_mlp_actor_critic(
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

        run = _wandb_run(cfg, self.settings)
        history: list[dict[str, float]] = []
        try:
            for iteration, data in enumerate(collector):
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
                grad_norm = 0.0
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
                        grad_norm = float(
                            clip_grad_norm_(
                                loss_module.parameters(), cfg.ppo.max_grad_norm
                            )
                        )
                        optimiser.step()
                        optimiser.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                collector.update_policy_weights_()

                policy.eval()
                critic.eval()

                metrics: dict[str, float] = {
                    "iteration": float(iteration),
                    "train_episode_reward": _mean_episode_reward(data),
                    "grad_norm": grad_norm,
                    "lr": float(optimiser.param_groups[0]["lr"]),
                }
                if self.designer is not None:
                    self.designer.update(data)
                    metrics.update(self.designer.get_logs())
                if iteration % cfg.eval_interval == 0:
                    metrics["eval_episode_reward"] = self._eval_reward(policy)

                is_final = iteration == cfg.n_iters - 1
                if iteration % cfg.checkpoint_interval == 0 or is_final:
                    self._save_checkpoint(policy, critic, str(iteration))
                if is_final:
                    self._save_checkpoint(policy, critic, "final")

                if run is not None:
                    run.log(metrics)
                history.append(metrics)
        finally:
            collector.shutdown()
            ref_env.close()
            if run is not None:
                run.finish()

        return history


def _wandb_run(cfg: TrainingConfig, settings: WindRlSettings) -> Any:
    if not cfg.logging.use_wandb or settings.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=cfg.logging.project,
        name=cfg.experiment_name,
        mode=settings.wandb_mode,
        config=cfg.model_dump(),
    )
