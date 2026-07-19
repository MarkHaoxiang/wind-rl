from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from wind_rl.env.factory import make_env
from wind_rl.env.windfarm import GROUP_NAME
from wind_rl.models.mlp import MlpModelConfig, build_mlp_actor_critic
from wind_rl.rl.mappo import PPOConfig
from wind_rl.rl.trainer import LoggingConfig, MappoTrainer, TrainingConfig
from wind_rl.scenario import ScenarioConfig
from wind_rl.utils import seed_all

_LAYOUT = [[252.0, 1000.0], [756.0, 1000.0], [1260.0, 1000.0]]


def _config() -> TrainingConfig:
    return TrainingConfig(
        experiment_name="test_trainer",
        seed=0,
        device="cpu",
        n_iters=2,
        frames_per_batch=48,
        eval_interval=1,
        eval_episodes=1,
        checkpoint_interval=1,
        layout=_LAYOUT,
        scenario=ScenarioConfig(
            name="smoke3",
            n_turbines=3,
            max_steps=8,
            map_x_length=2000.0,
            map_y_length=2000.0,
            min_distance_between_turbines=400.0,
        ),
        model=MlpModelConfig(num_cells=16, depth=1),
        ppo=PPOConfig(n_epochs=2, num_minibatches=2),
        logging=LoggingConfig(use_wandb=False),
    )


@pytest.fixture(autouse=True)
def _wdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIND_RL_WDIR", str(tmp_path))
    monkeypatch.setenv("WIND_RL_WANDB_MODE", "disabled")


def _initial_policy_state(cfg: TrainingConfig) -> dict[str, torch.Tensor]:
    seed_all(cfg.seed)
    env = make_env("train", cfg.scenario, layout=np.asarray(cfg.layout))
    policy, _ = build_mlp_actor_critic(env, cfg.scenario, cfg.model, "cpu")
    env.close()
    return {
        k: v.clone()
        for k, v in policy.state_dict().items()
        if isinstance(v, torch.Tensor)
    }


def test_trainer_reports_loss_metrics_and_updates_weights(tmp_path: Path) -> None:
    cfg = _config()
    initial = _initial_policy_state(cfg)

    history = MappoTrainer(cfg).run()

    assert len(history) == cfg.n_iters
    assert all("train_episode_reward" in m for m in history)
    assert all("eval_episode_reward" in m for m in history)
    assert history[-1]["grad_norm"] > 0.0

    final = torch.load(
        tmp_path / cfg.experiment_name / "checkpoint_final.pt", weights_only=False
    )["policy"]
    changed = [
        k for k in initial if k in final and not torch.equal(initial[k], final[k])
    ]
    assert changed, "training left every policy weight unchanged"


def test_checkpoint_reload_matches_pretrained_outputs(tmp_path: Path) -> None:
    cfg = _config()
    MappoTrainer(cfg).run()

    checkpoint = tmp_path / cfg.experiment_name / "checkpoint_final.pt"
    assert checkpoint.exists()

    payload = torch.load(checkpoint, weights_only=False)
    assert set(payload) == {"policy", "critic", "config"}
    assert payload["config"]["model"]["kind"] == "mlp"

    # Reference: architecture from the in-memory cfg, loaded with the saved weights.
    env = make_env("train", cfg.scenario, layout=np.asarray(cfg.layout))
    reference_policy, reference_critic = build_mlp_actor_critic(
        env, cfg.scenario, cfg.model, "cpu"
    )
    reference_policy.load_state_dict(payload["policy"])
    reference_critic.load_state_dict(payload["critic"])

    # Fresh model rebuilt entirely from the serialized config, exercising the
    # round trip a real reload would take.
    reloaded_cfg = TrainingConfig.model_validate(payload["config"])
    fresh_policy, fresh_critic = build_mlp_actor_critic(
        env, reloaded_cfg.scenario, reloaded_cfg.model, "cpu"
    )
    fresh_policy.load_state_dict(payload["policy"])
    fresh_critic.load_state_dict(payload["critic"])

    obs = env.reset()
    action_key = env.action_key
    env.close()

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        reference_action = reference_policy(obs.clone())[action_key]
        fresh_action = fresh_policy(obs.clone())[action_key]
        reference_value = reference_critic(obs.clone())[GROUP_NAME, "state_value"]
        fresh_value = fresh_critic(obs.clone())[GROUP_NAME, "state_value"]

    torch.testing.assert_close(reference_action, fresh_action)
    torch.testing.assert_close(reference_value, fresh_value)
