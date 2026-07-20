from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from pydantic import ValidationError
from torchrl.envs import ParallelEnv
from torchrl.envs.utils import ExplorationType, check_env_specs, set_exploration_type

pytest.importorskip("wfcrl")

from wind_rl.design import RandomDesigner, RandomDesignerConfig, create_layout_buffer
from wind_rl.env.factory import make_env
from wind_rl.models import build_actor_critic
from wind_rl.models.mlp import MlpModelConfig
from wind_rl.rl.mappo import PPOConfig
from wind_rl.rl.trainer import (
    LoggingConfig,
    MappoTrainer,
    TrainingConfig,
)
from wind_rl.scenario import ScenarioConfig
from wind_rl.static import GROUP_NAME
from wind_rl.utils import seed_all

pytestmark = pytest.mark.sim

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


def _designer_config() -> TrainingConfig:
    cfg = _config()
    return cfg.model_copy(
        update={
            "experiment_name": "test_trainer_designer",
            "n_iters": 1,
            "layout": None,
            "designer": RandomDesignerConfig(seed=0),
        }
    )


@pytest.fixture(autouse=True)
def _wdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIND_RL_WDIR", str(tmp_path))
    monkeypatch.setenv("WIND_RL_WANDB_MODE", "disabled")


def _initial_policy_state(cfg: TrainingConfig) -> dict[str, torch.Tensor]:
    seed_all(cfg.seed)
    env = make_env("train", cfg.scenario, layout=np.asarray(cfg.layout))
    policy, _ = build_actor_critic(env, cfg.scenario, cfg.model, "cpu")
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
    assert all("train/episode_reward_mean" in m for m in history)
    assert all("eval/episode_reward_mean" in m for m in history)
    assert history[-1]["optim/grad_norm"] > 0.0

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
    reference_policy, reference_critic = build_actor_critic(
        env, cfg.scenario, cfg.model, "cpu"
    )
    reference_policy.load_state_dict(payload["policy"])
    reference_critic.load_state_dict(payload["critic"])

    # Fresh model rebuilt entirely from the serialized config, exercising the
    # round trip a real reload would take.
    reloaded_cfg = TrainingConfig.model_validate(payload["config"])
    fresh_policy, fresh_critic = build_actor_critic(
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


def test_layout_and_designer_are_mutually_exclusive() -> None:
    base = _config()
    with pytest.raises(ValidationError):
        TrainingConfig(
            **{
                **base.model_dump(),
                "layout": _LAYOUT,
                "designer": {"kind": "random", "seed": 0},
            }
        )


def test_random_designer_drives_train_env_layouts() -> None:
    cfg = _designer_config()
    trainer = MappoTrainer(cfg)
    assert trainer.designer is not None

    produced: list[NDArray[np.float64]] = []
    original = trainer.designer.generate_layout_batch

    def _spy(batch_size: int) -> NDArray[np.float64]:
        out = original(batch_size)
        produced.append(np.asarray(out[0]).copy())
        return out

    trainer.designer.generate_layout_batch = _spy  # type: ignore[method-assign]

    history = trainer.run()

    assert len(history) == cfg.n_iters
    assert len(produced) >= 2, "designer reset policy was not invoked per episode"
    assert any(not np.allclose(produced[0], layout) for layout in produced[1:])


def test_parallel_env_batches_distinct_designer_layouts(tmp_path: Path) -> None:
    scenario = _config().scenario
    producer, consumer = create_layout_buffer(tmp_path / "buf")
    producer.push(RandomDesigner(scenario, seed=0).generate_layout_batch(8))

    worker = functools.partial(
        make_env, "train", scenario, layout_consumer=consumer, device="cpu"
    )
    penv = ParallelEnv(2, worker, mp_start_method="fork", device="cpu")
    try:
        check_env_specs(penv)

        td = penv.reset()
        layouts = td["state", "layout"]
        assert layouts.shape == (2, scenario.n_turbines, 2)
        # Each worker popped a different buffer layout, so the farms differ.
        assert not torch.allclose(layouts[0], layouts[1])

        rollout = penv.rollout(3)
        assert rollout.shape[0] == 2
        assert rollout["next", GROUP_NAME, "reward"].shape[0] == 2
    finally:
        penv.close()


def test_parallel_collection_completes_and_frames_match() -> None:
    cfg = _config().model_copy(
        update={"experiment_name": "test_parallel", "n_iters": 1, "n_envs": 2}
    )
    trainer = MappoTrainer(cfg)

    history = trainer.run()

    assert trainer._resolve_n_envs() == 2
    assert len(history) == 1
    # A LazyTensorStorage sized to frames_per_batch would overflow on extend if
    # the two-worker collector returned the wrong frame count.
    assert history[0]["train/total_frames"] == float(cfg.frames_per_batch)
    assert history[0]["train/episodes"] > 0.0
    assert history[0]["optim/grad_norm"] > 0.0
