from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from wind_rl.models.gnn import (
    _GcnCritic,
    _GcnEncoder,
    _GcnGaussianParams,
    _knn_adjacency,
)
from wind_rl.models.mlp import _FEATURE_DIM


def _encoder() -> _GcnEncoder:
    return _GcnEncoder(
        in_dim=_FEATURE_DIM, hidden_dim=8, num_layers=2, connectivity="knn", k=3
    )


def test_gcn_policy_is_permutation_equivariant_and_critic_invariant() -> None:
    torch.manual_seed(0)
    n_turbines, action_dim = 6, 1
    features = torch.randn(n_turbines, _FEATURE_DIM)
    perm = torch.randperm(n_turbines)

    policy = _GcnGaussianParams(_encoder(), hidden_dim=8, action_dim=action_dim)
    critic = _GcnCritic(_encoder(), hidden_dim=8)

    with torch.no_grad():
        loc = policy(features)[..., :action_dim]
        loc_perm = policy(features[perm])[..., :action_dim]
        value = critic(features)
        value_perm = critic(features[perm])

    # Policy yaws permute with the turbines; critic value is permutation invariant.
    torch.testing.assert_close(loc_perm, loc[perm])
    torch.testing.assert_close(value_perm, value)
    # The centralized value is the same pooled scalar broadcast to every node.
    torch.testing.assert_close(value, value[:1].expand_as(value))


def test_knn_adjacency_matches_hand_computed_neighbours() -> None:
    # Two well-separated pairs: 0-1 and 2-3 are each other's nearest neighbour.
    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
    adjacency = _knn_adjacency(positions, k=1)
    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    torch.testing.assert_close(adjacency, expected)


pytest.importorskip("wfcrl")

from wind_rl.env.factory import make_env  # noqa: E402
from wind_rl.models import GcnModelConfig, build_actor_critic  # noqa: E402
from wind_rl.rl.mappo import PPOConfig  # noqa: E402
from wind_rl.rl.trainer import (  # noqa: E402
    LoggingConfig,
    MappoTrainer,
    TrainingConfig,
)
from wind_rl.scenario import ScenarioConfig  # noqa: E402
from wind_rl.utils import seed_all  # noqa: E402

_LAYOUT = [[252.0, 1000.0], [756.0, 1000.0], [1260.0, 1000.0]]


def _gcn_config() -> TrainingConfig:
    return TrainingConfig(
        experiment_name="test_gcn_trainer",
        seed=0,
        device="cpu",
        n_iters=1,
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
        model=GcnModelConfig(hidden_dim=8, num_layers=2),
        ppo=PPOConfig(n_epochs=2, num_minibatches=2),
        logging=LoggingConfig(use_wandb=False),
    )


@pytest.mark.sim
def test_gcn_trainer_run_updates_a_graph_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIND_RL_WDIR", str(tmp_path))
    monkeypatch.setenv("WIND_RL_WANDB_MODE", "disabled")
    cfg = _gcn_config()

    seed_all(cfg.seed)
    env = make_env("train", cfg.scenario, layout=np.asarray(cfg.layout))
    policy, _ = build_actor_critic(env, cfg.scenario, cfg.model, "cpu")
    env.close()
    graph_keys = [k for k in policy.state_dict() if "encoder.layers" in k]
    assert graph_keys, "no GCN graph-convolution weights found in the policy"
    initial = {k: policy.state_dict()[k].clone() for k in graph_keys}

    history = MappoTrainer(cfg).run()
    assert len(history) == cfg.n_iters

    final = torch.load(
        tmp_path / cfg.experiment_name / "checkpoint_final.pt", weights_only=False
    )["policy"]
    changed = [k for k in initial if not torch.equal(initial[k], final[k])]
    assert changed, "training left every GCN graph weight unchanged"
