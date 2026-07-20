from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch import Tensor

from wind_rl.models import SetTransformerModelConfig, build_actor_critic
from wind_rl.models.mlp import _FEATURE_DIM
from wind_rl.models.transformer import (
    _SetTransformerCritic,
    _SetTransformerGaussianParams,
    _TransformerEncoder,
)
from wind_rl.scenario import ScenarioConfig
from wind_rl.utils import seed_all

if TYPE_CHECKING:
    from wind_rl.rl.trainer import TrainingConfig

_EMBED_DIM = 16
_NUM_HEADS = 2


def _encoder(canonicalize_wind: bool) -> _TransformerEncoder:
    return _TransformerEncoder(
        embed_dim=_EMBED_DIM,
        num_heads=_NUM_HEADS,
        num_layers=2,
        mlp_ratio=2.0,
        canonicalize_wind=canonicalize_wind,
    )


def test_policy_is_permutation_equivariant_and_critic_invariant() -> None:
    torch.manual_seed(0)
    n_turbines, action_dim = 6, 1
    features = torch.randn(n_turbines, _FEATURE_DIM)
    perm = torch.randperm(n_turbines)

    policy = _SetTransformerGaussianParams(
        _encoder(canonicalize_wind=True), _EMBED_DIM, action_dim
    )
    critic = _SetTransformerCritic(_encoder(canonicalize_wind=True), _EMBED_DIM)

    with torch.no_grad():
        loc = policy(features)[..., :action_dim]
        loc_perm = policy(features[perm])[..., :action_dim]
        value = critic(features)
        value_perm = critic(features[perm])

    # Attention has no positional encodings: yaws permute with the turbines and
    # the mean-pooled critic value is permutation invariant.
    torch.testing.assert_close(loc_perm, loc[perm])
    torch.testing.assert_close(value_perm, value)
    torch.testing.assert_close(value, value[:1].expand_as(value))


def _global_frame_features(
    wind_angle: float,
    positions: Tensor,
    yaw: Tensor,
    speed: float,
) -> Tensor:
    n = positions.shape[0]
    wind = torch.tensor(
        [speed * math.cos(wind_angle), speed * math.sin(wind_angle)]
    ).expand(n, 2)
    return torch.cat([wind, yaw, positions], dim=-1)


def _rotate(positions: Tensor, angle: float) -> Tensor:
    cos, sin = math.cos(angle), math.sin(angle)
    x, y = positions[..., 0:1], positions[..., 1:2]
    return torch.cat([cos * x - sin * y, sin * x + cos * y], dim=-1)


def test_wind_frame_canonicalisation_makes_policy_rotation_invariant() -> None:
    torch.manual_seed(0)
    n, action_dim, speed = 5, 1, 0.7
    wind_angle, rotation = 0.4, 1.1
    positions = torch.randn(n, 2)
    yaw = torch.randn(n, 1) * 0.1

    # Co-rotating the global frame: positions rotate by theta, wind shifts by
    # theta. FLORIS yaws are wind-relative misalignments, so they are unchanged.
    frame = _global_frame_features(wind_angle, positions, yaw, speed)
    rotated = _global_frame_features(
        wind_angle + rotation, _rotate(positions, rotation), yaw, speed
    )

    canon = _SetTransformerGaussianParams(
        _encoder(canonicalize_wind=True), _EMBED_DIM, action_dim
    )
    plain = _SetTransformerGaussianParams(
        _encoder(canonicalize_wind=False), _EMBED_DIM, action_dim
    )

    with torch.no_grad():
        canon_loc = canon(frame)[..., :action_dim]
        canon_loc_rot = canon(rotated)[..., :action_dim]
        plain_loc = plain(frame)[..., :action_dim]
        plain_loc_rot = plain(rotated)[..., :action_dim]

    # The invariant that justifies v1: canonicalisation makes the policy's
    # actions identical across the co-rotated frames.
    torch.testing.assert_close(canon_loc, canon_loc_rot, atol=1e-5, rtol=1e-5)
    # Teeth: without canonicalisation the same rotation changes the actions.
    assert not torch.allclose(plain_loc, plain_loc_rot, atol=1e-3)


_LAYOUT = [[252.0, 1000.0], [756.0, 1000.0], [1260.0, 1000.0]]


def _transformer_config() -> TrainingConfig:
    from wind_rl.rl.mappo import PPOConfig
    from wind_rl.rl.trainer import LoggingConfig, TrainingConfig

    return TrainingConfig(
        experiment_name="test_set_transformer_trainer",
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
        model=SetTransformerModelConfig(
            embed_dim=_EMBED_DIM, num_heads=_NUM_HEADS, num_layers=1
        ),
        ppo=PPOConfig(n_epochs=2, num_minibatches=2),
        logging=LoggingConfig(use_wandb=False),
    )


@pytest.mark.sim
def test_set_transformer_trainer_run_updates_an_encoder_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("wfcrl")
    from wind_rl.env.factory import make_env
    from wind_rl.rl.trainer import MappoTrainer

    monkeypatch.setenv("WIND_RL_WDIR", str(tmp_path))
    monkeypatch.setenv("WIND_RL_WANDB_MODE", "disabled")
    cfg = _transformer_config()

    seed_all(cfg.seed)
    env = make_env("train", cfg.scenario, layout=np.asarray(cfg.layout))
    policy, _ = build_actor_critic(env, cfg.scenario, cfg.model, "cpu")
    env.close()
    encoder_keys = [k for k in policy.state_dict() if "encoder" in k]
    assert encoder_keys, "no transformer encoder weights found in the policy"
    initial = {k: policy.state_dict()[k].clone() for k in encoder_keys}

    history = MappoTrainer(cfg).run()
    assert len(history) == cfg.n_iters

    final = torch.load(
        tmp_path / cfg.experiment_name / "checkpoint_final.pt", weights_only=False
    )["policy"]
    changed = [k for k in initial if not torch.equal(initial[k], final[k])]
    assert changed, "training left every transformer encoder weight unchanged"
