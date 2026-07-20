from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch import Tensor

from wind_rl.models import So2ModelConfig, build_actor_critic
from wind_rl.models.so2 import (
    _grid_size,
    _readout_dim,
    _So2Critic,
    _So2Encoder,
    _So2GaussianParams,
    _tensor_product,
)
from wind_rl.scenario import ScenarioConfig
from wind_rl.utils import seed_all

if TYPE_CHECKING:
    from wind_rl.rl.trainer import TrainingConfig

_MAX_M = 3
_EMBED_DIM = 16
_NUM_HEADS = 4


def _encoder() -> _So2Encoder:
    torch.manual_seed(0)
    return _So2Encoder(
        max_m=_MAX_M,
        embed_dim=_EMBED_DIM,
        num_layers=2,
        num_heads=_NUM_HEADS,
        ff_mult=2,
    )


def _readout() -> int:
    return _readout_dim(
        So2ModelConfig(max_m=_MAX_M, embed_dim=_EMBED_DIM, num_heads=_NUM_HEADS)
    )


def _global_frame_features(
    wind_angle: float, positions: Tensor, yaw: Tensor, speed: float
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


def test_policy_is_rotation_invariant_and_permutation_equivariant() -> None:
    torch.manual_seed(0)
    n, action_dim, speed = 6, 1, 1.3
    wind_angle, rotation = 0.9, 2.3
    positions = torch.randn(n, 2) * 2
    yaw = torch.randn(n, 1)
    perm = torch.randperm(n)

    frame = _global_frame_features(wind_angle, positions, yaw, speed)
    # Co-rotate positions AND wind by `rotation`; yaws are wind-relative, so they
    # do not change. No canonicalisation flag is involved: invariance is structural.
    rotated = _global_frame_features(
        wind_angle + rotation, _rotate(positions, rotation), yaw, speed
    )

    policy = _So2GaussianParams(_encoder(), _readout(), action_dim)
    critic = _So2Critic(_encoder(), _readout())

    with torch.no_grad():
        loc = policy(frame)[..., :action_dim]
        loc_rot = policy(rotated)[..., :action_dim]
        loc_perm = policy(frame[perm])[..., :action_dim]
        value = critic(frame)
        value_rot = critic(rotated)
        value_perm = critic(frame[perm])

    # Exact SO(2) invariance by construction (no augmentation, no canonicalisation).
    torch.testing.assert_close(loc, loc_rot, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(value, value_rot, atol=1e-5, rtol=1e-5)
    # Permutation equivariance (policy) / invariance (pooled critic), as v0/v1.
    torch.testing.assert_close(loc_perm, loc[perm])
    torch.testing.assert_close(value_perm, value)
    torch.testing.assert_close(value, value[:1].expand_as(value))


def _full_modes(s: Tensor, v: Tensor) -> Tensor:
    max_m, channels = v.shape[-2], v.shape[-1]
    modes = torch.zeros(2 * max_m + 1, channels, dtype=torch.cfloat)
    modes[max_m] = s.to(torch.cfloat)
    for m in range(1, max_m + 1):
        modes[max_m + m] = v[m - 1]
        modes[max_m - m] = v[m - 1].conj()
    return modes


def _tensor_product_direct(
    s_a: Tensor, v_a: Tensor, s_b: Tensor, v_b: Tensor
) -> tuple[Tensor, Tensor]:
    max_m, channels = v_a.shape[-2], v_a.shape[-1]
    a, b = _full_modes(s_a, v_a), _full_modes(s_b, v_b)
    out = torch.zeros(2 * max_m + 1, channels, dtype=torch.cfloat)
    for m in range(-max_m, max_m + 1):
        for m1 in range(-max_m, max_m + 1):
            k = m - m1
            if -max_m <= k <= max_m:
                out[max_m + m] = out[max_m + m] + a[max_m + m1] * b[max_m + k]
    return out[max_m].real, out[max_m + 1 :]


def test_fft_tensor_product_matches_direct_index_addition() -> None:
    torch.manual_seed(0)
    channels = 4
    s_a, s_b = torch.randn(channels), torch.randn(channels)
    v_a = torch.randn(_MAX_M, channels, dtype=torch.cfloat)
    v_b = torch.randn(_MAX_M, channels, dtype=torch.cfloat)

    fft_s, fft_v = _tensor_product(s_a, v_a, s_b, v_b, _grid_size(_MAX_M))
    direct_s, direct_v = _tensor_product_direct(s_a, v_a, s_b, v_b)

    # The FFT-grid product must reproduce the O(m^2) convolution c_m = sum a_m1
    # b_(m-m1) exactly (this is the check that catches phase sign / conjugation
    # bugs); the m=0 output is the invariant scalar channel.
    torch.testing.assert_close(fft_s, direct_s, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fft_v, direct_v, atol=1e-5, rtol=1e-5)


_LAYOUT = [[252.0, 1000.0], [756.0, 1000.0], [1260.0, 1000.0]]


def _so2_config() -> TrainingConfig:
    from wind_rl.rl.mappo import PPOConfig
    from wind_rl.rl.trainer import LoggingConfig, TrainingConfig

    return TrainingConfig(
        experiment_name="test_so2_trainer",
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
        model=So2ModelConfig(
            max_m=_MAX_M, embed_dim=_EMBED_DIM, num_heads=_NUM_HEADS, num_layers=1
        ),
        ppo=PPOConfig(n_epochs=2, num_minibatches=2),
        logging=LoggingConfig(use_wandb=False),
    )


@pytest.mark.sim
def test_so2_trainer_run_updates_an_encoder_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("wfcrl")
    from wind_rl.env.factory import make_env
    from wind_rl.rl.trainer import MappoTrainer

    monkeypatch.setenv("WIND_RL_WDIR", str(tmp_path))
    monkeypatch.setenv("WIND_RL_WANDB_MODE", "disabled")
    cfg = _so2_config()

    seed_all(cfg.seed)
    env = make_env("train", cfg.scenario, layout=np.asarray(cfg.layout))
    policy, _ = build_actor_critic(env, cfg.scenario, cfg.model, "cpu")
    env.close()
    encoder_keys = [k for k in policy.state_dict() if "encoder.blocks" in k]
    assert encoder_keys, "no SO(2) encoder block weights found in the policy"
    initial = {k: policy.state_dict()[k].clone() for k in encoder_keys}

    history = MappoTrainer(cfg).run()
    assert len(history) == cfg.n_iters

    final = torch.load(
        tmp_path / cfg.experiment_name / "checkpoint_final.pt", weights_only=False
    )["policy"]
    changed = [k for k in initial if not torch.equal(initial[k], final[k])]
    assert changed, "training left every SO(2) encoder weight unchanged"
