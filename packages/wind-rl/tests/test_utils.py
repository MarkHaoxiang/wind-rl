import numpy as np
import torch

from wind_rl.utils import resolve_device, seed_all


def test_seed_all_determinism() -> None:
    seed_all(123)
    a_np = np.random.rand(5)
    a_torch = torch.rand(5)

    seed_all(123)
    b_np = np.random.rand(5)
    b_torch = torch.rand(5)

    assert np.array_equal(a_np, b_np)
    assert torch.equal(a_torch, b_torch)


def test_resolve_device_explicit_arg_wins() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_fallback() -> None:
    device = resolve_device()
    assert device.type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        assert device.type == "cpu"
