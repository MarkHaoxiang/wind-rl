import os

import numpy as np
import torch

from wind_rl.utils import (
    THREAD_ENV_VARS,
    pinned_worker_threads,
    resolve_device,
    seed_all,
)


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


def test_pinned_worker_threads_sets_and_restores_env_vars() -> None:
    for i, var in enumerate(THREAD_ENV_VARS):
        os.environ[var] = str(i + 4)
    try:
        with pinned_worker_threads(True):
            assert all(os.environ[var] == "1" for var in THREAD_ENV_VARS)
        assert [os.environ[var] for var in THREAD_ENV_VARS] == [
            str(i + 4) for i in range(len(THREAD_ENV_VARS))
        ]
    finally:
        for var in THREAD_ENV_VARS:
            os.environ.pop(var, None)


def test_pinned_worker_threads_removes_previously_unset_vars() -> None:
    for var in THREAD_ENV_VARS:
        os.environ.pop(var, None)
    with pinned_worker_threads(True):
        assert all(os.environ[var] == "1" for var in THREAD_ENV_VARS)
    assert not any(var in os.environ for var in THREAD_ENV_VARS)


def test_pinned_worker_threads_disabled_is_a_no_op() -> None:
    os.environ.pop(THREAD_ENV_VARS[0], None)
    with pinned_worker_threads(False):
        assert THREAD_ENV_VARS[0] not in os.environ
