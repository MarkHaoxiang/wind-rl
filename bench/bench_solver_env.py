"""Throughput benchmark for the windrl-engine solver and batched env (Benchmark B).

Runs one (platform, dtype) point per process because ``jax_enable_x64`` and
``JAX_PLATFORMS`` are process-global. Emits one JSON line per
(kind, layout, batch) measurement to stdout and to ``--out``.

Compile time (first traced call, blocked) is reported separately from
steady-state (median of blocked calls after warmup). Every timed region ends in
``block_until_ready`` so async dispatch never leaks into the number.
"""

from __future__ import annotations

# Imports are deliberately staged: JAX_PLATFORMS and jax_enable_x64 must be set
# from CLI flags before jax initializes, so jax imports come after arg parsing.
# ruff: noqa: E402
import argparse
import json
import statistics
import time
from collections.abc import Callable
from typing import Any, cast

# x64 must be set before jax initializes arrays; do it from the dtype flag.
_P = argparse.ArgumentParser()
_P.add_argument("--platform", choices=["cpu", "gpu"], required=True)
_P.add_argument("--dtype", choices=["f32", "f64"], required=True)
_P.add_argument("--out", required=True)
_P.add_argument("--repeats", type=int, default=30)
_P.add_argument("--batches", type=str, default="1,8,64,256,1024,4096")
_P.add_argument("--layouts", type=str, default="turb3_row1,ablaincourt,horns_rev2")
_ARGS = _P.parse_args()

import os

os.environ["JAX_PLATFORMS"] = "cuda" if _ARGS.platform == "gpu" else "cpu"

import jax

jax.config.update("jax_enable_x64", _ARGS.dtype == "f64")

import jax.numpy as jnp

from windrl_engine.env.batched import BatchedWindFarmEnv
from windrl_engine.env.config import LayoutName, WindFarmEnvConfig
from windrl_engine.farm.layout import ablaincourt, horns_rev2, turb3_row1
from windrl_engine.farm.wind import WindCondition, sample_wind
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import solve_farm

_LAYOUTS = {
    "turb3_row1": turb3_row1,
    "ablaincourt": ablaincourt,
    "horns_rev2": horns_rev2,
}
DTYPE = jnp.float64 if _ARGS.dtype == "f64" else jnp.float32


def _time_blocked(
    fn: Callable[[], Any], repeats: int
) -> tuple[float, float, list[float]]:
    """Return (compile_s, steady_median_s, samples). fn returns a jax array/pytree."""
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    # warmup a couple to settle
    for _ in range(3):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append(time.perf_counter() - t)
    return compile_s, statistics.median(samples), samples


def bench_solver(layout_name: str, batch: int, repeats: int) -> dict[str, Any]:
    layout = _LAYOUTS[layout_name]()
    n = int(layout.x.shape[0])
    keys = jax.random.split(jax.random.key(0), batch)
    winds = jax.vmap(sample_wind)(keys)
    yaws = (jax.random.uniform(jax.random.key(1), (batch, n)) * 20.0 - 10.0).astype(
        DTYPE
    )

    def one(wind: WindCondition, yaw: Any) -> Any:
        sol = solve_farm(layout, wind, yaw)
        return turbine_powers(sol.u, yaw)

    fn = jax.jit(jax.vmap(one))
    call = lambda: fn(winds, yaws)  # noqa: E731
    compile_s, med, _ = _time_blocked(call, repeats)
    return {
        "kind": "solver",
        "layout": layout_name,
        "turbines": n,
        "batch": batch,
        "compile_s": compile_s,
        "steady_s": med,
        "solves_per_s": batch / med,
    }


def bench_env_step(layout_name: str, batch: int, repeats: int) -> dict[str, Any]:
    # layout_name is validated against _LAYOUTS' keys (== LayoutName's members)
    # by the bench_solver dict lookup that runs alongside this in main().
    cfg = WindFarmEnvConfig(
        layout=cast(LayoutName, layout_name), n_envs=batch, horizon=1000
    )
    env = BatchedWindFarmEnv(cfg)
    n = env.n_turbines
    env.reset(jax.random.key(0))
    actions = jnp.zeros((batch, n), dtype=DTYPE)

    def call() -> Any:
        _obs, reward, _trunc = env.step(actions)
        return reward

    compile_s, med, _ = _time_blocked(call, repeats)
    return {
        "kind": "env_step",
        "layout": layout_name,
        "turbines": n,
        "batch": batch,
        "compile_s": compile_s,
        "steady_s": med,
        "env_steps_per_s": batch / med,
    }


def main() -> None:
    batches = [int(b) for b in _ARGS.batches.split(",")]
    layouts = _ARGS.layouts.split(",")
    results = []
    for kind, fn in (("solver", bench_solver), ("env_step", bench_env_step)):
        for layout_name in layouts:
            for batch in batches:
                try:
                    rec = fn(layout_name, batch, _ARGS.repeats)
                except Exception as e:  # OOM / other: record and continue
                    rec = {
                        "kind": kind,
                        "layout": layout_name,
                        "batch": batch,
                        "error": f"{type(e).__name__}: {e}"[:200],
                    }
                rec["platform"] = _ARGS.platform
                rec["dtype"] = _ARGS.dtype
                results.append(rec)
                print(json.dumps(rec), flush=True)
    with open(_ARGS.out, "a") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
