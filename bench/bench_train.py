"""MAPPO (Mava ff_mappo) training throughput matrix (Benchmark C).

Shells out to ``windrl_train.train`` once per (platform, layout, num_envs) point
and parses Mava's console ``ACTOR``/``EVALUATOR`` "Steps per second" lines. Mava
(Anakin) fuses the whole rollout+update+eval loop into one jitted program, so
the first eval interval's SPS includes XLA compilation and later intervals are
steady-state; we report the compile-inclusive first value and the steady max
separately.

Run from the windrl-train package dir with its venv's python. ``JAX_PLATFORMS``
is set per point (``cuda`` vs ``cpu``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

_ACTOR = re.compile(r"ACTOR - Steps per second:\s*([\d.]+)")
_EVAL = re.compile(r"EVALUATOR - Steps per second:\s*([\d.]+)")


def run_point(
    python: str, platform: str, layout: str, num_envs: int, rollout: int, updates: int
) -> dict[str, object]:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cuda" if platform == "gpu" else "cpu"
    env["WIND_RL_WANDB_MODE"] = "disabled"
    cmd = [
        python,
        "-m",
        "windrl_train.train",
        f"env.kwargs.layout={layout}",
        "env.kwargs.horizon=200",
        f"arch.num_envs={num_envs}",
        "system.update_batch_size=1",
        f"system.rollout_length={rollout}",
        f"system.num_updates={updates}",
        "arch.num_evaluation=4",
        "arch.num_eval_episodes=4",
        "system.num_minibatches=2",
        "arch.absolute_metric=False",
        "network=mlp",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    wall = time.perf_counter() - t0
    out = proc.stdout + proc.stderr
    actor = [float(m) for m in _ACTOR.findall(out)]
    evalr = [float(m) for m in _EVAL.findall(out)]
    rec: dict[str, object] = {
        "kind": "train",
        "platform": platform,
        "layout": layout,
        "num_envs": num_envs,
        "rollout_length": rollout,
        "num_updates": updates,
        "wall_s": wall,
        "actor_sps_first": actor[0] if actor else None,
        "actor_sps_steady": max(actor[1:]) if len(actor) > 1 else None,
        "eval_sps_steady": max(evalr[1:]) if len(evalr) > 1 else None,
        "returncode": proc.returncode,
    }
    if proc.returncode != 0:
        rec["error_tail"] = out[-500:]
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--python", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--platform", choices=["cpu", "gpu"], required=True)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--updates", type=int, default=32)
    # points: "layout:num_envs,layout:num_envs,..."
    p.add_argument("--points", required=True)
    args = p.parse_args()

    points = []
    for tok in args.points.split(","):
        layout, ne = tok.split(":")
        points.append((layout, int(ne)))

    for layout, ne in points:
        rec = run_point(
            args.python, args.platform, layout, ne, args.rollout, args.updates
        )
        print(json.dumps(rec), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["returncode"] != 0:
            print(f"  FAILED: {rec.get('error_tail')}", file=sys.stderr)


if __name__ == "__main__":
    main()
