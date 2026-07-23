"""Freeze WFCRL env-level trajectory goldens for the differential reference tests.

WARNING -- NOT runnable after the WFCRL removal wave. This script imports
``wfcrl`` (the central multi-agent env + FLORIS 3.5) to capture, once, the
per-step observations/rewards/truncation the ``windrl_engine`` env is checked
against. Once the wfcrl fork submodule is gone it cannot be reinstalled from this
tree; regenerating these goldens requires reinstating the fork. It is kept purely
for provenance -- it documents exactly how ``goldens/wfcrl_env_trajectories.npz``
was produced.

Run in the CURRENT project venv (wfcrl still importable), float64 enabled:

    JAX_ENABLE_X64=1 uv run python \
        packages/windrl-engine/tests/generate_wfcrl_env_goldens.py

Each case records the reset wind + yaw, the exact (seeded) delta-action stream,
and per-step yaw/freewind/wind/reward/power/truncated/fired, plus the control
config (yaw step, load coef, horizon) read live off the WFCRL env. The reference
tests replay the stored deltas through ``windrl_engine`` and assert parity.
"""

from pathlib import Path

import numpy as np
from wfcrl.environments import make


def _zeros(n: int, step: float):
    return lambda k: np.zeros(n)


def _max(n: int, step: float):
    return lambda k: np.full(n, step)


def _random(n: int, step: float):
    rng = np.random.default_rng(0)
    return lambda k: rng.uniform(-step, step, n).astype(np.float32)


STREAMS = {"zeros": _zeros, "max": _max, "random": _random}

# (case id, env id, speed, direction, stream name, n_steps, max_num_steps|None)
CASES = [
    ("row3-zeros", "Turb3_Row1_Floris", 8.0, 270.0, "zeros", 5, None),
    ("row3-maxdelta", "Turb3_Row1_Floris", 8.0, 270.0, "max", 12, None),
    ("row3-random", "Turb3_Row1_Floris", 11.0, 240.0, "random", 6, None),
    ("ablaincourt-random", "Ablaincourt_Floris", 11.0, 270.0, "random", 6, None),
    ("duty-cycle", "Turb3_Row1_Floris", 8.0, 270.0, "max", 12, None),
    ("truncation-boundary", "Turb3_Row1_Floris", 8.0, 270.0, "zeros", 6, 6),
]


def drive(env_id, speed, direction, stream_name, n_steps, max_num_steps):
    kwargs = {} if max_num_steps is None else {"max_num_steps": max_num_steps}
    env = make(env_id, controls=["yaw"], log=False, **kwargs)
    n = env.num_turbines
    yaw_step = float(env.controls["yaw"][2])
    load_coef = float(env.load_coef)
    horizon = int(env.mdp.horizon)
    reset_obs = env.reset(options={"wind_speed": speed, "wind_direction": direction})
    stream = STREAMS[stream_name](n, yaw_step)
    deltas = [np.asarray(stream(k)) for k in range(n_steps)]
    prev_yaw = np.asarray(reset_obs["yaw"], dtype=float)

    yaws, freewinds, wspeeds, wdirs, rewards, powers, truncs, fireds = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for delta in deltas:
        requested = delta.copy()
        obs, reward, _term, trunc, info = env.step(
            {"yaw": delta.astype(np.float32).copy()}
        )
        yaw = np.asarray(obs["yaw"], dtype=float)
        fired = bool(np.any(requested != 0.0) and np.allclose(yaw, prev_yaw))
        yaws.append(yaw)
        freewinds.append(np.asarray(obs["freewind_measurements"], dtype=float))
        wspeeds.append(np.asarray(obs["wind_speed"], dtype=float))
        wdirs.append(np.asarray(obs["wind_direction"], dtype=float))
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
        powers.append(np.asarray(info["power"], dtype=float))
        truncs.append(bool(trunc))
        fireds.append(fired)
        prev_yaw = yaw

    return {
        "reset/yaw": np.asarray(reset_obs["yaw"], dtype=float),
        "reset/wind_speed": np.asarray(reset_obs["wind_speed"], dtype=float),
        "reset/wind_direction": np.asarray(reset_obs["wind_direction"], dtype=float),
        "reset/freewind": np.asarray(reset_obs["freewind_measurements"], dtype=float),
        "deltas": np.asarray(deltas, dtype=float),
        "step/yaw": np.asarray(yaws, dtype=float),
        "step/freewind": np.asarray(freewinds, dtype=float),
        "step/wind_speed": np.asarray(wspeeds, dtype=float),
        "step/wind_direction": np.asarray(wdirs, dtype=float),
        "step/reward": np.asarray(rewards, dtype=float),
        "step/power_mw": np.asarray(powers, dtype=float),
        "step/truncated": np.asarray(truncs, dtype=bool),
        "step/fired": np.asarray(fireds, dtype=bool),
        "ctrl/yaw_step": np.asarray(yaw_step, dtype=float),
        "ctrl/load_coef": np.asarray(load_coef, dtype=float),
        "ctrl/horizon": np.asarray(horizon, dtype=np.int64),
        "meta/max_num_steps": np.asarray(
            -1 if max_num_steps is None else max_num_steps, dtype=np.int64
        ),
    }


def main() -> None:
    out: dict[str, np.ndarray] = {}
    for case_id, env_id, speed, direction, stream_name, n_steps, mns in CASES:
        result = drive(env_id, speed, direction, stream_name, n_steps, mns)
        for key, value in result.items():
            out[f"{case_id}/{key}"] = value

    goldens_dir = Path(__file__).parent / "goldens"
    goldens_dir.mkdir(exist_ok=True)
    path = goldens_dir / "wfcrl_env_trajectories.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
