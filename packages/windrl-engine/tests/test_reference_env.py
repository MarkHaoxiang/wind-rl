"""Differential agreement of windrl_engine's env against frozen WFCRL/FLORIS trajectory goldens."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from windrl_engine.env.env import reset as wr_reset
from windrl_engine.env.env import step as wr_step
from windrl_engine.farm.layout import ablaincourt, turb3_row1
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import solve_farm

# Golden cases were captured once from WFCRL's central multi-agent env (FLORIS
# 3.5 backend) by generate_wfcrl_env_goldens.py, which is no longer runnable
# now that wfcrl is gone -- see its docstring. Each case stores the reset
# wind/yaw, the seeded delta-action stream, per-step
# observations/rewards/truncation, and the control config read live off that
# env; this module replays the deltas through windrl_engine and asserts
# parity, so it never imports wfcrl itself.
RTOL = 1e-6
# WFCRL rounds the integrated yaw through float32 at the env boundary while
# windrl_engine stays float64, so env-level quantities agree to ~1e-6
# relative, looser than the 1e-9 raw-solve tolerance.
ATOL = 1e-5
GOLDENS = Path(__file__).parent / "goldens"


def _load() -> dict[str, np.ndarray]:
    with np.load(GOLDENS / "wfcrl_env_trajectories.npz", allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _drive_windrl(build_layout, speed, direction, deltas, ctrl):
    layout = build_layout()
    wind = WindCondition(speed=jnp.asarray(speed), direction=jnp.asarray(direction))
    state, obs0 = wr_reset(layout, jax.random.key(0), wind=wind)
    reset_obs = {
        "yaw": np.asarray(obs0.yaw),
        "wind_speed": np.asarray(obs0.wind_speed),
        "wind_direction": np.asarray(obs0.wind_direction),
        "freewind": np.asarray(obs0.freewind),
    }
    steps = []
    for delta in deltas:
        state, obs, reward, trunc = wr_step(
            layout,
            state,
            jnp.asarray(delta),
            yaw_step=ctrl["yaw_step"],
            load_coef=ctrl["load_coef"],
            horizon=ctrl["horizon"],
        )
        flow = solve_farm(layout, state.wind, state.yaw)
        power_mw = np.asarray(turbine_powers(flow.u, state.yaw)) / 1e6
        steps.append(
            {
                "yaw": np.asarray(obs.yaw),
                "freewind": np.asarray(obs.freewind),
                "wind_speed": np.asarray(obs.wind_speed),
                "wind_direction": np.asarray(obs.wind_direction),
                "reward": float(np.asarray(reward)),
                "power_mw": power_mw,
                "truncated": bool(np.asarray(trunc)),
            }
        )
    return reset_obs, steps


def _replay(case_id: str, build_layout, speed, direction):
    g = _load()
    deltas = list(g[f"{case_id}/deltas"])
    ctrl = {
        "yaw_step": float(g[f"{case_id}/ctrl/yaw_step"]),
        "load_coef": float(g[f"{case_id}/ctrl/load_coef"]),
        "horizon": int(g[f"{case_id}/ctrl/horizon"]),
    }
    reset, steps = _drive_windrl(build_layout, speed, direction, deltas, ctrl)
    return g, reset, steps


# (golden case id, layout builder, speed, direction)
ENV_CASES = [
    ("row3-zeros", turb3_row1, 8.0, 270.0),
    ("row3-maxdelta", turb3_row1, 8.0, 270.0),
    ("row3-random", turb3_row1, 11.0, 240.0),
    ("ablaincourt-random", ablaincourt, 11.0, 270.0),
]


@pytest.mark.parametrize(
    ("case_id", "build_layout", "speed", "direction"),
    [pytest.param(*c, id=c[0]) for c in ENV_CASES],
)
def test_env_matches_wfcrl_per_step(case_id, build_layout, speed, direction):
    g, our_reset, our_steps = _replay(case_id, build_layout, speed, direction)

    np.testing.assert_allclose(
        our_reset["wind_speed"], g[f"{case_id}/reset/wind_speed"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        our_reset["wind_direction"],
        g[f"{case_id}/reset/wind_direction"],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(our_reset["yaw"], g[f"{case_id}/reset/yaw"], atol=ATOL)

    for i, ours in enumerate(our_steps):
        ctx = f"{case_id} step {i}"
        np.testing.assert_allclose(
            ours["yaw"],
            g[f"{case_id}/step/yaw"][i],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} yaw",
        )
        np.testing.assert_allclose(
            ours["freewind"],
            g[f"{case_id}/step/freewind"][i],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} freewind",
        )
        np.testing.assert_allclose(
            ours["wind_speed"],
            g[f"{case_id}/step/wind_speed"][i],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} wind_speed",
        )
        np.testing.assert_allclose(
            ours["wind_direction"],
            g[f"{case_id}/step/wind_direction"][i],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} wind_direction",
        )
        np.testing.assert_allclose(
            ours["power_mw"],
            g[f"{case_id}/step/power_mw"][i],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} power",
        )
        np.testing.assert_allclose(
            ours["reward"],
            g[f"{case_id}/step/reward"][i],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} reward",
        )
        assert ours["truncated"] == bool(g[f"{case_id}/step/truncated"][i]), (
            f"{ctx} truncated"
        )


def test_duty_cycle_limiter_fires_and_matches():
    case_id = "duty-cycle"
    g, _our_reset, our_steps = _replay(case_id, turb3_row1, 8.0, 270.0)
    # A constant +max delta must trip the 10%-actuation limiter within 12 steps.
    assert bool(g[f"{case_id}/step/fired"].any()), (
        "reference duty-cycle limiter never fired"
    )
    for i, ours in enumerate(our_steps):
        np.testing.assert_allclose(
            ours["yaw"],
            g[f"{case_id}/step/yaw"][i],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"yaw trajectory diverges at step {i} (limiter parity)",
        )


def test_truncation_boundary_matches_wfcrl():
    # WFCRL's reset burn-in advances its step counter to 1, so an episode yields
    # max_num_steps - 1 agent steps before truncation. The boundary was measured
    # from the live reference (max_num_steps=6) and frozen into the golden.
    case_id = "truncation-boundary"
    g, _our_reset, our_steps = _replay(case_id, turb3_row1, 8.0, 270.0)

    ref_trunc = g[f"{case_id}/step/truncated"]
    ref_idx = int(np.argmax(ref_trunc)) if ref_trunc.any() else None
    assert ref_idx is not None, "reference never truncated within the horizon"

    our_idx = next((i for i, s in enumerate(our_steps) if s["truncated"]), None)
    assert our_idx == ref_idx, (
        f"first-truncation step index mismatch: ours={our_idx}, wfcrl={ref_idx}"
    )

    for i, ours in enumerate(our_steps):
        ctx = f"boundary step {i} (ref truncates at {ref_idx})"
        np.testing.assert_allclose(
            ours["yaw"],
            g[f"{case_id}/step/yaw"][i],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} yaw",
        )
        np.testing.assert_allclose(
            ours["wind_speed"],
            g[f"{case_id}/step/wind_speed"][i],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} wind_speed",
        )
        np.testing.assert_allclose(
            ours["wind_direction"],
            g[f"{case_id}/step/wind_direction"][i],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} wind_direction",
        )
        np.testing.assert_allclose(
            ours["reward"],
            g[f"{case_id}/step/reward"][i],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} reward",
        )


def test_row3_reset_and_first_step_anchor():
    # Hardcoded WFCRL/FLORIS output; guards both the env and the frozen golden.
    anchor_ws = np.array([7.973632994592287, 4.896107438268562, 4.728997035154612])
    anchor_wd = np.array([270.26410649314425, 270.5904376583105, 270.7200314568065])
    anchor_reward = 1.5290698105938687
    anchor_power_mw = np.array(
        [1.6913266483808453, 0.3627339011381991, 0.32266166918940115]
    )

    g, our_reset, our_steps = _replay("row3-zeros", turb3_row1, 8.0, 270.0)
    np.testing.assert_allclose(g["row3-zeros/reset/wind_speed"], anchor_ws, rtol=1e-6)
    np.testing.assert_allclose(
        g["row3-zeros/reset/wind_direction"], anchor_wd, rtol=1e-6
    )
    np.testing.assert_allclose(g["row3-zeros/step/reward"][0], anchor_reward, rtol=1e-6)
    np.testing.assert_allclose(
        g["row3-zeros/step/power_mw"][0], anchor_power_mw, rtol=1e-6
    )

    np.testing.assert_allclose(our_reset["wind_speed"], anchor_ws, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        our_reset["wind_direction"], anchor_wd, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        our_steps[0]["reward"], anchor_reward, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        our_steps[0]["power_mw"], anchor_power_mw, rtol=RTOL, atol=ATOL
    )
