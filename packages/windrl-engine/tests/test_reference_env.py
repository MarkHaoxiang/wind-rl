"""Differential agreement of windrl_engine's env against the WFCRL central env.

Both envs are reset with the same wind override (`options={"wind_speed", ...}`
vs `WindCondition`) and driven with identical delta-action streams. The control
config (yaw step, load coefficient, horizon) is read off the live WFCRL env so
this file never hardcodes assumptions that could drift from the reference.

WFCRL rounds the integrated yaw through float32 at the env boundary (spec §2)
while windrl_engine stays float64, so env-level quantities agree to ~1e-6
relative, not the 1e-9 of the raw solve.
"""

import numpy as np
import pytest

pytest.importorskip("wfcrl")

import jax
import jax.numpy as jnp
from wfcrl.environments import make

from windrl_engine.env.env import reset as wr_reset
from windrl_engine.env.env import step as wr_step
from windrl_engine.farm.layout import ablaincourt, turb3_row1
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import solve_farm

pytestmark = [pytest.mark.sim]

RTOL = 1e-6
ATOL = 1e-5


def _zeros(n: int, step: float):
    return lambda k: np.zeros(n)


def _max(n: int, step: float):
    return lambda k: np.full(n, step)


def _random(n: int, step: float):
    rng = np.random.default_rng(0)
    return lambda k: rng.uniform(-step, step, n).astype(np.float32)


def _drive_wfcrl(env_id, speed, direction, stream, n_steps, max_num_steps=None):
    kwargs = {} if max_num_steps is None else {"max_num_steps": max_num_steps}
    env = make(env_id, controls=["yaw"], log=False, **kwargs)
    n = env.num_turbines
    yaw_step = float(env.controls["yaw"][2])
    load_coef = float(env.load_coef)
    horizon = int(env.mdp.horizon)
    reset_obs = env.reset(options={"wind_speed": speed, "wind_direction": direction})
    deltas = [np.asarray(stream(n, yaw_step)(k)) for k in range(n_steps)]
    prev_yaw = np.asarray(reset_obs["yaw"], dtype=float)
    steps = []
    for delta in deltas:
        requested = delta.copy()
        obs, reward, _term, trunc, info = env.step(
            {"yaw": delta.astype(np.float32).copy()}
        )
        yaw = np.asarray(obs["yaw"], dtype=float)
        fired = bool(np.any(requested != 0.0) and np.allclose(yaw, prev_yaw))
        steps.append(
            {
                "yaw": yaw,
                "freewind": np.asarray(obs["freewind_measurements"], dtype=float),
                "wind_speed": np.asarray(obs["wind_speed"], dtype=float),
                "wind_direction": np.asarray(obs["wind_direction"], dtype=float),
                "reward": float(np.asarray(reward).reshape(-1)[0]),
                "power_mw": np.asarray(info["power"], dtype=float),
                "truncated": bool(trunc),
                "fired": fired,
            }
        )
        prev_yaw = yaw
    ctrl = {"yaw_step": yaw_step, "load_coef": load_coef, "horizon": horizon}
    return reset_obs, deltas, steps, ctrl


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


ENV_CASES = [
    ("Turb3_Row1_Floris", turb3_row1, 8.0, 270.0, _zeros, 5, "row3-zeros"),
    ("Turb3_Row1_Floris", turb3_row1, 8.0, 270.0, _max, 12, "row3-maxdelta"),
    ("Turb3_Row1_Floris", turb3_row1, 11.0, 240.0, _random, 6, "row3-random"),
    ("Ablaincourt_Floris", ablaincourt, 11.0, 270.0, _random, 6, "ablaincourt-random"),
]


@pytest.mark.parametrize(
    ("env_id", "build_layout", "speed", "direction", "stream", "n_steps"),
    [pytest.param(*c[:6], id=c[6]) for c in ENV_CASES],
)
def test_env_matches_wfcrl_per_step(
    env_id, build_layout, speed, direction, stream, n_steps
):
    ref_reset, deltas, ref_steps, ctrl = _drive_wfcrl(
        env_id, speed, direction, stream, n_steps
    )
    our_reset, our_steps = _drive_windrl(build_layout, speed, direction, deltas, ctrl)

    np.testing.assert_allclose(
        our_reset["wind_speed"], ref_reset["wind_speed"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        our_reset["wind_direction"], ref_reset["wind_direction"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        our_reset["yaw"], np.asarray(ref_reset["yaw"]), atol=ATOL
    )

    for i, (ours, ref) in enumerate(zip(our_steps, ref_steps, strict=True)):
        ctx = f"{env_id} step {i}"
        np.testing.assert_allclose(
            ours["yaw"], ref["yaw"], rtol=RTOL, atol=ATOL, err_msg=f"{ctx} yaw"
        )
        np.testing.assert_allclose(
            ours["freewind"],
            ref["freewind"],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} freewind",
        )
        np.testing.assert_allclose(
            ours["wind_speed"],
            ref["wind_speed"],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} wind_speed",
        )
        np.testing.assert_allclose(
            ours["wind_direction"],
            ref["wind_direction"],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} wind_direction",
        )
        np.testing.assert_allclose(
            ours["power_mw"],
            ref["power_mw"],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} power",
        )
        np.testing.assert_allclose(
            ours["reward"], ref["reward"], rtol=RTOL, atol=ATOL, err_msg=f"{ctx} reward"
        )
        assert ours["truncated"] == ref["truncated"], f"{ctx} truncated"


def test_duty_cycle_limiter_fires_and_matches():
    env_id = "Turb3_Row1_Floris"
    _reset, deltas, ref_steps, ctrl = _drive_wfcrl(env_id, 8.0, 270.0, _max, 12)
    # A constant +max delta must trip the 10%-actuation limiter within 12 steps.
    fire_steps = [i for i, s in enumerate(ref_steps) if s["fired"]]
    assert fire_steps, "reference duty-cycle limiter never fired"

    _our_reset, our_steps = _drive_windrl(turb3_row1, 8.0, 270.0, deltas, ctrl)
    for i, (ours, ref) in enumerate(zip(our_steps, ref_steps, strict=True)):
        np.testing.assert_allclose(
            ours["yaw"],
            ref["yaw"],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"yaw trajectory diverges at step {i} (limiter parity)",
        )


def test_truncation_boundary_matches_wfcrl():
    # WFCRL's reset burn-in advances _num_iter to 1, so an episode yields
    # max_num_steps - 1 agent steps before truncation. The expected boundary is
    # measured from the live reference, never assumed from our implementation.
    max_num_steps = 6
    _ref_reset, deltas, ref_steps, ctrl = _drive_wfcrl(
        "Turb3_Row1_Floris", 8.0, 270.0, _zeros, max_num_steps, max_num_steps
    )
    ref_idx = next((i for i, s in enumerate(ref_steps) if s["truncated"]), None)
    assert ref_idx is not None, "reference never truncated within the horizon"

    _our_reset, our_steps = _drive_windrl(turb3_row1, 8.0, 270.0, deltas, ctrl)
    our_idx = next((i for i, s in enumerate(our_steps) if s["truncated"]), None)

    assert our_idx == ref_idx, (
        f"first-truncation step index mismatch: ours={our_idx}, "
        f"wfcrl={ref_idx} (max_num_steps={max_num_steps})"
    )
    assert (our_idx + 1) == (ref_idx + 1), (
        f"agent-step count per episode mismatch: ours={our_idx + 1}, "
        f"wfcrl={ref_idx + 1}"
    )
    for i, (ours, ref) in enumerate(zip(our_steps, ref_steps, strict=True)):
        ctx = f"boundary step {i} (ref truncates at {ref_idx})"
        np.testing.assert_allclose(
            ours["yaw"], ref["yaw"], rtol=RTOL, atol=ATOL, err_msg=f"{ctx} yaw"
        )
        np.testing.assert_allclose(
            ours["wind_speed"],
            ref["wind_speed"],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} wind_speed",
        )
        np.testing.assert_allclose(
            ours["wind_direction"],
            ref["wind_direction"],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{ctx} wind_direction",
        )
        np.testing.assert_allclose(
            ours["reward"], ref["reward"], rtol=RTOL, atol=ATOL, err_msg=f"{ctx} reward"
        )


def test_row3_reset_and_first_step_anchor():
    # Hardcoded WFCRL/FLORIS output; catches reference-environment drift.
    anchor_ws = np.array([7.973632994592287, 4.896107438268562, 4.728997035154612])
    anchor_wd = np.array([270.26410649314425, 270.5904376583105, 270.7200314568065])
    anchor_reward = 1.5290698105938687
    anchor_power_mw = np.array(
        [1.6913266483808453, 0.3627339011381991, 0.32266166918940115]
    )

    ref_reset, deltas, ref_steps, ctrl = _drive_wfcrl(
        "Turb3_Row1_Floris", 8.0, 270.0, _zeros, 1
    )
    np.testing.assert_allclose(ref_reset["wind_speed"], anchor_ws, rtol=1e-6)
    np.testing.assert_allclose(ref_reset["wind_direction"], anchor_wd, rtol=1e-6)
    np.testing.assert_allclose(ref_steps[0]["reward"], anchor_reward, rtol=1e-6)
    np.testing.assert_allclose(ref_steps[0]["power_mw"], anchor_power_mw, rtol=1e-6)

    our_reset, our_steps = _drive_windrl(turb3_row1, 8.0, 270.0, deltas, ctrl)
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
