"""Differential agreement of windrl_engine's wake solve against frozen FLORIS goldens.

The reference is FLORIS 3.5 driven through WFCRL's shipped GCH template (secondary
steering / yaw-added recovery / transverse velocities on, crespo constant 0.5),
captured once into ``goldens/floris_v3.5.npz`` by
``generate_floris35_goldens.py`` and re-verified against the live WFCRL oracle at
freeze time. WFCRL itself is no longer a dependency, so this test is CI-safe: it
loads the golden arrays (u/v/w/TI/powers + layout coords per case) rather than
building any FLORIS object. Both stacks run float64; the spec targets <1e-10
relative, so 1e-9 leaves one order of margin.

A second test checks the v4.6.6 turbine model against ``goldens/floris_v4.6.6.npz``
(FLORIS 4.6.6 "defaults": byte-identical wake params, cosine-loss nrel_5MW).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from windrl_engine.farm.layout import FarmLayout, ablaincourt, horns_rev2, turb3_row1
from windrl_engine.farm.turbine import nrel5mw_v4
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import solve_farm

RTOL = 1e-9
GOLDENS = Path(__file__).parent / "goldens"


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(GOLDENS / name, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


# (golden case id, layout builder, wind direction, wind speed, markers)
CASES = [
    ("3t-270-8-flat", turb3_row1, 270.0, 8.0, ()),
    ("3t-270-8-yaw", turb3_row1, 270.0, 8.0, ()),
    ("3t-240-11-flat", turb3_row1, 240.0, 11.0, ()),
    ("3t-83.5-8-flat", turb3_row1, 83.5, 8.0, ()),
    ("7t-270-8-flat", ablaincourt, 270.0, 8.0, ()),
    ("7t-270-11-yaw", ablaincourt, 270.0, 11.0, ()),
    ("91t-270-8-flat", horns_rev2, 270.0, 8.0, (pytest.mark.slow,)),
]


@pytest.mark.parametrize(
    ("case_id", "build_layout", "direction", "speed"),
    [pytest.param(*c[:4], marks=c[4], id=c[0]) for c in CASES],
)
def test_flow_ti_and_power_match_floris(case_id, build_layout, direction, speed):
    golden = _load("floris_v3.5.npz")
    layout: FarmLayout = build_layout()
    yaw = golden[f"{case_id}/yaw"]

    # Layout mismatch must fail loudly before any physics comparison.
    np.testing.assert_allclose(
        np.asarray(layout.x), golden[f"{case_id}/layout_x"], atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(layout.y), golden[f"{case_id}/layout_y"], atol=1e-6
    )

    wind = WindCondition(speed=jnp.asarray(speed), direction=jnp.asarray(direction))
    solution = solve_farm(layout, wind, jnp.asarray(yaw))

    np.testing.assert_allclose(
        np.asarray(solution.u), golden[f"{case_id}/u"], rtol=RTOL, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(solution.v), golden[f"{case_id}/v"], rtol=RTOL, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(solution.w), golden[f"{case_id}/w"], rtol=RTOL, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(solution.turbulence_intensity), golden[f"{case_id}/ti"], rtol=RTOL
    )

    powers = turbine_powers(solution.u, jnp.asarray(yaw))
    np.testing.assert_allclose(
        np.asarray(powers), golden[f"{case_id}/powers"], rtol=RTOL
    )


def test_row3_270_8_zeroyaw_regression_anchor():
    # Hardcoded FLORIS 3.5 output; guards both the solver and the frozen golden
    # against silent drift (e.g. a corrupted or wrongly regenerated npz).
    anchor_powers = np.array(
        [1691326.6483808453, 362733.9011381991, 322661.66918940115]
    )
    anchor_ti = np.array(
        [0.060203038909290935, 0.10191178582829581, 0.11959153122862591]
    )

    golden = _load("floris_v3.5.npz")
    np.testing.assert_allclose(golden["3t-270-8-flat/powers"], anchor_powers, rtol=1e-6)
    np.testing.assert_allclose(golden["3t-270-8-flat/ti"], anchor_ti, rtol=1e-6)

    layout = turb3_row1()
    wind = WindCondition(speed=jnp.asarray(8.0), direction=jnp.asarray(270.0))
    solution = solve_farm(layout, wind, jnp.zeros(3))
    powers = turbine_powers(solution.u, jnp.zeros(3))
    np.testing.assert_allclose(np.asarray(powers), anchor_powers, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(solution.turbulence_intensity), anchor_ti, rtol=1e-6
    )


# (golden case id, layout builder, wind direction, wind speed, yaw)
V4_CASES = [
    ("row3_270_8_flat", turb3_row1, 270.0, 8.0, [0.0, 0.0, 0.0]),
    ("row3_270_8_yaw", turb3_row1, 270.0, 8.0, [20.0, -15.0, 10.0]),
    ("ablaincourt_240_10_flat", ablaincourt, 240.0, 10.0, [0.0] * 7),
]


@pytest.mark.parametrize(
    ("case_id", "build_layout", "direction", "speed", "yaw"),
    [pytest.param(*c, id=c[0]) for c in V4_CASES],
)
def test_flow_and_power_match_floris_v4(case_id, build_layout, direction, speed, yaw):
    golden = _load("floris_v4.6.6.npz")
    layout: FarmLayout = build_layout()
    turbine = nrel5mw_v4()
    yaw_arr = jnp.asarray(yaw)

    wind = WindCondition(speed=jnp.asarray(speed), direction=jnp.asarray(direction))
    solution = solve_farm(layout, wind, yaw_arr, turbine=turbine)

    np.testing.assert_allclose(
        np.asarray(solution.u), golden[f"{case_id}/u"], rtol=RTOL, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(solution.v), golden[f"{case_id}/v"], rtol=RTOL, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(solution.w), golden[f"{case_id}/w"], rtol=RTOL, atol=1e-9
    )

    powers = turbine_powers(solution.u, yaw_arr, turbine=turbine)
    np.testing.assert_allclose(
        np.asarray(powers), golden[f"{case_id}/powers"], rtol=RTOL
    )
