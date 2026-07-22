"""Differential agreement of windrl_engine's wake solve against raw FLORIS 3.5.

The reference is FLORIS driven through WFCRL's `FlorisInterface.from_case`, which
loads WFCRL's shipped GCH template (`simulators/floris/inputs/template/case.yaml`:
secondary steering / yaw-added recovery / transverse velocities on, crespo
constant 0.5). Expectations are read straight off the built FLORIS objects
(`fi.floris.flow_field`, `get_turbine_powers`) so this file never depends on the
windrl_engine solver internals it is checking. Both stacks run float64; the spec
targets <1e-10 relative, so 1e-9 leaves one order of margin.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("wfcrl")

import jax.numpy as jnp
from wfcrl.environments.data_cases import (
    FarmCase,
    floris_3t,
    floris_ablaincourt,
    floris_hornsrev2,
)
from wfcrl.interface import FlorisInterface

from windrl_engine.farm.layout import FarmLayout, ablaincourt, horns_rev2, turb3_row1
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import solve_farm

pytestmark = [pytest.mark.sim]

RTOL = 1e-9


def _reference(case: FarmCase, direction: float, speed: float, yaw: np.ndarray):
    out_dir = Path(tempfile.mkdtemp(prefix="floris_ref_")) / "case"
    fi = FlorisInterface.from_case(case, output_dir=str(out_dir))
    fi.fi.reinitialize(wind_speeds=[speed], wind_directions=[direction])
    fi.fi.calculate_wake(yaw_angles=yaw[None, None, :])
    ff = fi.fi.floris.flow_field
    return {
        "layout_x": np.asarray(fi.fi.layout_x, dtype=float),
        "layout_y": np.asarray(fi.fi.layout_y, dtype=float),
        "u": np.asarray(ff.u[0, 0]),
        "v": np.asarray(ff.v[0, 0]),
        "w": np.asarray(ff.w[0, 0]),
        "ti": np.asarray(ff.turbulence_intensity_field.squeeze(axis=(0, 1))).reshape(
            -1
        ),
        "powers": np.asarray(fi.fi.get_turbine_powers().flatten()),
    }


def _yaw(n: int, mixed: bool) -> np.ndarray:
    yaw = np.zeros(n)
    if mixed:
        yaw[: min(3, n)] = [20.0, -15.0, 10.0][:n]
    return yaw


# (layout builder, reference case, wind direction, wind speed, mixed-yaw, markers)
CASES = [
    (turb3_row1, floris_3t, 270.0, 8.0, False, ()),
    (turb3_row1, floris_3t, 270.0, 8.0, True, ()),
    (turb3_row1, floris_3t, 240.0, 11.0, False, ()),
    (turb3_row1, floris_3t, 83.5, 8.0, False, ()),
    (ablaincourt, floris_ablaincourt, 270.0, 8.0, False, ()),
    (ablaincourt, floris_ablaincourt, 270.0, 11.0, True, ()),
    (horns_rev2, floris_hornsrev2, 270.0, 8.0, False, (pytest.mark.slow,)),
]


@pytest.mark.parametrize(
    ("build_layout", "case", "direction", "speed", "mixed"),
    [
        pytest.param(
            *c[:5],
            marks=c[5],
            id=f"{c[1].num_turbines}t-{c[2]}-{c[3]}-{'yaw' if c[4] else 'flat'}",
        )
        for c in CASES
    ],
)
def test_flow_ti_and_power_match_floris(build_layout, case, direction, speed, mixed):
    layout: FarmLayout = build_layout()
    n = int(layout.x.shape[0])
    yaw = _yaw(n, mixed)
    ref = _reference(case, direction, speed, yaw)

    # Layout mismatch must fail loudly before any physics comparison.
    np.testing.assert_allclose(np.asarray(layout.x), ref["layout_x"], atol=1e-6)
    np.testing.assert_allclose(np.asarray(layout.y), ref["layout_y"], atol=1e-6)

    wind = WindCondition(speed=jnp.asarray(speed), direction=jnp.asarray(direction))
    solution = solve_farm(layout, wind, jnp.asarray(yaw))

    np.testing.assert_allclose(np.asarray(solution.u), ref["u"], rtol=RTOL, atol=1e-9)
    np.testing.assert_allclose(np.asarray(solution.v), ref["v"], rtol=RTOL, atol=1e-9)
    np.testing.assert_allclose(np.asarray(solution.w), ref["w"], rtol=RTOL, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(solution.turbulence_intensity), ref["ti"], rtol=RTOL
    )

    powers = turbine_powers(solution.u, jnp.asarray(yaw))
    np.testing.assert_allclose(np.asarray(powers), ref["powers"], rtol=RTOL)


def test_row3_270_8_zeroyaw_regression_anchor():
    # Hardcoded FLORIS 3.5 output so the suite also fails on reference-env drift.
    anchor_powers = np.array(
        [1691326.6483808453, 362733.9011381991, 322661.66918940115]
    )
    anchor_ti = np.array(
        [0.060203038909290935, 0.10191178582829581, 0.11959153122862591]
    )

    ref = _reference(floris_3t, 270.0, 8.0, np.zeros(3))
    np.testing.assert_allclose(ref["powers"], anchor_powers, rtol=1e-6)
    np.testing.assert_allclose(ref["ti"], anchor_ti, rtol=1e-6)

    layout = turb3_row1()
    wind = WindCondition(speed=jnp.asarray(8.0), direction=jnp.asarray(270.0))
    solution = solve_farm(layout, wind, jnp.zeros(3))
    powers = turbine_powers(solution.u, jnp.zeros(3))
    np.testing.assert_allclose(np.asarray(powers), anchor_powers, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(solution.turbulence_intensity), anchor_ti, rtol=1e-6
    )
