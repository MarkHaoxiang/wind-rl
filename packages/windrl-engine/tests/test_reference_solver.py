"""Differential agreement of windrl_engine's wake solve against a frozen FLORIS golden.

The reference is FLORIS 4.6.6 driven through its ``"defaults"`` configuration
(GCH: secondary steering / yaw-added recovery / transverse velocities on, crespo
constant 0.5, cosine-loss nrel_5MW), captured once into ``goldens/floris_v4.6.6.npz``
by ``generate_floris_goldens.py``. FLORIS itself is not a dependency, so this test
is CI-safe: it loads the golden arrays (u/v/w/TI/powers per case) rather than
building any FLORIS object. Both stacks run float64; the spec targets <1e-10
relative, so 1e-9 leaves one order of margin.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from windrl_engine.farm.layout import FarmLayout, ablaincourt, turb3_row1
from windrl_engine.farm.turbine import nrel5mw_v4
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import solve_farm

RTOL = 1e-9
GOLDENS = Path(__file__).parent / "goldens"


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(GOLDENS / name, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


# (golden case id, layout builder, wind direction, wind speed, yaw)
CASES = [
    ("row3_270_8_flat", turb3_row1, 270.0, 8.0, [0.0, 0.0, 0.0]),
    ("row3_270_8_yaw", turb3_row1, 270.0, 8.0, [20.0, -15.0, 10.0]),
    ("ablaincourt_240_10_flat", ablaincourt, 240.0, 10.0, [0.0] * 7),
]


@pytest.mark.parametrize(
    ("case_id", "build_layout", "direction", "speed", "yaw"),
    [pytest.param(*c, id=c[0]) for c in CASES],
)
def test_flow_ti_and_power_match_floris(case_id, build_layout, direction, speed, yaw):
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
    np.testing.assert_allclose(
        np.asarray(solution.turbulence_intensity),
        golden[f"{case_id}/turbulence_intensities"],
        rtol=RTOL,
    )

    powers = turbine_powers(solution.u, yaw_arr, turbine=turbine)
    np.testing.assert_allclose(
        np.asarray(powers), golden[f"{case_id}/powers"], rtol=RTOL
    )


def test_row3_270_8_zeroyaw_regression_anchor():
    # Hardcoded FLORIS 4.6.6 output; guards both the solver and the frozen golden
    # against silent drift (e.g. a corrupted or wrongly regenerated npz).
    anchor_powers = np.array([1753954.45917917, 356384.89485703, 344414.69675443])
    anchor_ti = np.array([0.06021557, 0.10448007, 0.12594702])

    golden = _load("floris_v4.6.6.npz")
    np.testing.assert_allclose(
        golden["row3_270_8_flat/powers"], anchor_powers, rtol=1e-6
    )
    np.testing.assert_allclose(
        golden["row3_270_8_flat/turbulence_intensities"], anchor_ti, rtol=1e-6
    )

    layout = turb3_row1()
    turbine = nrel5mw_v4()
    wind = WindCondition(speed=jnp.asarray(8.0), direction=jnp.asarray(270.0))
    solution = solve_farm(layout, wind, jnp.zeros(3), turbine=turbine)
    powers = turbine_powers(solution.u, jnp.zeros(3), turbine=turbine)
    np.testing.assert_allclose(np.asarray(powers), anchor_powers, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(solution.turbulence_intensity), anchor_ti, rtol=1e-6
    )
