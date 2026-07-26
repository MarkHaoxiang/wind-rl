"""Differential agreement of windrl_engine's wake solve against a live FLORIS reference.

The reference is FLORIS 4.6.6 driven through its ``"defaults"`` configuration
(GCH: secondary steering / yaw-added recovery / transverse velocities on, crespo
constant 0.5, cosine-loss nrel_5MW), computed in-process once per session. Both
stacks run float64.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from floris import FlorisModel

from windrl_engine.farm.layout import FarmLayout, ablaincourt, turb3_row1
from windrl_engine.farm.turbine import nrel5mw_v4
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import solve_farm

RTOL = 1e-12
# v and w cross zero across the rotor plane, so their relative error blows up where
# the reference is near-zero; they get a looser rtol and lean on the absolute floor.
TRANSVERSE_RTOL = 1e-9
AMBIENT_TI = 0.06

# (case id, layout builder, wind direction [deg, 270 = from the west], speed, yaw)
CASES: list[tuple[str, Callable[[], FarmLayout], float, float, list[float]]] = [
    ("row3_270_8_flat", turb3_row1, 270.0, 8.0, [0.0, 0.0, 0.0]),
    ("row3_270_8_yaw", turb3_row1, 270.0, 8.0, [20.0, -15.0, 10.0]),
    ("ablaincourt_240_10_flat", ablaincourt, 240.0, 10.0, [0.0] * 7),
    # Power-curve corners the 8-10 m/s cases never reach: below cut-in the two
    # waked turbines produce exactly 0 W; at 4 m/s the row is non-monotone
    # (turbine 1 dead, turbine 2 recovered to ~42 kW by wake-added mixing); at
    # 15 m/s the front two sit exactly on the 5 MW rated plateau.
    ("row3_270_3_below_cutin", turb3_row1, 270.0, 3.0, [0.0, 0.0, 0.0]),
    ("row3_270_4_partial_cutin", turb3_row1, 270.0, 4.0, [0.0, 0.0, 0.0]),
    ("row3_270_15_rated", turb3_row1, 270.0, 15.0, [0.0, 0.0, 0.0]),
    # Yawed *and* non-axis-aligned: every turbine deflects a wake, and no rotor
    # plane shares a wind-frame x with another, so the deflection/secondary-
    # steering path is exercised off the degenerate 270 deg geometry.
    (
        "ablaincourt_313_9_yaw",
        ablaincourt,
        313.0,
        9.0,
        [10.0, -20.0, 5.0, 0.0, -8.0, 15.0, -3.0],
    ),
]


def _floris_reference(
    layout: FarmLayout, direction: float, speed: float, yaw: list[float]
) -> dict[str, npt.NDArray[np.float64]]:
    fmodel = FlorisModel("defaults")
    fmodel.set(
        layout_x=np.asarray(layout.x),
        layout_y=np.asarray(layout.y),
        wind_directions=[direction],
        wind_speeds=[speed],
        turbulence_intensities=[AMBIENT_TI],
        yaw_angles=np.asarray([yaw]),
    )
    fmodel.run()
    flow = fmodel.core.flow_field
    return {
        "powers": fmodel.get_turbine_powers().flatten(),
        "turbulence_intensities": fmodel.get_turbine_TIs().flatten(),
        "u": flow.u[0],
        "v": flow.v[0],
        "w": flow.w[0],
    }


@pytest.fixture(scope="session")
def floris_reference() -> dict[str, dict[str, npt.NDArray[np.float64]]]:
    return {
        case_id: _floris_reference(build_layout(), direction, speed, yaw)
        for case_id, build_layout, direction, speed, yaw in CASES
    }


@pytest.mark.parametrize(
    ("case_id", "build_layout", "direction", "speed", "yaw"),
    [pytest.param(*c, id=c[0]) for c in CASES],
)
def test_flow_ti_and_power_match_floris(
    case_id: str,
    build_layout: Callable[[], FarmLayout],
    direction: float,
    speed: float,
    yaw: tuple[float, ...],
    floris_reference: dict[str, dict[str, npt.NDArray[np.float64]]],
) -> None:
    reference = floris_reference[case_id]
    layout: FarmLayout = build_layout()
    turbine = nrel5mw_v4()
    yaw_arr = jnp.asarray(yaw)

    wind = WindCondition(speed=jnp.asarray(speed), direction=jnp.asarray(direction))
    solution = solve_farm(layout, wind, yaw_arr, turbine=turbine)

    np.testing.assert_allclose(np.asarray(solution.u), reference["u"], rtol=RTOL)
    np.testing.assert_allclose(
        np.asarray(solution.v), reference["v"], rtol=TRANSVERSE_RTOL, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(solution.w), reference["w"], rtol=TRANSVERSE_RTOL, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(solution.turbulence_intensity),
        reference["turbulence_intensities"],
        rtol=RTOL,
    )

    powers = turbine_powers(solution.u, yaw_arr, turbine=turbine)
    np.testing.assert_allclose(np.asarray(powers), reference["powers"], rtol=RTOL)


def test_power_curve_corner_cases_sit_on_the_corners_they_were_chosen_for(
    floris_reference: dict[str, dict[str, npt.NDArray[np.float64]]],
) -> None:
    # The corner cases only test the cut-in/rated clamps while the geometry keeps
    # putting them there; pin the corner so a re-tuned CASES entry that quietly
    # slides back into the smooth part of the power curve fails loudly.
    below_cutin = floris_reference["row3_270_3_below_cutin"]["powers"]
    assert below_cutin[0] > 0.0
    assert np.array_equal(below_cutin[1:], np.zeros(2))

    partial = floris_reference["row3_270_4_partial_cutin"]["powers"]
    assert partial[1] == 0.0
    assert partial[2] > 0.0  # wake-added mixing recovers turbine 2 past cut-in

    rated = floris_reference["row3_270_15_rated"]["powers"]
    assert np.array_equal(rated[:2], np.full(2, 5.0e6))
    assert rated[2] < 5.0e6


def test_row3_270_8_zeroyaw_regression_anchor(
    floris_reference: dict[str, dict[str, npt.NDArray[np.float64]]],
) -> None:
    # Hardcoded FLORIS 4.6.6 output; guards both the solver and the live FLORIS
    # reference against silent drift (e.g. an upstream floris version bump).
    anchor_powers = np.array([1753954.45917917, 356384.89485703, 344414.69675443])
    anchor_ti = np.array([0.06021557, 0.10448007, 0.12594702])

    reference = floris_reference["row3_270_8_flat"]
    np.testing.assert_allclose(reference["powers"], anchor_powers, rtol=1e-6)
    np.testing.assert_allclose(
        reference["turbulence_intensities"], anchor_ti, rtol=1e-6
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
