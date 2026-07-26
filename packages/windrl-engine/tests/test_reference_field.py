"""Agreement of the query-point field pass with FLORIS and with our own farm solve.

The field pass re-casts a converged solution onto arbitrary points, the way
FLORIS's ``full_flow_sequential_solver`` does; these tests pin it to FLORIS's
own plane solve and to the rotor-grid velocities the reward is computed from.
"""

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from floris import FlorisModel

from windrl_engine.farm.layout import turb3_row1
from windrl_engine.farm.turbine import nrel5mw_v4
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.frame import rotate_to_wind_frame
from windrl_engine.physics.query_field import solve_query_points
from windrl_engine.physics.solver import solve_farm
from windrl_engine.viz.plane import horizontal_slice, vertical_slice

RTOL = 1e-9
AMBIENT_TI = 0.06
# FLORIS's plane API interprets its bounds in a frame that stops matching ours away
# from 270 deg (1.69 relative at 240 deg), so plane parity is asserted at 270 only.
DIRECTION = 270.0
SPEED = 8.0
YAW = [20.0, -15.0, 10.0]
X_BOUNDS = (-300.0, 900.0)
Y_BOUNDS = (-200.0, 200.0)
Z_BOUNDS = (30.0, 200.0)
RESOLUTION = (7, 5)


def _wind() -> WindCondition:
    return WindCondition(speed=jnp.asarray(SPEED), direction=jnp.asarray(DIRECTION))


@pytest.fixture(scope="module")
def floris_planes() -> dict[str, npt.NDArray[np.float64]]:
    layout = turb3_row1()
    fmodel = FlorisModel("defaults")
    fmodel.set(
        layout_x=np.asarray(layout.x),
        layout_y=np.asarray(layout.y),
        wind_directions=[DIRECTION],
        wind_speeds=[SPEED],
        turbulence_intensities=[AMBIENT_TI],
        yaw_angles=np.asarray([YAW]),
    )
    fmodel.run()
    nx, n_other = RESOLUTION
    horizontal = fmodel.calculate_horizontal_plane(
        height=nrel5mw_v4().hub_height,
        x_resolution=nx,
        y_resolution=n_other,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
    )
    vertical = fmodel.calculate_y_plane(
        crossstream_dist=0.0,
        x_resolution=nx,
        z_resolution=n_other,
        x_bounds=X_BOUNDS,
        z_bounds=Z_BOUNDS,
    )
    return {
        "horizontal": horizontal.df["u"].to_numpy().reshape(n_other, nx),
        "vertical": vertical.df["u"].to_numpy().reshape(n_other, nx),
    }


def test_horizontal_slice_matches_the_floris_horizontal_plane(
    floris_planes: dict[str, npt.NDArray[np.float64]],
) -> None:
    field, _ = horizontal_slice(
        turb3_row1(),
        _wind(),
        jnp.asarray(YAW),
        bounds=(*X_BOUNDS, *Y_BOUNDS),
        resolution=RESOLUTION,
    )
    np.testing.assert_allclose(
        np.asarray(field), floris_planes["horizontal"], rtol=RTOL
    )


def test_vertical_slice_matches_the_floris_y_plane(
    floris_planes: dict[str, npt.NDArray[np.float64]],
) -> None:
    field, _ = vertical_slice(
        turb3_row1(),
        _wind(),
        jnp.asarray(YAW),
        bounds=(*X_BOUNDS, *Z_BOUNDS),
        resolution=RESOLUTION,
    )
    np.testing.assert_allclose(np.asarray(field), floris_planes["vertical"], rtol=RTOL)


def test_corrected_field_at_the_rotor_centres_reproduces_the_farm_solve() -> None:
    # Under "floris" the two passes legitimately disagree by up to ~24% under yaw
    # (the field pass reuses converged v and the rotor-mean TI, as the reference
    # does); "corrected" removes both quirks, leaving only the 1-ULP rounding of
    # that rotor-plane TI mean.
    layout = turb3_row1()
    wind = _wind()
    yaw = jnp.asarray(YAW)
    turbine = nrel5mw_v4()

    x_rot, y_rot = rotate_to_wind_frame(layout.x, layout.y, wind.direction)
    hub = jnp.full_like(x_rot, turbine.hub_height)
    field = solve_query_points(
        layout,
        wind,
        yaw,
        x_rot[None, :],
        y_rot[None, :],
        hub[None, :],
        fidelity="corrected",
        turbine=turbine,
    )
    solution = solve_farm(layout, wind, yaw, fidelity="corrected", turbine=turbine)

    np.testing.assert_allclose(
        np.asarray(field[0]), np.asarray(solution.u[:, 1, 1]), rtol=1e-14
    )
