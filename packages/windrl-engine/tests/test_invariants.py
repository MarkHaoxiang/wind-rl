"""Physics invariants of the wake solve, asserted at both fidelities."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from windrl_engine.farm.layout import ROW_SPACING, FarmLayout, horns_rev2, row_layout
from windrl_engine.farm.turbine import DEFAULT_TURBINE, POWER_SCALE
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import Fidelity, solve_farm

FIDELITIES: list[Fidelity] = ["floris", "corrected"]

# Rotor grid z-offsets: disc_grid = linspace(-D/4, D/4, 3) for the solver's
# turbine, centered on hub height (not tower base) -- z_grid = HH + disc.
_HUB_HEIGHT = DEFAULT_TURBINE.hub_height
_GRID_Z = _HUB_HEIGHT + jnp.asarray([-0.25, 0.0, 0.25]) * DEFAULT_TURBINE.rotor_diameter
RATED_POWER_W = float(np.max(DEFAULT_TURBINE.power_table) * POWER_SCALE)


def _shear_ceiling(speed: jax.Array) -> jax.Array:
    return speed * (_GRID_Z / _HUB_HEIGHT) ** 0.12


@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_solve_farm_is_equivariant_to_turbine_permutation(fidelity: Fidelity) -> None:
    layout = row_layout(4)
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    yaw = jnp.asarray([5.0, -3.0, 0.0, 8.0])
    perm = jnp.asarray([2, 0, 3, 1])

    baseline = solve_farm(layout, wind, yaw, fidelity=fidelity)
    permuted_layout = FarmLayout(x=layout.x[perm], y=layout.y[perm])
    permuted = solve_farm(permuted_layout, wind, yaw[perm], fidelity=fidelity)

    assert jnp.allclose(permuted.u, baseline.u[perm], atol=1e-9)
    assert jnp.allclose(permuted.v, baseline.v[perm], atol=1e-9)
    assert jnp.allclose(permuted.w, baseline.w[perm], atol=1e-9)
    assert jnp.allclose(
        permuted.turbulence_intensity, baseline.turbulence_intensity[perm], atol=1e-9
    )


@pytest.mark.parametrize(
    ("fidelity", "spacing"),
    [
        # Rotation invariance is exact under "corrected" at any spacing, but under
        # "floris" it holds only where the rotor-mean x_i happens to round the same
        # way in both frames -- true at the nominal 504 m, false at a true 4D
        # 503.52 m (2.5e-2 relative). Keep the floris case on the geometry the rest
        # of the suite uses; the extra corrected case pins the spacing-independence.
        ("floris", ROW_SPACING),
        ("corrected", ROW_SPACING),
        ("corrected", 4.0 * DEFAULT_TURBINE.rotor_diameter),
    ],
)
def test_solve_farm_is_rotation_invariant_with_matched_wind_direction(
    fidelity: Fidelity, spacing: float
) -> None:
    layout = row_layout(4, spacing=spacing)
    yaw = jnp.asarray([0.0, 4.0, -6.0, 2.0])
    wind = WindCondition(speed=jnp.asarray(10.0), direction=jnp.asarray(270.0))

    theta = 37.0
    x_c = (layout.x.min() + layout.x.max()) / 2.0
    y_c = (layout.y.min() + layout.y.max()) / 2.0
    dx, dy = layout.x - x_c, layout.y - y_c
    rad = jnp.deg2rad(theta)
    rotated_layout = FarmLayout(
        x=dx * jnp.cos(rad) - dy * jnp.sin(rad) + x_c,
        y=dx * jnp.sin(rad) + dy * jnp.cos(rad) + y_c,
    )
    # Meteorological from-azimuth is clockwise-positive; the planar (x,y)
    # rotation above is counterclockwise-positive. A CCW layout rotation by
    # +theta must therefore pair with a CW wind-direction rotation, i.e.
    # subtract theta from the from-azimuth to keep the flow geometry matched.
    rotated_wind = WindCondition(
        speed=wind.speed, direction=(wind.direction - theta) % 360.0
    )

    baseline = solve_farm(layout, wind, yaw, fidelity=fidelity)
    rotated = solve_farm(rotated_layout, rotated_wind, yaw, fidelity=fidelity)

    baseline_power = turbine_powers(baseline.u, yaw)
    rotated_power = turbine_powers(rotated.u, yaw)
    assert jnp.allclose(rotated_power, baseline_power, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_solve_farm_vmapped_over_conditions_matches_individual_solves(
    fidelity: Fidelity,
) -> None:
    layout = row_layout(3)
    yaw = jnp.zeros(3)
    speeds = jnp.asarray([6.0, 9.0, 12.0])
    directions = jnp.asarray([260.0, 270.0, 280.0])
    conditions = WindCondition(speed=speeds, direction=directions)

    solve = functools.partial(solve_farm, fidelity=fidelity)
    batched = jax.vmap(solve, in_axes=(None, 0, None))(layout, conditions, yaw)

    for i in range(3):
        single = solve(
            layout, WindCondition(speed=speeds[i], direction=directions[i]), yaw
        )
        assert jnp.allclose(batched.u[i], single.u, atol=1e-9)
        assert jnp.allclose(batched.v[i], single.v, atol=1e-9)
        assert jnp.allclose(batched.w[i], single.w, atol=1e-9)
        assert jnp.allclose(
            batched.turbulence_intensity[i], single.turbulence_intensity, atol=1e-9
        )


@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_solve_farm_wakes_every_turbine_behind_the_unwaked_front_of_the_row(
    fidelity: Fidelity,
) -> None:
    # 270 deg = wind from the west, flowing in +x; row_layout is aligned along
    # +x with y=0, so turbines 1..3 all sit directly downstream of turbine 0.
    layout = row_layout(4)
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    yaw = jnp.zeros(4)

    solution = solve_farm(layout, wind, yaw, fidelity=fidelity)
    ceiling = _shear_ceiling(
        wind.speed
    )  # shape (grid,), broadcasts over (turbines, grid)

    assert bool(jnp.all(solution.u > 0.0))
    assert bool(jnp.all(solution.u <= ceiling[None, None, :] + 1e-9))
    # Nothing is upstream of turbine 0, so its inflow is the sheared freestream
    # bit for bit -- no wake contribution may round its way in.
    assert bool(jnp.all(solution.u[0] == ceiling[None, :]))
    assert bool(jnp.all(solution.u[1:] < ceiling[None, None, :] - 1e-6))


def test_turbine_powers_never_exceed_rated_and_front_turbine_dominates_the_row() -> (
    None
):
    # GCH wake-added turbulence mixing lets deep-row turbines partially recover
    # (row power need not be monotone non-increasing), so the sharp invariants
    # here are: nothing exceeds the rated ceiling, the unwaked front turbine is
    # the row's strict maximum, and turbine 2 -- compounded (SOSFS) deficit
    # from both upstream wakes, before turbulence-mixing recovery has built up
    # -- is the row's strict minimum (recovery is then visible at turbines 3-4).
    layout = row_layout(5)
    wind = WindCondition(speed=jnp.asarray(11.0), direction=jnp.asarray(270.0))
    yaw = jnp.zeros(5)

    solution = solve_farm(layout, wind, yaw)
    powers = turbine_powers(solution.u, yaw)

    assert bool(jnp.all(powers <= RATED_POWER_W))
    assert bool(jnp.all(powers[0] > powers[1:]))  # front turbine: strict row max
    assert bool(
        jnp.all(powers[2] < powers[jnp.asarray([0, 1, 3, 4])])
    )  # strict row min


@pytest.mark.slow
def test_solve_farm_stays_bounded_on_the_full_91_turbine_horns_rev2_site() -> None:
    # The 3-7 turbine cases never exercise a deep interior: at 91 turbines the
    # SOSFS sum compounds ~13 rows of wakes, which is where a sign slip or an
    # unguarded division would first surface as a NaN or a negative velocity.
    layout = horns_rev2()
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    yaw = jnp.zeros(91)

    solution = solve_farm(layout, wind, yaw)
    powers = turbine_powers(solution.u, yaw)
    ceiling = _shear_ceiling(wind.speed)

    assert bool(jnp.all(jnp.isfinite(solution.u)))
    assert bool(jnp.all(solution.u > 0.0))
    assert bool(jnp.all(solution.u <= ceiling[None, None, :] + 1e-9))
    assert bool(jnp.all(solution.turbulence_intensity >= 0.0))
    assert bool(jnp.all(powers > 0.0))
    assert bool(jnp.all(powers <= RATED_POWER_W))
