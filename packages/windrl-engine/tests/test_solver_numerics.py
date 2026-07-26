"""Floating-point contracts of the wake solve: the rotor-plane rounding trick,
its ULP sensitivity, and the degenerate no-wake geometry both fidelities agree on."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from windrl_engine.farm.layout import (
    FarmLayout,
    ablaincourt,
    horns_rev2,
    row_layout,
    turb3_row1,
)
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.flow import AMBIENT_TI
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import Fidelity, rotor_plane_x, solve_farm


def _rotor_plane_x_samples() -> np.ndarray:  # type: ignore[type-arg]
    rng = np.random.default_rng(0)
    return np.concatenate(
        [
            rng.uniform(-1e4, 1e4, 20_000),
            rng.uniform(-1.0, 1.0, 20_000),
            np.asarray(turb3_row1().x),
            np.asarray(ablaincourt().x),
            np.asarray(ablaincourt().y),
            np.asarray(horns_rev2().x),
            np.asarray(horns_rev2().y),
            np.asarray(row_layout(10).x),
        ]
    )


def test_rotor_plane_x_is_the_floris_rotor_mean_clamped_up_to_x() -> None:
    # The docstring's "reproduces FLORIS's np.mean" is only half the contract:
    # np.mean of nine copies of x rounds *below* x on ~3.1% of inputs, and the
    # implementation clamps those back up. Pin the clamp, not just the mean.
    samples = _rotor_plane_x_samples()
    result = np.asarray(rotor_plane_x(jnp.asarray(samples)))
    floris_mean = np.asarray([np.mean(np.full(9, x)) for x in samples])

    np.testing.assert_array_equal(result, np.maximum(floris_mean, samples))
    assert np.all(result >= samples)
    assert np.mean(floris_mean < samples) > 0.01  # the clamp is load-bearing


def test_rotor_plane_x_is_bitwise_identical_under_jit_and_vmap() -> None:
    # The residual `e1` relies on two Sterbenz-exact subtractions; an FMA-fusing
    # or reassociating backend would silently flip the mean-rounding decision.
    samples = jnp.asarray(_rotor_plane_x_samples())
    eager = rotor_plane_x(samples)

    assert jnp.array_equal(jax.jit(rotor_plane_x)(samples), eager)
    assert jnp.array_equal(
        jax.vmap(rotor_plane_x)(samples[:, None]).ravel(),
        eager,
    )


def test_a_one_ulp_layout_nudge_moves_floris_fidelity_power_by_percent() -> None:
    # The whole reason fidelity="corrected" exists: under "floris", x_i is the
    # rotor-plane mean, whose one-ulp rounding decides the `delta_x >= 0`
    # transverse gate. Nudging a layout coordinate by a single ULP -- physically
    # nothing, ~1e-13 m -- flips that gate and moves turbine power by ~8%.
    layout = turb3_row1()
    nudged = FarmLayout(x=jnp.nextafter(layout.x, jnp.inf), y=layout.y)
    wind = WindCondition(speed=jnp.asarray(8.0), direction=jnp.asarray(270.0))
    yaw = jnp.asarray([20.0, -15.0, 10.0])

    def max_relative_power_shift(fidelity: Fidelity) -> float:
        baseline = turbine_powers(
            solve_farm(layout, wind, yaw, fidelity=fidelity).u, yaw
        )
        shifted = turbine_powers(
            solve_farm(nudged, wind, yaw, fidelity=fidelity).u, yaw
        )
        return float(jnp.max(jnp.abs(shifted - baseline) / baseline))

    assert max_relative_power_shift("floris") > 1e-3  # measured 7.8e-2
    assert max_relative_power_shift("corrected") < 1e-12  # measured 6.7e-16


def _perpendicular_column(num_turbines: int) -> FarmLayout:
    # All at the same wind-frame x under a 270 deg wind: nobody is downstream of
    # anybody, so no turbine may wake another.
    return FarmLayout(
        x=jnp.zeros(num_turbines),
        y=jnp.arange(num_turbines, dtype=jnp.float64) * 504.0,
    )


@pytest.mark.parametrize("fidelity", ["floris", "corrected"])
def test_a_column_perpendicular_to_the_wind_produces_isolated_turbine_power(
    fidelity: Fidelity,
) -> None:
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    column = _perpendicular_column(4)
    isolated = FarmLayout(x=jnp.zeros(1), y=jnp.zeros(1))

    powers = turbine_powers(
        solve_farm(column, wind, jnp.zeros(4), fidelity=fidelity).u, jnp.zeros(4)
    )
    reference = turbine_powers(
        solve_farm(isolated, wind, jnp.zeros(1), fidelity=fidelity).u, jnp.zeros(1)
    )

    assert jnp.array_equal(powers, jnp.full(4, reference[0]))


def test_a_column_perpendicular_to_the_wind_leaves_the_corrected_flow_untouched() -> (
    None
):
    # Only "corrected" is exact here. Under "floris" the rotor-mean x_i lets the
    # `delta_x >= 0` gate admit a turbine's own plane, so the column picks up
    # wake-rotation vortices (|v| ~ 0.23 m/s) and per-turbine TI noise even
    # though the streamwise deficit -- and hence the power -- is untouched.
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    solution = solve_farm(
        _perpendicular_column(4), wind, jnp.zeros(4), fidelity="corrected"
    )

    assert jnp.array_equal(solution.v, jnp.zeros_like(solution.v))
    assert jnp.array_equal(solution.w, jnp.zeros_like(solution.w))
    assert jnp.array_equal(solution.turbulence_intensity, jnp.full(4, AMBIENT_TI))
