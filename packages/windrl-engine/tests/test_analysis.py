"""analysis/ (metrics + flow_viz), CI-safe (no wfcrl, no matplotlib import)."""

import jax.numpy as jnp

from windrl_engine.analysis.flow_viz import (
    PAD_DIAMETERS,
    horizontal_slice,
    vertical_slice,
)
from windrl_engine.analysis.metrics import HOURS_PER_YEAR, aep, power_surface, wake_loss
from windrl_engine.farm.layout import FarmLayout, row_layout
from windrl_engine.farm.turbine import D
from windrl_engine.farm.wind import WindCondition, WindRose, make_wind_rose


def _rose() -> WindRose:
    return make_wind_rose(
        jnp.asarray([260.0, 270.0, 280.0]), jnp.asarray([8.0, 10.0]), jnp.ones((3, 2))
    )


def test_power_surface_shape_is_directions_by_speeds_by_turbines() -> None:
    layout = row_layout(2)
    powers = power_surface(layout, _rose(), jnp.zeros(2))
    assert powers.shape == (3, 2, 2)


def test_make_wind_rose_normalizes_frequency_to_sum_to_one() -> None:
    rose = make_wind_rose(
        jnp.asarray([260.0, 270.0, 280.0]),
        jnp.asarray([8.0, 10.0]),
        jnp.asarray([[3.0, 1.0], [2.0, 4.0], [1.0, 1.0]]),  # sums to 12
    )
    assert float(jnp.sum(rose.frequency)) == 1.0


def test_aep_matches_hand_calc_for_a_single_turbine_delta_function_rose() -> None:
    # A one-bin ("delta function") rose puts all probability on one (dir, speed)
    # pair, so AEP must equal that single condition's power run out over a
    # full 8760h calendar, in GWh -- no wake, no rose-averaging to get wrong.
    layout = FarmLayout(x=jnp.asarray([0.0]), y=jnp.asarray([0.0]))
    yaw = jnp.zeros(1)
    rose = make_wind_rose(
        jnp.asarray([270.0]), jnp.asarray([10.0]), jnp.asarray([[1.0]])
    )

    powers = power_surface(layout, rose, yaw)
    result = aep(rose, powers)

    expected = float(powers[0, 0, 0]) * HOURS_PER_YEAR / 1e9
    assert float(result) == expected


def test_wake_loss_is_near_zero_for_an_isolated_single_turbine() -> None:
    # No upstream turbine exists, so farm power must equal the isolated
    # reference power exactly (up to floating-point noise), regardless of the
    # turbine's absolute world position (nothing else in the domain to
    # interact with).
    layout = FarmLayout(x=jnp.asarray([1234.0]), y=jnp.asarray([-567.0]))
    loss = wake_loss(layout, _rose(), jnp.zeros(1))
    assert abs(float(loss)) < 1e-9


def test_wake_loss_is_strictly_between_0_and_1_for_a_waked_row() -> None:
    # test_invariants.py already establishes that downstream row turbines are
    # strictly waked (power < isolated) at 270 deg, so the rose-weighted farm
    # power must fall strictly short of the isolated reference, but never to
    # or past total loss.
    layout = row_layout(3)
    loss = wake_loss(layout, _rose(), jnp.zeros(3))
    assert 0.0 < float(loss) < 1.0


def test_aep_is_invariant_to_an_unnormalized_rose_frequency() -> None:
    directions = jnp.asarray([260.0, 270.0, 280.0])
    speeds = jnp.asarray([8.0, 10.0])
    counts = jnp.asarray([[3.0, 1.0], [2.0, 4.0], [1.0, 1.0]])  # sums to 12, not 1
    unnormalized = WindRose(
        direction_bins=directions, speed_bins=speeds, frequency=counts
    )
    normalized = make_wind_rose(directions, speeds, counts)

    layout = row_layout(2)
    yaw = jnp.zeros(2)
    powers = power_surface(layout, unnormalized, yaw)

    assert float(aep(unnormalized, powers)) == float(aep(normalized, powers))


def test_wake_loss_is_invariant_to_an_unnormalized_rose_frequency() -> None:
    directions = jnp.asarray([260.0, 270.0, 280.0])
    speeds = jnp.asarray([8.0, 10.0])
    counts = jnp.asarray([[3.0, 1.0], [2.0, 4.0], [1.0, 1.0]])
    unnormalized = WindRose(
        direction_bins=directions, speed_bins=speeds, frequency=counts
    )
    normalized = make_wind_rose(directions, speeds, counts)

    layout = row_layout(2)
    yaw = jnp.zeros(2)

    assert float(wake_loss(layout, unnormalized, yaw)) == float(
        wake_loss(layout, normalized, yaw)
    )


# --- flow_viz.horizontal_slice -----------------------------------------------


def test_horizontal_slice_extent_matches_the_requested_bounds() -> None:
    layout = row_layout(2)
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    bounds = (-252.0, 756.0, -50.0, 50.0)

    _, extent = horizontal_slice(
        layout, wind, jnp.zeros(2), height=90.0, bounds=bounds, resolution=(5, 3)
    )
    assert extent == bounds


def test_horizontal_slice_upstream_edge_is_freestream_at_hub_height() -> None:
    # At z=hub height, the shear profile u=ws*(z/HH)^0.12 collapses to ws
    # exactly, and a point well upstream of every turbine sees no deficit (the
    # deficit masks are gated on x downstream of the rotor).
    layout = row_layout(2)  # turbines at world x=0 and x=504
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    bounds = (-252.0, 756.0, -50.0, 50.0)  # x=-252 is 2 diameters upstream of x=0

    field, _ = horizontal_slice(
        layout, wind, jnp.zeros(2), height=90.0, bounds=bounds, resolution=(5, 3)
    )
    upstream_column = field[:, 0]
    assert jnp.allclose(upstream_column, wind.speed)


def test_horizontal_slice_shows_a_deficit_directly_downstream_of_a_turbine() -> None:
    layout = row_layout(2)
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    bounds = (-252.0, 756.0, -50.0, 50.0)

    field, _ = horizontal_slice(
        layout, wind, jnp.zeros(2), height=90.0, bounds=bounds, resolution=(5, 3)
    )
    # Grid x = [-252, 0, 252, 504, 756], y = [-50, 0, 50]; row index 1 (y=0)
    # aligns with the row, column index 2 (x=252) sits between the turbines,
    # directly downstream of turbine 0.
    downstream_point = field[1, 2]
    assert float(downstream_point) < float(wind.speed) - 1e-3


def test_horizontal_slice_default_bounds_pad_the_layout_extent_by_pad_diameters() -> (
    None
):
    layout = row_layout(2)  # x in [0, 504], y in [0, 0]
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    pad = PAD_DIAMETERS * D

    _, extent = horizontal_slice(layout, wind, jnp.zeros(2), resolution=(3, 3))
    assert extent == (0.0 - pad, 504.0 + pad, 0.0 - pad, 0.0 + pad)


# --- flow_viz.vertical_slice --------------------------------------------------


def test_vertical_slice_default_bounds_pad_the_wind_aligned_x_extent_with_a_fixed_z_range() -> (
    None
):
    layout = row_layout(2)
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    pad = PAD_DIAMETERS * D

    _, extent = vertical_slice(layout, wind, jnp.zeros(2), resolution=(3, 3))
    assert extent == (0.0 - pad, 504.0 + pad, 1.0, 270.0)


def test_vertical_slice_shows_a_deficit_directly_downstream_of_a_turbine_at_hub_height() -> (
    None
):
    layout = row_layout(2)  # turbines at world x=0 and x=504
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    bounds = (-252.0, 756.0, 0.0, 180.0)  # z=90 (hub height) lands at row index 1

    field, _ = vertical_slice(
        layout, wind, jnp.zeros(2), bounds=bounds, resolution=(5, 3)
    )
    # Grid x = [-252, 0, 252, 504, 756]; column 0 is 2 diameters upstream of every
    # turbine, column 2 sits directly downstream of the turbine at x=0.
    upstream_point = field[1, 0]
    downstream_point = field[1, 2]
    assert jnp.allclose(upstream_point, wind.speed)
    assert float(downstream_point) < float(wind.speed) - 1e-3
