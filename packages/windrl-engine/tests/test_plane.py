"""viz/plane.py: horizontal and vertical u-velocity slices."""

import jax.numpy as jnp

from windrl_engine.farm.layout import row_layout
from windrl_engine.farm.turbine import DEFAULT_TURBINE
from windrl_engine.farm.wind import WindCondition
from windrl_engine.viz.plane import PAD_DIAMETERS, horizontal_slice, vertical_slice


def _wind() -> WindCondition:
    return WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))


def test_horizontal_slice_extent_matches_the_requested_bounds() -> None:
    bounds = (-252.0, 756.0, -50.0, 50.0)

    _, extent = horizontal_slice(
        row_layout(2), _wind(), jnp.zeros(2), bounds=bounds, resolution=(5, 3)
    )
    assert extent == bounds


def test_horizontal_slice_upstream_edge_is_freestream_at_hub_height() -> None:
    # At z=hub height, the shear profile u=ws*(z/HH)^0.12 collapses to ws
    # exactly, and a point well upstream of every turbine sees no deficit (the
    # deficit masks are gated on x downstream of the rotor).
    layout = row_layout(2)  # turbines at world x=0 and x=504
    wind = _wind()
    bounds = (-252.0, 756.0, -50.0, 50.0)  # x=-252 is 2 diameters upstream of x=0

    field, _ = horizontal_slice(
        layout, wind, jnp.zeros(2), bounds=bounds, resolution=(5, 3)
    )
    assert jnp.allclose(field[:, 0], wind.speed)


def test_horizontal_slice_shows_a_deficit_directly_downstream_of_a_turbine() -> None:
    wind = _wind()
    bounds = (-252.0, 756.0, -50.0, 50.0)

    field, _ = horizontal_slice(
        row_layout(2), wind, jnp.zeros(2), bounds=bounds, resolution=(5, 3)
    )
    # Grid x = [-252, 0, 252, 504, 756], y = [-50, 0, 50]; row index 1 (y=0)
    # aligns with the row, column index 2 (x=252) sits between the turbines,
    # directly downstream of turbine 0.
    assert float(field[1, 2]) < float(wind.speed) - 1e-3


def test_horizontal_slice_default_bounds_pad_the_layout_extent_by_pad_diameters() -> (
    None
):
    layout = row_layout(2)  # x in [0, 504], y in [0, 0]
    pad = PAD_DIAMETERS * DEFAULT_TURBINE.rotor_diameter

    _, extent = horizontal_slice(layout, _wind(), jnp.zeros(2), resolution=(3, 3))
    assert extent == (0.0 - pad, 504.0 + pad, 0.0 - pad, 0.0 + pad)


def test_horizontal_slice_default_height_is_the_turbine_hub_height() -> None:
    layout = row_layout(2)
    bounds = (-252.0, 756.0, -50.0, 50.0)
    default_height, _ = horizontal_slice(
        layout, _wind(), jnp.zeros(2), bounds=bounds, resolution=(5, 3)
    )
    at_hub, _ = horizontal_slice(
        layout,
        _wind(),
        jnp.zeros(2),
        height=DEFAULT_TURBINE.hub_height,
        bounds=bounds,
        resolution=(5, 3),
    )
    assert jnp.array_equal(default_height, at_hub)


def test_vertical_slice_default_bounds_span_three_hub_heights_above_ground() -> None:
    layout = row_layout(2)
    pad = PAD_DIAMETERS * DEFAULT_TURBINE.rotor_diameter

    _, extent = vertical_slice(layout, _wind(), jnp.zeros(2), resolution=(3, 3))
    assert extent == (0.0 - pad, 504.0 + pad, 1.0, 3.0 * DEFAULT_TURBINE.hub_height)


def test_vertical_slice_shows_a_deficit_directly_downstream_at_hub_height() -> None:
    wind = _wind()
    bounds = (-252.0, 756.0, 0.0, 180.0)  # z=90 (hub height) lands at row index 1

    field, _ = vertical_slice(
        row_layout(2), wind, jnp.zeros(2), bounds=bounds, resolution=(5, 3)
    )
    # Grid x = [-252, 0, 252, 504, 756]; column 0 is 2 diameters upstream of every
    # turbine, column 2 sits directly downstream of the turbine at x=0.
    assert jnp.allclose(field[1, 0], wind.speed)
    assert float(field[1, 2]) < float(wind.speed) - 1e-3
