from typing import Final

import jax.numpy as jnp
from jaxtyping import Array, Float

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.turbine import DEFAULT_TURBINE
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.frame import (
    layout_center,
    rotate_about,
    rotate_to_wind_frame,
    wind_deviation,
)
from windrl_engine.physics.query_field import QueryPlane, solve_query_points

PAD_DIAMETERS: Final = 3.0

Extent = tuple[float, float, float, float]


def _padded_bounds(low: Array, high: Array) -> tuple[float, float]:
    pad = PAD_DIAMETERS * DEFAULT_TURBINE.rotor_diameter
    return float(low) - pad, float(high) + pad


def horizontal_slice(
    layout: FarmLayout,
    wind: WindCondition,
    yaw: Float[Array, "turbines"],
    *,
    height: float = 90.0,
    bounds: Extent | None = None,
    resolution: tuple[int, int] = (200, 200),
) -> tuple[QueryPlane, Extent]:
    """u-velocity (m/s) on a z=height plane in WORLD coordinates; composes with plot_flow_slice."""
    if bounds is None:
        xmin, xmax = _padded_bounds(jnp.min(layout.x), jnp.max(layout.x))
        ymin, ymax = _padded_bounds(jnp.min(layout.y), jnp.max(layout.y))
    else:
        xmin, xmax, ymin, ymax = bounds
    nx, ny = resolution

    xs = jnp.linspace(xmin, xmax, nx)
    ys = jnp.linspace(ymin, ymax, ny)
    x_world, y_world = jnp.meshgrid(xs, ys)  # (ny, nx)

    x_center, y_center = layout_center(layout.x, layout.y)
    x_query, y_query = rotate_about(
        x_world, y_world, wind_deviation(wind.direction), x_center, y_center
    )
    z_query = jnp.full_like(x_query, height)

    field = solve_query_points(layout, wind, yaw, x_query, y_query, z_query)
    return field, (xmin, xmax, ymin, ymax)


def vertical_slice(
    layout: FarmLayout,
    wind: WindCondition,
    yaw: Float[Array, "turbines"],
    *,
    y_offset: float = 0.0,
    bounds: Extent | None = None,
    resolution: tuple[int, int] = (200, 200),
) -> tuple[QueryPlane, Extent]:
    """u-velocity (m/s) on a streamwise x'-z plane in the WIND-ALIGNED frame at lateral y_offset."""
    x_rot, _ = rotate_to_wind_frame(layout.x, layout.y, wind.direction)
    _, y_center = layout_center(layout.x, layout.y)
    if bounds is None:
        xmin, xmax = _padded_bounds(jnp.min(x_rot), jnp.max(x_rot))
        zmin, zmax = 1.0, 3.0 * 90.0
    else:
        xmin, xmax, zmin, zmax = bounds
    nx, nz = resolution

    xs = jnp.linspace(xmin, xmax, nx)
    zs = jnp.linspace(zmin, zmax, nz)
    x_query, z_query = jnp.meshgrid(xs, zs)  # (nz, nx)
    y_query = jnp.full_like(x_query, float(y_center) + y_offset)

    field = solve_query_points(layout, wind, yaw, x_query, y_query, z_query)
    return field, (xmin, xmax, zmin, zmax)
