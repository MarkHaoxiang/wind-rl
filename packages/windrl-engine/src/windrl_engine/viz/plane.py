"""Rectangular u-velocity slices through a farm: plot framing over the physics query."""

from typing import Final

import jax.numpy as jnp
from jaxtyping import Array, Float

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.frame import (
    layout_center,
    rotate_about,
    rotate_to_wind_frame,
    wind_deviation,
)
from windrl_engine.physics.query_field import QueryPlane, solve_query_points
from windrl_engine.physics.solver import Fidelity

PAD_DIAMETERS: Final = 3.0
GROUND_CLEARANCE: Final = 1.0
DOMAIN_HUB_HEIGHTS: Final = 3.0

Extent = tuple[float, float, float, float]


def padded_extent(layout: FarmLayout, rotor_diameter: float) -> Extent:
    """Layout bounding box (xmin, xmax, ymin, ymax) grown by ``PAD_DIAMETERS`` rotors."""
    pad = PAD_DIAMETERS * rotor_diameter
    return (
        float(jnp.min(layout.x)) - pad,
        float(jnp.max(layout.x)) + pad,
        float(jnp.min(layout.y)) - pad,
        float(jnp.max(layout.y)) + pad,
    )


def horizontal_slice(
    layout: FarmLayout,
    wind: WindCondition,
    yaw: Float[Array, "turbines"],
    *,
    height: float | None = None,
    bounds: Extent | None = None,
    resolution: tuple[int, int] = (200, 200),
    fidelity: Fidelity = "floris",
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> tuple[QueryPlane, Extent]:
    """u-velocity (m/s) on a z=height plane in WORLD coordinates; height defaults to hub."""
    if bounds is None:
        bounds = padded_extent(layout, turbine.rotor_diameter)
    xmin, xmax, ymin, ymax = bounds
    nx, ny = resolution

    xs = jnp.linspace(xmin, xmax, nx)
    ys = jnp.linspace(ymin, ymax, ny)
    x_world, y_world = jnp.meshgrid(xs, ys)  # (ny, nx)

    x_center, y_center = layout_center(layout.x, layout.y)
    x_query, y_query = rotate_about(
        x_world, y_world, wind_deviation(wind.direction), x_center, y_center
    )
    z_query = jnp.full_like(x_query, turbine.hub_height if height is None else height)

    field = solve_query_points(
        layout,
        wind,
        yaw,
        x_query,
        y_query,
        z_query,
        fidelity=fidelity,
        turbine=turbine,
    )
    return field, bounds


def vertical_slice(
    layout: FarmLayout,
    wind: WindCondition,
    yaw: Float[Array, "turbines"],
    *,
    y_offset: float = 0.0,
    bounds: Extent | None = None,
    resolution: tuple[int, int] = (200, 200),
    fidelity: Fidelity = "floris",
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> tuple[QueryPlane, Extent]:
    """u-velocity (m/s) on a streamwise x'-z plane in the WIND-ALIGNED frame at lateral y_offset."""
    x_rot, y_rot = rotate_to_wind_frame(layout.x, layout.y, wind.direction)
    _, y_center = layout_center(layout.x, layout.y)
    if bounds is None:
        xmin, xmax, _, _ = padded_extent(
            FarmLayout(x=x_rot, y=y_rot), turbine.rotor_diameter
        )
        bounds = (
            xmin,
            xmax,
            GROUND_CLEARANCE,
            DOMAIN_HUB_HEIGHTS * turbine.hub_height,
        )
    xmin, xmax, zmin, zmax = bounds
    nx, nz = resolution

    xs = jnp.linspace(xmin, xmax, nx)
    zs = jnp.linspace(zmin, zmax, nz)
    x_query, z_query = jnp.meshgrid(xs, zs)  # (nz, nx)
    y_query = jnp.full_like(x_query, float(y_center) + y_offset)

    field = solve_query_points(
        layout,
        wind,
        yaw,
        x_query,
        y_query,
        z_query,
        fidelity=fidelity,
        turbine=turbine,
    )
    return field, bounds
