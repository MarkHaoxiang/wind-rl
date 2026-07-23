from typing import Final

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.turbine import D
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.deficit import deficit_field
from windrl_engine.physics.deflection import deflection_field, wake_added_yaw
from windrl_engine.physics.flow import initial_flow
from windrl_engine.physics.frame import (
    layout_center,
    rotate_about,
    rotate_to_wind_frame,
    rotor_grid,
    upstream_order,
    wind_deviation,
)
from windrl_engine.physics.solver import rotor_plane_x, solve_farm
from windrl_engine.physics.thrust import axial_induction, cubic_mean, effective_ct

PAD_DIAMETERS: Final = 3.0

QueryPlane = Float[Array, "res_a res_b"]
Extent = tuple[float, float, float, float]


def _u_on_plane(
    layout: FarmLayout,
    wind: WindCondition,
    yaw: Float[Array, "turbines"],
    x_query: QueryPlane,
    y_query: QueryPlane,
    z_query: QueryPlane,
) -> QueryPlane:
    solution = solve_farm(layout, wind, yaw)

    x_rot, y_rot = rotate_to_wind_frame(layout.x, layout.y, wind.direction)
    x_grid, y_grid, z_grid = rotor_grid(x_rot, y_rot)
    sorted_idx, _ = upstream_order(x_rot)
    x_sorted = x_grid[sorted_idx]
    y_sorted = y_grid[sorted_idx]
    z_sorted = z_grid[sorted_idx]

    yaw_sorted = yaw[sorted_idx]
    u_turbine = solution.u[sorted_idx]
    v_turbine = solution.v[sorted_idx]
    ti_sorted = solution.turbulence_intensity[sorted_idx]

    u_initial_turbine, _ = initial_flow(z_sorted, wind.speed)
    u_inf_mean = jnp.mean(u_initial_turbine)
    x_i_all = rotor_plane_x(x_sorted[:, 0, 0])
    y_i_all = jnp.mean(y_sorted, axis=(1, 2))

    u_initial_query, _ = initial_flow(z_query, wind.speed)
    n_turbines = x_sorted.shape[0]

    def body(i: Array, wake_field: QueryPlane) -> QueryPlane:
        x_i = x_i_all[i]
        y_i = y_i_all[i]
        yaw_i = yaw_sorted[i]
        ti_i = ti_sorted[i]

        rotor_speed = cubic_mean(u_turbine[i])
        ct_i = effective_ct(rotor_speed, yaw_i)
        a_i = axial_induction(ct_i, yaw_i)

        added = wake_added_yaw(
            v_turbine[i],
            y_sorted[i] - y_i,
            z_sorted[i],
            u_inf_mean,
            rotor_speed,
            ct_i,
            a_i,
        )
        effective_yaw = yaw_i + added

        deflection = deflection_field(
            x_query, u_initial_query, x_i, effective_yaw, ti_i, ct_i
        )
        deficit = deficit_field(
            x_query,
            y_query,
            z_query,
            u_initial_query,
            deflection,
            x_i,
            y_i,
            ct_i,
            yaw_i,
            ti_i,
        )
        return jnp.hypot(wake_field, deficit * u_initial_query)

    wake_field: QueryPlane = jax.lax.fori_loop(
        0, n_turbines, body, jnp.zeros_like(x_query)
    )
    u: QueryPlane = u_initial_query - wake_field
    return u


def _padded_bounds(low: Array, high: Array) -> tuple[float, float]:
    pad = PAD_DIAMETERS * D
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

    field = _u_on_plane(layout, wind, yaw, x_query, y_query, z_query)
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

    field = _u_on_plane(layout, wind, yaw, x_query, y_query, z_query)
    return field, (xmin, xmax, zmin, zmax)
