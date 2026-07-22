from typing import Final

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from windrl_engine.farm.turbine import HUB_HEIGHT, D

GRID: Final = 3
DISC_AREA_RADIUS: Final = 0.5 * D / 2

Scalar = Float[Array, ""]
Turbines = Float[Array, "turbines"]
RotorField = Float[Array, "turbines grid grid"]
RotorPlane = Float[Array, "grid grid"]
Permutation = Int[Array, "turbines"]


def cosd(angle: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    return jnp.cos(jnp.deg2rad(angle))


def sind(angle: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    return jnp.sin(jnp.deg2rad(angle))


def wind_deviation(direction: Scalar) -> Scalar:
    return (direction - 270.0) % 360.0


def layout_center(x: Turbines, y: Turbines) -> tuple[Scalar, Scalar]:
    return (jnp.min(x) + jnp.max(x)) / 2, (jnp.min(y) + jnp.max(y)) / 2


def rotate_about(
    x: Float[Array, "*shape"],
    y: Float[Array, "*shape"],
    deviation: Scalar,
    xc: Scalar,
    yc: Scalar,
) -> tuple[Float[Array, "*shape"], Float[Array, "*shape"]]:
    dx = x - xc
    dy = y - yc
    x_rot = dx * cosd(deviation) - dy * sind(deviation) + xc
    y_rot = dx * sind(deviation) + dy * cosd(deviation) + yc
    return x_rot, y_rot


def rotate_to_wind_frame(
    x: Turbines, y: Turbines, direction: Scalar
) -> tuple[Turbines, Turbines]:
    """Rotate world coordinates so flow points in +x (270 deg = wind from west)."""
    xc, yc = layout_center(x, y)
    return rotate_about(x, y, wind_deviation(direction), xc, yc)


def rotor_grid(
    x_rot: Turbines, y_rot: Turbines
) -> tuple[RotorField, RotorField, RotorField]:
    """3x3 rotor grids: spanwise offset on axis -2, vertical (z=HH+offset) on axis -1."""
    disc = jnp.linspace(-DISC_AREA_RADIUS, DISC_AREA_RADIUS, GRID)
    x_grid = jnp.broadcast_to(x_rot[:, None, None], (x_rot.shape[0], GRID, GRID))
    y_grid = y_rot[:, None, None] + disc[None, :, None]
    z_grid = (HUB_HEIGHT + disc)[None, None, :] + jnp.zeros_like(x_grid)
    return x_grid, jnp.broadcast_to(y_grid, x_grid.shape), z_grid


def upstream_order(x_rot: Turbines) -> tuple[Permutation, Permutation]:
    """Sort permutation (upstream->downstream) and its inverse for unsorting."""
    sorted_indices = jnp.argsort(x_rot)
    unsorted_indices = jnp.argsort(sorted_indices)
    return sorted_indices, unsorted_indices
