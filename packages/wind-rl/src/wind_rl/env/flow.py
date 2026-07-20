"""Wake-resolved hub-height flow extraction from a live FLORIS interface.

This module owns the coordinate-frame conventions shared by the wind-farm
renderer (:mod:`wind_rl.env.render`) and the replay recorder
(:mod:`wind_rl.viz.trajectory`); it is the single place velocity frames are
reasoned about.

Angle conventions (from wfcrl/FLORIS, all degrees):

* ``wind_dir`` is meteorological -- the compass bearing the wind blows *from*.
  The air therefore flows *towards* the unit vector ``(-sin phi, -cos phi)`` in
  map ``(x, y)`` coordinates (``phi = 270`` => wind from the west, flowing
  +x/east).
* ``yaw`` is relative to the inflow: a turbine at zero yaw faces straight into
  the wind (nacelle axis along the upwind unit ``(sin phi, cos phi)``). Positive
  yaw rotates the nacelle counter-clockwise from that upwind direction (FLORIS
  sign), which deflects the wake sideways.

Velocity frame (the subtle one). FLORIS samples the cut-plane on a grid it
rotates so the inflow is always along +x. ``calculate_horizontal_plane`` returns
the point coordinates ``x1, x2`` in the inertial *map* frame
(``x/y_sorted_inertial_frame``) but the velocity components ``u, v`` in that
*wind-aligned* frame (``u_sorted``/``v_sorted``, which FLORIS never rotates
back) -- so ``(u, v)`` are streamwise/spanwise, not east/north. To draw them
over the map they must be rotated by the inverse of FLORIS's rel-west rotation
``theta = wind_dir - 270`` (``floris.utilities.rotate_coordinates_rel_west``);
:func:`sample_plane` does this once, so every consumer gets map-frame ``(u, v)``.
At ``wind_dir = 270`` the rotation is the identity, which is why treating the
raw components as map-frame stayed invisible until off-axis wind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from wind_rl.env.windfarm import DesignableWindFarmEnv

FlowFill = Literal["nan", "ambient"]


class _PlaneColumn(Protocol):
    @property
    def values(self) -> NDArray[np.float64]: ...


class _PlaneFrame(Protocol):
    x1: _PlaneColumn
    x2: _PlaneColumn
    u: _PlaneColumn
    v: _PlaneColumn


class _CutPlane(Protocol):
    df: _PlaneFrame


class FlorisPlaneSource(Protocol):
    """The slice of the live ``floris`` interface the flow sampler reads."""

    layout_x: NDArray[np.float64]
    layout_y: NDArray[np.float64]

    def calculate_horizontal_plane(
        self,
        *,
        height: float,
        x_resolution: int,
        y_resolution: int,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        yaw_angles: NDArray[np.float64],
    ) -> _CutPlane: ...


@dataclass(frozen=True)
class FarmState:
    layout: NDArray[np.float64]
    yaw: NDArray[np.float64]
    powers_mw: NDArray[np.float64]
    wind_speed: float
    wind_dir: float
    hub_height: float
    rotor_diameter: float


def read_farm_state(designable: DesignableWindFarmEnv) -> FarmState:
    """Snapshot the live farm telemetry from a designable env's FLORIS interface."""
    fi = designable.floris
    interface = designable.mdp.interface
    layout = np.column_stack([np.asarray(fi.layout_x), np.asarray(fi.layout_y)])
    n_turbines = layout.shape[0]
    return FarmState(
        layout=layout.astype(np.float64),
        yaw=np.asarray(interface.get_yaw_command(), dtype=np.float64).reshape(
            n_turbines
        ),
        powers_mw=np.asarray(interface.avg_powers(), dtype=np.float64).reshape(
            n_turbines
        )
        / 1e6,
        wind_speed=float(np.asarray(interface.wind_speed).reshape(-1)[0]),
        wind_dir=float(np.asarray(interface.wind_dir).reshape(-1)[0]),
        hub_height=float(np.asarray(fi.floris.farm.hub_heights).reshape(-1)[0]),
        rotor_diameter=float(np.asarray(fi.floris.farm.rotor_diameters).reshape(-1)[0]),
    )


def sample_plane(
    fi: FlorisPlaneSource,
    *,
    hub_height: float,
    bounds: tuple[float, float],
    yaw: NDArray[np.float64],
    wind_speed: float,
    wind_dir: float,
    resolution: int,
    fill: FlowFill,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Wake-resolved hub-height ``(grid_x, grid_y, u, v)`` on a regular map grid.

    ``u, v`` are in the map frame (see the module docstring). Rows of the grids
    index map ``y`` (ascending), columns map ``x``. FLORIS samples on a
    wind-aligned (for off-axis wind, rotated) grid, so the scattered velocity is
    interpolated onto the regular map grid; cells outside the sampled hull are
    filled per ``fill`` -- ``"nan"`` leaves them NaN, ``"ambient"`` fills with
    the free-stream velocity.
    """
    map_x, map_y = bounds
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, map_x, resolution), np.linspace(0.0, map_y, resolution)
    )
    plane = fi.calculate_horizontal_plane(
        height=hub_height,
        x_resolution=resolution,
        y_resolution=resolution,
        x_bounds=(0.0, map_x),
        y_bounds=(0.0, map_y),
        yaw_angles=yaw.reshape(1, 1, -1),
    )
    px = np.asarray(plane.df.x1.values, dtype=np.float64)
    py = np.asarray(plane.df.x2.values, dtype=np.float64)
    # Fill in the wind-aligned frame (free-stream is +x there) so the rotation
    # below carries the ambient fill into the map frame consistently.
    fill_u, fill_v = (wind_speed, 0.0) if fill == "ambient" else (np.nan, np.nan)
    u_wa = _interpolate_to_grid(px, py, plane.df.u.values, grid_x, grid_y, fill_u)
    v_wa = _interpolate_to_grid(px, py, plane.df.v.values, grid_x, grid_y, fill_v)
    u, v = _to_map_frame(u_wa, v_wa, wind_dir)
    return grid_x, grid_y, u, v


def ambient_uv(
    grid_x: NDArray[np.float64], wind_speed: float, wind_dir: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Uniform free-stream ``(u, v)`` over ``grid_x``'s shape, in the map frame."""
    phi = np.deg2rad(wind_dir)
    u = np.full_like(grid_x, -wind_speed * np.sin(phi))
    v = np.full_like(grid_x, -wind_speed * np.cos(phi))
    return u, v


def _to_map_frame(
    u: NDArray[np.float64], v: NDArray[np.float64], wind_dir: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Invert FLORIS's rel-west rotation (theta = wind_dir - 270); identity at 270.
    theta = np.deg2rad(wind_dir - 270.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return u * cos_t + v * sin_t, -u * sin_t + v * cos_t


def _interpolate_to_grid(
    px: NDArray[np.float64],
    py: NDArray[np.float64],
    values: NDArray[np.float64],
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
    fill: float,
) -> NDArray[np.float64]:
    from matplotlib.tri import LinearTriInterpolator, Triangulation

    tri = Triangulation(px, py)
    out = LinearTriInterpolator(tri, np.asarray(values, dtype=np.float64))(
        grid_x, grid_y
    )
    return np.asarray(np.ma.filled(out, fill), dtype=np.float64)
