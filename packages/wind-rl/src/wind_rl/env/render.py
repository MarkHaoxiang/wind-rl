"""Wake-resolved wind-farm visualiser -> RGB array (for wandb / debugging).

The centrepiece is the *wake-resolved* horizontal flow field: the live FLORIS
interface owned by the env computes a hub-height cut-plane of the velocity field,
so the streaks of slowed air behind each turbine (and their deflection under yaw)
are visible -- exactly what wake-steering research needs to see. On top of the
flow field we draw the turbines coloured by power, their nacelle/yaw orientation,
index labels, and a wind indicator.

Angle conventions (derived from wfcrl/FLORIS, all degrees):

* ``wind_direction`` is meteorological -- the compass bearing the wind blows
  *from*. So the air flows *towards* the unit vector ``(-sin phi, -cos phi)`` in
  map ``(x, y)`` coordinates (phi = 270 => wind from the west, flowing +x/east).
* ``yaw`` is relative to the inflow: a turbine at zero yaw faces straight into the
  wind (nacelle axis along the upwind unit ``(sin phi, cos phi)``). Positive yaw
  rotates the nacelle counter-clockwise from that upwind direction (FLORIS sign),
  which is what deflects the wake sideways.

FLORIS returns the cut-plane velocity components ``u, v`` already in the inertial
map frame (verified: phi=270 gives ``u ~ +ws``, ``v ~ 0``), so they can be drawn
directly over the ``(x, y)`` grid with no extra rotation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray

from wind_rl.config import Config

if TYPE_CHECKING:
    from floris.tools import FlorisInterface as FlorisSimulator

    from wind_rl.env.windfarm import DesignableWindFarmEnv


class RenderConfig(Config):
    """Knobs for :func:`render_farm`. Defaults target < ~1s per eval frame."""

    dpi: int = 100
    figsize: float = 6.0
    # Cut-plane grid resolution (points per axis). 120 renders in well under a
    # second for a handful of turbines; raise for crisper wakes at some cost.
    flow_resolution: int = 120
    # Streamlines over the speed heatmap (default) vs. a coarse quiver.
    streamlines: bool = True
    # Draw the per-turbine minimum-distance exclusion circles.
    show_min_distance_circles: bool = False
    # Skip the (dominant) flow-field computation and draw only the layer.
    show_flow_field: bool = True


def render_farm(
    env: DesignableWindFarmEnv,
    *,
    config: RenderConfig | None = None,
) -> NDArray[np.uint8]:
    """Render ``env``'s live wake-resolved flow field to an ``(H, W, 3)`` uint8 array.

    Reads the live FLORIS interface the env owns (``env.floris``) for the flow
    plane, turbine powers, yaw command and free-stream wind; the map bounds come
    from ``env.scenario``. If the flow-plane computation fails the field falls
    back to a uniform free-stream quiver (still a real vector field).
    """
    cfg = config or RenderConfig()
    scenario = env.scenario
    if scenario is None:  # pragma: no cover - defensive; make_env always sets it
        raise ValueError("render_farm requires env.scenario to be set")

    fi = env.floris
    interface = env.mdp.interface
    layout = np.column_stack([np.asarray(fi.layout_x), np.asarray(fi.layout_y)])
    n_turbines = layout.shape[0]
    yaw = np.asarray(interface.get_yaw_command(), dtype=float).reshape(n_turbines)
    powers = np.asarray(interface.avg_powers(), dtype=float).reshape(n_turbines)
    wind_speed = float(np.asarray(interface.wind_speed).reshape(-1)[0])
    wind_dir = float(np.asarray(interface.wind_dir).reshape(-1)[0])
    hub_height = float(np.asarray(fi.floris.farm.hub_heights).reshape(-1)[0])
    rotor_diameter = float(np.asarray(fi.floris.farm.rotor_diameters).reshape(-1)[0])

    map_x = scenario.map_x_length
    map_y = scenario.map_y_length

    aspect = map_x / map_y
    fig, ax = plt.subplots(
        figsize=(cfg.figsize * max(1.0, aspect), cfg.figsize * max(1.0, 1.0 / aspect)),
        dpi=cfg.dpi,
    )
    canvas = FigureCanvasAgg(fig)

    _draw_flow_field(
        ax,
        fi,
        yaw=yaw,
        hub_height=hub_height,
        bounds=(map_x, map_y),
        wind_speed=wind_speed,
        wind_dir=wind_dir,
        cfg=cfg,
    )
    _draw_turbines(
        ax,
        fig,
        layout=layout,
        powers=powers,
        yaw=yaw,
        wind_dir=wind_dir,
        rotor_diameter=rotor_diameter,
        min_distance=scenario.min_distance_between_turbines,
        cfg=cfg,
    )
    _draw_wind_indicator(ax, wind_dir=wind_dir)

    ax.set_xlim(0.0, map_x)
    ax.set_ylim(0.0, map_y)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Farm power {powers.sum() / 1e6:.2f} MW   |   "
        f"wind {wind_speed:.1f} m/s @ {wind_dir:.0f}\N{DEGREE SIGN}"
    )

    fig.tight_layout()
    canvas.draw()  # type: ignore[no-untyped-call]
    width, height = canvas.get_width_height()
    buffer = canvas.buffer_rgba()  # type: ignore[no-untyped-call]
    image = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
    rgb = np.ascontiguousarray(image[:, :, :3])
    plt.close(fig)
    return rgb


def _draw_flow_field(
    ax: Axes,
    fi: FlorisSimulator,
    *,
    yaw: NDArray[np.float64],
    hub_height: float,
    bounds: tuple[float, float],
    wind_speed: float,
    wind_dir: float,
    cfg: RenderConfig,
) -> None:
    """Speed heatmap + streamlines of the wake-resolved hub-height plane.

    FLORIS samples the plane on a wind-aligned grid, so for off-axis wind the
    returned ``(x1, x2)`` map-frame points are a rotated (non-axis-aligned) grid.
    We interpolate the scattered ``(u, v)`` onto a regular map-frame grid (via
    ``matplotlib.tri``, dependency-free) so ``streamplot`` -- which needs a
    monotone grid -- works for every wind direction. Falls back to a uniform
    free-stream field if the cut-plane extraction raises.
    """
    map_x, map_y = bounds
    res = cfg.flow_resolution
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, map_x, res), np.linspace(0.0, map_y, res)
    )
    u, v = _uniform_uv(grid_x, wind_speed, wind_dir)
    if cfg.show_flow_field:
        try:
            plane = fi.calculate_horizontal_plane(
                height=hub_height,
                x_resolution=res,
                y_resolution=res,
                x_bounds=(0.0, map_x),
                y_bounds=(0.0, map_y),
                yaw_angles=yaw.reshape(1, 1, -1),
            )
            px = np.asarray(plane.df.x1.values, dtype=np.float64)
            py = np.asarray(plane.df.x2.values, dtype=np.float64)
            u = _interpolate_to_grid(px, py, plane.df.u.values, grid_x, grid_y)
            v = _interpolate_to_grid(px, py, plane.df.v.values, grid_x, grid_y)
        except Exception:  # pragma: no cover - fall back to ambient uniform field
            u, v = _uniform_uv(grid_x, wind_speed, wind_dir)

    speed = np.hypot(u, v)
    ax.pcolormesh(
        grid_x, grid_y, speed, cmap="viridis", shading="auto", zorder=0, alpha=0.9
    )
    # streamplot rejects NaN velocities; zero-fill the un-sampled (rotated-grid)
    # corners so lines simply stop there.
    su, sv = np.nan_to_num(u), np.nan_to_num(v)
    if cfg.streamlines:
        ax.streamplot(
            grid_x[0, :],
            grid_y[:, 0],
            su,
            sv,
            color="white",
            density=1.1,
            linewidth=0.6,
            arrowsize=0.7,
            zorder=1,
        )
    else:
        step = max(1, res // 24)
        ax.quiver(
            grid_x[::step, ::step],
            grid_y[::step, ::step],
            su[::step, ::step],
            sv[::step, ::step],
            color="white",
            zorder=1,
        )


def _interpolate_to_grid(
    px: NDArray[np.float64],
    py: NDArray[np.float64],
    values: NDArray[np.float64],
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Linearly interpolate a scattered field onto a regular grid.

    Points outside the sampled (rotated) hull become NaN.
    """
    from matplotlib.tri import LinearTriInterpolator, Triangulation

    tri = Triangulation(px, py)
    out = LinearTriInterpolator(tri, np.asarray(values, dtype=np.float64))(
        grid_x, grid_y
    )
    return np.asarray(np.ma.filled(out, np.nan), dtype=np.float64)


def _uniform_uv(
    grid_x: NDArray[np.float64], wind_speed: float, wind_dir: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A uniform free-stream ``(u, v)`` field (ambient-wind fallback)."""
    phi = np.deg2rad(wind_dir)
    u = np.full_like(grid_x, -wind_speed * np.sin(phi))
    v = np.full_like(grid_x, -wind_speed * np.cos(phi))
    return u, v


def _draw_turbines(
    ax: Axes,
    fig: Figure,
    *,
    layout: NDArray[np.float64],
    powers: NDArray[np.float64],
    yaw: NDArray[np.float64],
    wind_dir: float,
    rotor_diameter: float,
    min_distance: float,
    cfg: RenderConfig,
) -> None:
    """Turbine markers coloured by power, yaw-oriented nacelle lines, labels."""
    xs, ys = layout[:, 0], layout[:, 1]
    powers_mw = powers / 1e6

    # Nacelle axis: the upwind unit (sin phi, cos phi) turned by the yaw offset.
    facing = np.deg2rad(wind_dir - yaw)
    nacelle_dx = np.sin(facing)
    nacelle_dy = np.cos(facing)
    half = rotor_diameter  # visual length of the nacelle line
    ax.quiver(
        xs,
        ys,
        nacelle_dx,
        nacelle_dy,
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1.0 / half,
        width=0.006,
        zorder=4,
        label="nacelle (yaw)",
    )

    if cfg.show_min_distance_circles:
        for x, y in layout:
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=min_distance / 2.0,
                    edgecolor="white",
                    facecolor="none",
                    linewidth=0.8,
                    linestyle="--",
                    alpha=0.5,
                    zorder=2,
                )
            )

    scatter = ax.scatter(
        xs,
        ys,
        c=powers_mw,
        cmap="plasma",
        s=90,
        edgecolors="k",
        linewidths=1.0,
        zorder=5,
    )
    fig.colorbar(scatter, ax=ax, label="Turbine power [MW]", fraction=0.046, pad=0.04)
    for i, (x, y) in enumerate(layout):
        ax.annotate(
            str(i),
            (x, y),
            color="white",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )


def _draw_wind_indicator(ax: Axes, *, wind_dir: float) -> None:
    """A small arrow (top-left, axes fraction) showing the inflow direction."""
    phi = np.deg2rad(wind_dir)
    length = 0.09
    base_x, base_y = 0.08, 0.92
    tip = (base_x - length * float(np.sin(phi)), base_y - length * float(np.cos(phi)))
    ax.annotate(
        "",
        xy=tip,
        xytext=(base_x, base_y),
        xycoords="axes fraction",
        arrowprops={"facecolor": "white", "edgecolor": "black", "width": 3.0},
        zorder=7,
    )
    ax.text(
        0.08,
        0.985,
        "wind",
        transform=ax.transAxes,
        color="white",
        fontsize=8,
        ha="center",
        va="top",
        zorder=7,
    )
