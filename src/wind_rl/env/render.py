"""Matplotlib layout rendering to an RGB array (for wandb video / debugging).

Ported and simplified from DiCoDe's quiver render: turbines are drawn as points
with their min-distance exclusion circles, and (if a live ``state`` is supplied)
the free-stream wind and per-turbine yaw are drawn as quivers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
from numpy.typing import NDArray

from wind_rl.scenario import ScenarioConfig


def render_layout(
    layout: NDArray[Any],
    scenario: ScenarioConfig,
    *,
    state: Mapping[str, Any] | None = None,
) -> NDArray[np.uint8]:
    """Render a turbine layout to an ``(H, W, 3)`` uint8 RGB array.

    Parameters
    ----------
    layout:
        ``(N, 2)`` turbine xy coordinates.
    scenario:
        Provides the map extent and turbine spacing (exclusion-circle radius).
    state:
        Optional environment state; if it contains ``"wind_speed"``,
        ``"wind_direction"`` and ``"yaw"`` the wind and yaw quivers are drawn.
    """
    coords = np.asarray(layout, dtype=float).reshape(-1, 2)
    map_width = scenario.map_x_length
    map_height = scenario.map_y_length
    radius = scenario.min_distance_between_turbines / 2.0

    aspect = max(1.0, map_width / map_height)
    fig, ax = plt.subplots(figsize=(5.0 * aspect, 5.0))
    canvas = FigureCanvasAgg(fig)

    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, edgecolors="k", zorder=2)
    for x, y in coords:
        ax.add_patch(
            Circle(
                (x, y),
                radius=radius,
                edgecolor="green",
                facecolor="none",
                linewidth=1.2,
                zorder=1,
            )
        )

    if state is not None and {"wind_speed", "wind_direction"} <= set(state):
        wind_speed = np.asarray(state["wind_speed"], dtype=float) / 28.0 * radius
        wind_dir = np.deg2rad(np.asarray(state["wind_direction"], dtype=float))
        ax.quiver(
            coords[:, 0],
            coords[:, 1],
            -wind_speed * np.sin(wind_dir),
            -wind_speed * np.cos(wind_dir),
            color="blue",
            width=0.005,
        )
        if "yaw" in state:
            yaw = np.deg2rad(
                np.asarray(state["yaw"], dtype=float)
                + np.asarray(state["wind_direction"], dtype=float)
            )
            ax.quiver(
                coords[:, 0],
                coords[:, 1],
                -radius * np.sin(yaw),
                -radius * np.cos(yaw),
                color="red",
                width=0.005,
            )

    ax.grid(True)
    ax.set_xlim(0.0, map_width)
    ax.set_ylim(0.0, map_height)

    canvas.draw()  # type: ignore[no-untyped-call]
    width, height = canvas.get_width_height()
    buffer = canvas.buffer_rgba()  # type: ignore[no-untyped-call]
    image = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
    rgb = np.ascontiguousarray(image[:, :, :3])
    plt.close(fig)
    return rgb
