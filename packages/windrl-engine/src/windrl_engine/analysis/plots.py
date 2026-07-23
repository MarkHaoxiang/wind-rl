from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.projections.polar import PolarAxes

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.turbine import D
from windrl_engine.farm.wind import WindRose


def plot_layout(
    layout: FarmLayout,
    *,
    ax: Axes | None = None,
    rotor_diameter: float = D,
    show_labels: bool = True,
) -> Axes:
    """Turbine positions (m) with rotor-diameter circles and index labels."""
    if ax is None:
        _, ax = plt.subplots()
    x = np.asarray(layout.x)
    y = np.asarray(layout.y)
    ax.scatter(x, y, s=10, color="tab:blue", zorder=3)
    for i, (turbine_x, turbine_y) in enumerate(zip(x, y, strict=True)):
        ax.add_patch(
            Circle(
                (turbine_x, turbine_y),
                radius=rotor_diameter / 2,
                fill=False,
                edgecolor="tab:blue",
            )
        )
        if show_labels:
            ax.annotate(
                str(i), (turbine_x, turbine_y), ha="center", va="center", fontsize=8
            )
    margin = rotor_diameter
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return ax


def plot_flow_slice(
    field: Float[Array, "ny nx"],
    extent: tuple[float, float, float, float],
    *,
    layout: FarmLayout | None = None,
    ax: Axes | None = None,
    cmap: str = "viridis",
) -> Axes:
    """Heatmap of a precomputed 2D velocity field (m/s) over `extent = (xmin, xmax, ymin, ymax)`."""
    if ax is None:
        _, ax = plt.subplots()
    image = ax.imshow(np.asarray(field), extent=extent, origin="lower", cmap=cmap)
    plt.colorbar(image, ax=ax, label="m/s")
    if layout is not None:
        ax.scatter(
            np.asarray(layout.x),
            np.asarray(layout.y),
            marker="o",
            color="white",
            edgecolors="black",
            zorder=3,
        )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return ax


def plot_wind_rose(rose: WindRose, *, ax: PolarAxes | None = None) -> PolarAxes:
    """Polar wind rose: direction = bearing wind comes FROM (0=N, clockwise), stacked by speed bin."""
    if ax is None:
        # subplots() is typed to return Axes even with subplot_kw={"projection": "polar"};
        # the runtime object is a PolarAxes.
        _, untyped_ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax = cast(PolarAxes, untyped_ax)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    direction_bins = np.asarray(rose.direction_bins)
    speed_bins = np.asarray(rose.speed_bins)
    frequency = np.asarray(rose.frequency)
    theta = np.deg2rad(direction_bins)
    width = (
        2.0 * np.pi / len(direction_bins)
    )  # rose direction bins assumed uniform over the circle
    bottom = np.zeros_like(direction_bins)
    cmap = plt.get_cmap("viridis")
    for speed_index, speed in enumerate(speed_bins):
        ax.bar(
            theta,
            frequency[:, speed_index],
            width=width,
            bottom=bottom,
            color=cmap(speed_index / max(len(speed_bins) - 1, 1)),
            label=f"{speed:.1f} m/s",
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += frequency[:, speed_index]
    ax.legend(title="wind speed", loc="upper left", bbox_to_anchor=(1.05, 1.0))
    return ax


def plot_power_by_direction(
    rose: WindRose,
    powers: Float[Array, "directions speeds turbines"],
    *,
    ax: Axes | None = None,
) -> Axes:
    """Farm power (MW) vs direction, one line per rose speed bin."""
    if ax is None:
        _, ax = plt.subplots()
    direction_bins = np.asarray(rose.direction_bins)
    speed_bins = np.asarray(rose.speed_bins)
    farm_power_mw = np.asarray(powers).sum(axis=-1) / 1e6
    for speed_index, speed in enumerate(speed_bins):
        ax.plot(
            direction_bins,
            farm_power_mw[:, speed_index],
            marker="o",
            label=f"{speed:.1f} m/s",
        )
    ax.set_xlabel("wind direction (deg)")
    ax.set_ylabel("farm power (MW)")
    ax.legend(title="wind speed")
    return ax
