from typing import Final

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.wind import WindCondition, WindRose
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import solve_farm

HOURS_PER_YEAR: Final = 8760.0


def _condition_power(
    layout: FarmLayout,
    direction: Float[Array, ""],
    speed: Float[Array, ""],
    yaw: Float[Array, "turbines"],
) -> Float[Array, "turbines"]:
    solution = solve_farm(layout, WindCondition(speed=speed, direction=direction), yaw)
    return turbine_powers(solution.u, yaw)


def power_surface(
    layout: FarmLayout, rose: WindRose, yaw: Float[Array, "turbines"]
) -> Float[Array, "directions speeds turbines"]:
    """Per-turbine power (W) at every rose (direction, speed) grid point."""
    over_speed = jax.vmap(_condition_power, in_axes=(None, None, 0, None))
    over_direction = jax.vmap(over_speed, in_axes=(None, 0, None, None))
    return over_direction(layout, rose.direction_bins, rose.speed_bins, yaw)


def aep(
    rose: WindRose, powers: Float[Array, "directions speeds turbines"]
) -> Float[Array, ""]:
    """Frequency-weighted annual energy production (GWh), assuming an 8760 h/yr calendar."""
    weights = rose.frequency / rose.frequency.sum()
    farm_power = jnp.sum(powers, axis=-1)
    expected_power = jnp.sum(weights * farm_power)
    return expected_power * HOURS_PER_YEAR / 1e9


def wake_loss(
    layout: FarmLayout, rose: WindRose, yaw: Float[Array, "turbines"]
) -> Float[Array, ""]:
    """Rose-weighted 1 - farm power / (N x isolated, unyawed single-turbine power)."""
    n_turbines = layout.x.shape[0]
    farm_power = jnp.sum(power_surface(layout, rose, yaw), axis=-1)
    isolated_layout = FarmLayout(
        x=jnp.zeros_like(layout.x[:1]), y=jnp.zeros_like(layout.y[:1])
    )
    isolated_yaw = jnp.zeros_like(yaw[:1])
    reference_power = (
        n_turbines * power_surface(isolated_layout, rose, isolated_yaw)[..., 0]
    )
    weights = rose.frequency / rose.frequency.sum()
    return jnp.sum(weights * (1.0 - farm_power / reference_power))
