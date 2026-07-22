from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key


class WindCondition(NamedTuple):
    speed: Float[Array, ""]  # m/s freestream
    direction: Float[Array, ""]  # deg, 270 = wind from west


class WindRose(NamedTuple):
    direction_bins: Float[Array, "directions"]  # deg
    speed_bins: Float[Array, "speeds"]  # m/s
    frequency: Float[Array, "directions speeds"]


def sample_wind(key: Key[Array, ""]) -> WindCondition:
    """Reset-time wind draw (spec §3): 8·Weibull(8) speed, Normal(270, 20) direction."""
    speed_key, direction_key = jax.random.split(key)
    u = jax.random.uniform(speed_key, ())
    speed = jnp.clip(8.0 * (-jnp.log(u)) ** (1.0 / 8.0), 0.0, 28.0)
    direction = jnp.clip(
        (jax.random.normal(direction_key, ()) * 20.0 + 270.0) % 360.0, 0.0, 360.0
    )
    return WindCondition(speed=speed, direction=direction)


def make_wind_rose(
    direction_bins: Float[Array, "directions"],
    speed_bins: Float[Array, "speeds"],
    frequency: Float[Array, "directions speeds"],
) -> WindRose:
    return WindRose(direction_bins, speed_bins, frequency / frequency.sum())
