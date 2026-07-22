from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.wind import WindCondition


class FarmState(NamedTuple):
    yaw: Float[Array, "turbines"]  # absolute deg, [-40, 40]
    yaw_accumulator: Float[Array, "turbines"]  # Σ|applied Δyaw| deg
    step_count: Int[Array, ""]
    wind: WindCondition
    key: Key[Array, ""]


def make_state(
    layout: FarmLayout, wind: WindCondition, key: Key[Array, ""]
) -> FarmState:
    zeros = jnp.zeros_like(layout.x)
    return FarmState(
        yaw=zeros,
        yaw_accumulator=zeros,
        step_count=jnp.asarray(0),
        wind=wind,
        key=key,
    )
