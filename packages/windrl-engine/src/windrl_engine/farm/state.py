from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.wind import WindCondition


class FarmState(NamedTuple):
    yaw: Float[Array, "turbines"]  # absolute deg, [-40, 40]
    yaw_accumulator: Float[Array, "turbines"]  # Σ|applied Δyaw| deg
    step_count: Int[Array, ""]  # WFCRL _num_iter; reset burn-in solve is step 1
    wind: WindCondition
    key: Key[Array, ""]


def make_state(
    layout: FarmLayout, wind: WindCondition, key: Key[Array, ""]
) -> FarmState:
    zeros = jnp.zeros_like(layout.x)
    return FarmState(
        yaw=zeros,
        yaw_accumulator=zeros,
        # WFCRL's reset advances _num_iter 0->1 via a zero-yaw burn-in solve, so the
        # reset-produced state already counts as step 1 (spec §8); horizon-1 agent
        # steps follow before truncation.
        step_count=jnp.asarray(1),
        wind=wind,
        key=key,
    )
