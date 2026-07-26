from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

#: Computes a farm's scalar reward from quantities the step already solved for
#: (no re-solving the wake): per-turbine power and load proxies, plus the
#: freestream speed used to normalize power. Must be jit/vmap-compatible.
RewardFn = Callable[
    [Float[Array, "turbines"], Float[Array, "turbines 4"], Float[Array, ""]],
    Float[Array, ""],
]


@dataclass(frozen=True, slots=True)
class WfcrlReward:
    """The WFCRL reward: mean normalized power minus ``load_coef`` times mean |load|."""

    load_coef: float

    def __call__(
        self,
        powers_watts: Float[Array, "turbines"],
        loads: Float[Array, "turbines 4"],
        freestream_speed: Float[Array, ""],
    ) -> Float[Array, ""]:
        powers_mw = powers_watts / 1e6
        normalized = powers_mw * 1e3 / freestream_speed**3
        load_penalty = jnp.mean(jnp.abs(loads))
        return jnp.mean(normalized) - self.load_coef * load_penalty
