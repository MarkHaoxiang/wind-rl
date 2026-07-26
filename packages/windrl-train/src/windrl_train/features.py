from typing import Final

import jax.numpy as jnp
from jaxtyping import Array, Float

from windrl_engine.env import Observation

NFEAT = 7

# Matches windrl_engine.env's Observation bounds: yaw in [-40, 40] deg,
# speed in [0, 28] m/s.
_YAW_LIMIT_DEG: Final = 40.0
_WIND_SPEED_MAX: Final = 28.0


def agent_features(obs: Observation) -> Float[Array, "envs agents 7"]:
    """The trainer-side per-agent feature vector built from a raw env ``Observation``."""
    wind_direction_rad = jnp.deg2rad(obs.wind_direction)
    freewind_direction_rad = jnp.deg2rad(obs.freewind[..., 1])

    freewind_speed = jnp.broadcast_to(obs.freewind[..., :1], obs.yaw.shape)
    freewind_sin = jnp.broadcast_to(
        jnp.sin(freewind_direction_rad)[..., None], obs.yaw.shape
    )
    freewind_cos = jnp.broadcast_to(
        jnp.cos(freewind_direction_rad)[..., None], obs.yaw.shape
    )

    return jnp.stack(
        [
            obs.yaw / _YAW_LIMIT_DEG,
            obs.wind_speed / _WIND_SPEED_MAX,
            jnp.sin(wind_direction_rad),
            jnp.cos(wind_direction_rad),
            freewind_speed / _WIND_SPEED_MAX,
            freewind_sin,
            freewind_cos,
        ],
        axis=-1,
    )
