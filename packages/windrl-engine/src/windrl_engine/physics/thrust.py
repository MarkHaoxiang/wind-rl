import jax.numpy as jnp

from windrl_engine.farm.turbine import ct_interp
from windrl_engine.physics.frame import Scalar, cosd


def cubic_mean(velocities: jnp.ndarray) -> jnp.ndarray:
    """Cube root of the mean cubed velocity over the rotor plane (last two axes)."""
    return jnp.cbrt(jnp.mean(velocities**3, axis=(-2, -1)))


def effective_ct(rotor_speed: Scalar, yaw: Scalar) -> Scalar:
    """C_t table lookup (cubic-mean speed) scaled by cos(yaw); tilt term is unity."""
    return ct_interp(rotor_speed) * cosd(yaw)


def axial_induction(ct_eff: Scalar, yaw: Scalar) -> Scalar:
    return 0.5 / cosd(yaw) * (1.0 - jnp.sqrt(1.0 - ct_eff * cosd(yaw)))
