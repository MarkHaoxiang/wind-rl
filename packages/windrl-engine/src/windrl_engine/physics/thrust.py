import jax.numpy as jnp
from jaxtyping import Array, Float

from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec, ct_lookup
from windrl_engine.physics.frame import Scalar, cosd


def cubic_mean(
    velocities: Float[Array, "*batch grid grid"],
) -> Float[Array, "*batch"]:
    """Cube root of the mean cubed velocity over a rotor plane."""
    return jnp.cbrt(jnp.mean(velocities**3, axis=(-2, -1)))


def effective_ct(
    rotor_speed: Scalar, yaw: Scalar, *, turbine: TurbineSpec = DEFAULT_TURBINE
) -> Scalar:
    """C_t table lookup (cubic-mean speed) scaled by cos(yaw); tilt term is unity."""
    return ct_lookup(turbine, rotor_speed) * cosd(yaw)


def axial_induction(ct_eff: Scalar, yaw: Scalar) -> Scalar:
    return 0.5 / cosd(yaw) * (1.0 - jnp.sqrt(1.0 - ct_eff * cosd(yaw)))
