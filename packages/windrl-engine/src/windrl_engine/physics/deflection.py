import math
from typing import Final

import jax.numpy as jnp

from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec
from windrl_engine.physics.flow import SHEAR
from windrl_engine.physics.frame import (
    QueryField,
    RotorPlane,
    Scalar,
    TurbineTI,
    cosd,
)

ALPHA: Final = 0.58
BETA: Final = 0.077
KA: Final = 0.38
KB: Final = 0.004
DM: Final = 1.0
# floris 4.6.6 hardcodes eps_gain = 0.2 in both wake_added_yaw and
# calculate_transverse_velocity, ignoring GaussVelocityDeflection.eps_gain.
EPS_GAIN: Final = 0.2
# BaseModel.NUM_EPS, floris 4.6.6: divide-by-zero guard, no yaml key.
NUM_EPS: Final = 0.001


def deflection_field(
    x: QueryField,
    freestream_velocity: QueryField,
    x_i: Scalar,
    yaw_i: Scalar,
    ti_i: TurbineTI,
    ct_i: Scalar,
    *,
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> QueryField:
    """Lateral wake-center deflection (m) of turbine `i`, opposite yaw sign convention."""
    D = turbine.rotor_diameter
    yaw = -1.0 * yaw_i

    uR = (
        freestream_velocity
        * ct_i
        * cosd(yaw)
        / (2.0 * (1.0 - jnp.sqrt(1.0 - ct_i * cosd(yaw))))
    )
    u0 = freestream_velocity * jnp.sqrt(1.0 - ct_i)

    x0 = (
        D
        * (cosd(yaw) * (1.0 + jnp.sqrt(1.0 - ct_i * cosd(yaw))))
        / (
            jnp.sqrt(2.0)
            * (4.0 * ALPHA * ti_i + 2.0 * BETA * (1.0 - jnp.sqrt(1.0 - ct_i)))
        )
        + x_i
    )
    ky = KA * ti_i + KB
    kz = KA * ti_i + KB

    C0 = 1.0 - u0 / freestream_velocity
    M0 = C0 * (2.0 - C0)
    E0 = C0**2 - 3.0 * math.exp(1.0 / 12.0) * C0 + 3.0 * math.exp(1.0 / 3.0)

    sigma_z0 = D * 0.5 * jnp.sqrt(uR / (freestream_velocity + u0))
    sigma_y0 = sigma_z0 * cosd(yaw)

    xR = x_i
    theta_c0 = (
        DM
        * (0.3 * jnp.deg2rad(yaw) / cosd(yaw))
        * (1.0 - jnp.sqrt(1.0 - ct_i * cosd(yaw)))
    )
    delta0 = jnp.tan(theta_c0) * (x0 - x_i)

    delta_near = ((x - xR) / (x0 - xR)) * delta0
    delta_near = delta_near * ((x >= xR) & (x <= x0))

    sigma_y = ky * (x - x0) + sigma_y0
    sigma_z = kz * (x - x0) + sigma_z0
    sigma_y = sigma_y * (x >= x0) + sigma_y0 * (x < x0)
    sigma_z = sigma_z * (x >= x0) + sigma_z0 * (x < x0)

    m0_sqrt = jnp.sqrt(M0)
    middle = jnp.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0))
    ln_num = (1.6 + m0_sqrt) * (1.6 * middle - m0_sqrt)
    ln_den = (1.6 - m0_sqrt) * (1.6 * middle + m0_sqrt)

    delta_far = delta0 + theta_c0 * E0 / 5.2 * jnp.sqrt(
        sigma_y0 * sigma_z0 / (ky * kz * M0)
    ) * jnp.log(ln_num / ln_den)
    delta_far = delta_far * (x > x0)

    return delta_near + delta_far


def wake_added_yaw(
    v_i: RotorPlane,
    delta_y_i: RotorPlane,
    z_i: RotorPlane,
    freestream_velocity: Scalar,
    rotor_speed_i: Scalar,
    ct_i: Scalar,
    a_i: Scalar,
    *,
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> Scalar:
    """Secondary-steering effective yaw (deg) that reproduces the induced spanwise velocity."""
    D = turbine.rotor_diameter
    hub_height = turbine.hub_height
    tip_speed_ratio = turbine.tsr
    eps = EPS_GAIN * D

    vel_top = ((hub_height + D / 2) / hub_height) ** SHEAR
    vel_bottom = ((hub_height - D / 2) / hub_height) ** SHEAR
    gamma_top = (math.pi / 8) * D * vel_top * freestream_velocity * ct_i
    gamma_bottom = -1.0 * (math.pi / 8) * D * vel_bottom * freestream_velocity * ct_i
    gamma_wake_rotation = (
        0.25 * 2 * math.pi * D * (a_i - a_i**2) * rotor_speed_i / tip_speed_ratio
    )

    y_locs = delta_y_i + NUM_EPS

    z_top = z_i - (hub_height + D / 2) + NUM_EPS
    r_top = y_locs**2 + z_top**2
    core_top = 1.0 - jnp.exp(-r_top / eps**2)
    v_top = jnp.mean(gamma_top * z_top / (2 * math.pi * r_top) * core_top)

    z_bottom = z_i - (hub_height - D / 2) + NUM_EPS
    r_bottom = y_locs**2 + z_bottom**2
    core_bottom = 1.0 - jnp.exp(-r_bottom / eps**2)
    v_bottom = jnp.mean(
        gamma_bottom * z_bottom / (2 * math.pi * r_bottom) * core_bottom
    )

    z_core = z_i - hub_height + NUM_EPS
    r_core = y_locs**2 + z_core**2
    core_core = 1.0 - jnp.exp(-r_core / eps**2)
    v_core = jnp.mean(gamma_wake_rotation * z_core / (2 * math.pi * r_core) * core_core)

    avg_v = jnp.mean(v_i)
    val = jnp.clip(2 * (avg_v - v_core) / (v_top + v_bottom), -1.0, 1.0)
    return jnp.rad2deg(0.5 * jnp.arcsin(val))
