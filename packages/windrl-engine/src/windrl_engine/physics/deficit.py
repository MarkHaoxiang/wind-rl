from typing import Final

import jax.numpy as jnp

from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec
from windrl_engine.physics.frame import QueryField, Scalar, TurbineTI, cosd

# FLORIS tunes the velocity and deflection Gauss models through two independent
# parameter sets (wake_velocity_parameters.gauss / wake_deflection_parameters.gauss);
# they merely happen to ship the same numbers, so neither may be sourced from the other.
ALPHA: Final = 0.58
BETA: Final = 0.077
KA: Final = 0.38
KB: Final = 0.004


def _gaussian_deficit_terms(
    sigma_y: QueryField,
    sigma_z: QueryField,
    y: QueryField,
    y_i: Scalar,
    deflection: QueryField,
    z: QueryField,
    ct_i: Scalar,
    yaw: Scalar,
    turbine: TurbineSpec,
) -> tuple[QueryField, QueryField]:
    # FLORIS's rC() weights a and c by cos^2/sin^2(wind_veer) and carries a
    # -2b*dy*dz cross term; all of it collapses to this at the default wind_veer = 0.
    a = 1.0 / (2.0 * sigma_y**2)
    c = 1.0 / (2.0 * sigma_z**2)
    r_squared = a * (y - y_i - deflection) ** 2 + c * (z - turbine.hub_height) ** 2
    d = jnp.clip(
        1.0 - ct_i * cosd(yaw) / (8.0 * sigma_y * sigma_z / turbine.rotor_diameter**2),
        0.0,
        1.0,
    )
    # (r_squared, C) of FLORIS's rC()
    return r_squared, 1.0 - jnp.sqrt(d)


def sosfs_combine(wake_field: QueryField, velocity_deficit: QueryField) -> QueryField:
    """Sum-of-squares-freestream superposition of one more deficit (m/s) onto a wake field."""
    return jnp.hypot(wake_field, velocity_deficit)


def deficit_field(
    x: QueryField,
    y: QueryField,
    z: QueryField,
    u_initial: QueryField,
    deflection: QueryField,
    x_i: Scalar,
    y_i: Scalar,
    ct_i: Scalar,
    yaw_i: Scalar,
    ti_i: TurbineTI,
    *,
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> QueryField:
    """Normalized Gaussian velocity deficit of turbine `i`; near/far computed and masked."""
    D = turbine.rotor_diameter
    yaw = -1.0 * yaw_i

    # No cosd(yaw) here: FLORIS's velocity gauss omits the factor its deflection
    # gauss applies to uR (with a hardcoded cosd(tilt) = 1). Upstream asymmetry.
    uR = u_initial * ct_i / (2.0 * (1.0 - jnp.sqrt(1.0 - ct_i)))
    u0 = u_initial * jnp.sqrt(1.0 - ct_i)
    sigma_z0 = D * 0.5 * jnp.sqrt(uR / (u_initial + u0))
    # FLORIS also multiplies by cosd(wind_veer), unity at the default wind_veer = 0.
    sigma_y0 = sigma_z0 * cosd(yaw)

    xR = x_i
    # sqrt(1 - ct) in this numerator, where the deflection model uses
    # sqrt(1 - ct*cosd(yaw)); the two FLORIS gauss models genuinely differ here.
    x0 = (
        D
        * cosd(yaw)
        * (1.0 + jnp.sqrt(1.0 - ct_i))
        / (
            jnp.sqrt(2.0)
            * (4.0 * ALPHA * ti_i + 2.0 * BETA * (1.0 - jnp.sqrt(1.0 - ct_i)))
        )
        + x_i
    )

    near_mask = (x > xR + 0.1) & (x < x0)
    far_mask = x >= x0

    ramp_up = (x - xR) / (x0 - xR)
    ramp_down = (x0 - x) / (x0 - xR)
    near_base = ramp_down * 0.501 * D * jnp.sqrt(ct_i / 2.0)
    sigma_y_near = (near_base + ramp_up * sigma_y0) * (x >= xR) + (x < xR) * 0.5 * D
    sigma_z_near = (near_base + ramp_up * sigma_z0) * (x >= xR) + (x < xR) * 0.5 * D
    r_squared_near, amplitude_near = _gaussian_deficit_terms(
        sigma_y_near, sigma_z_near, y, y_i, deflection, z, ct_i, yaw, turbine
    )
    # FLORIS's gaussian_function divides r_squared by 2*sqrt(0.5)**2, which evaluates
    # to 1.0000000000000002; that 1-ULP infidelity is ~1e-15 relative in u.
    near_deficit = amplitude_near * jnp.exp(-r_squared_near) * near_mask

    ky = KA * ti_i + KB
    kz = KA * ti_i + KB
    sigma_y_far = (ky * (x - x0) + sigma_y0) * far_mask + sigma_y0 * (x < x0)
    sigma_z_far = (kz * (x - x0) + sigma_z0) * far_mask + sigma_z0 * (x < x0)
    r_squared_far, amplitude_far = _gaussian_deficit_terms(
        sigma_y_far, sigma_z_far, y, y_i, deflection, z, ct_i, yaw, turbine
    )
    far_deficit = amplitude_far * jnp.exp(-r_squared_far) * far_mask

    return near_deficit + far_deficit
