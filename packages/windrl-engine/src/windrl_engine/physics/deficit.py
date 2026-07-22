import jax.numpy as jnp

from windrl_engine.farm.turbine import HUB_HEIGHT, D
from windrl_engine.physics.deflection import ALPHA, BETA, KA, KB
from windrl_engine.physics.frame import QueryField, Scalar, TurbineTI, cosd


def _rc(
    sigma_y: QueryField,
    sigma_z: QueryField,
    y: QueryField,
    y_i: Scalar,
    deflection: QueryField,
    z: QueryField,
    ct_i: Scalar,
    yaw: Scalar,
) -> tuple[QueryField, QueryField]:
    a = 1.0 / (2.0 * sigma_y**2)
    c = 1.0 / (2.0 * sigma_z**2)
    r = a * (y - y_i - deflection) ** 2 + c * (z - HUB_HEIGHT) ** 2
    d = jnp.clip(1.0 - ct_i * cosd(yaw) / (8.0 * sigma_y * sigma_z / D**2), 0.0, 1.0)
    return r, 1.0 - jnp.sqrt(d)


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
) -> QueryField:
    """Normalized Gaussian velocity deficit of turbine `i`; near/far computed and masked."""
    yaw = -1.0 * yaw_i

    uR = u_initial * ct_i / (2.0 * (1.0 - jnp.sqrt(1.0 - ct_i)))
    u0 = u_initial * jnp.sqrt(1.0 - ct_i)
    sigma_z0 = D * 0.5 * jnp.sqrt(uR / (u_initial + u0))
    sigma_y0 = sigma_z0 * cosd(yaw)

    xR = x_i
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
    r_near, c_near = _rc(sigma_y_near, sigma_z_near, y, y_i, deflection, z, ct_i, yaw)
    near_deficit = c_near * jnp.exp(-r_near) * near_mask

    ky = KA * ti_i + KB
    kz = KA * ti_i + KB
    sigma_y_far = (ky * (x - x0) + sigma_y0) * far_mask + sigma_y0 * (x < x0)
    sigma_z_far = (kz * (x - x0) + sigma_z0) * far_mask + sigma_z0 * (x < x0)
    r_far, c_far = _rc(sigma_y_far, sigma_z_far, y, y_i, deflection, z, ct_i, yaw)
    far_deficit = c_far * jnp.exp(-r_far) * far_mask

    return near_deficit + far_deficit
