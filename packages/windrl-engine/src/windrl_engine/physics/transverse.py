import math
from typing import Final

import jax.numpy as jnp

from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec
from windrl_engine.physics.deflection import EPS_GAIN, NUM_EPS
from windrl_engine.physics.flow import SHEAR
from windrl_engine.physics.frame import RotorField, Scalar, cosd, sind

KAPPA: Final = 0.41


def transverse_velocity(
    x: RotorField,
    y: RotorField,
    z: RotorField,
    dudz_initial: RotorField,
    uinf: Scalar,
    rotor_speed_i: Scalar,
    x_i: Scalar,
    y_i: Scalar,
    yaw_i: Scalar,
    ct_i: Scalar,
    a_i: Scalar,
    *,
    turbine: TurbineSpec = DEFAULT_TURBINE,
    self_exclude: bool = False,
) -> tuple[RotorField, RotorField]:
    """Spanwise/vertical velocities (v, w) from turbine `i`'s 3 real + 3 mirror vortices.

    ``self_exclude`` (corrected fidelity): gate the source planes with ``delta_x > 0``
    so turbine `i`'s own rotor plane is deterministically excluded, independent of the
    floating-point rounding of ``x_i`` (the FLORIS ULP quirk uses ``delta_x >= 0``).
    """
    D = turbine.rotor_diameter
    HUB_HEIGHT = turbine.hub_height
    TSR = turbine.tsr
    mixing_length = D / 8
    eps = EPS_GAIN * D

    vel_top = ((HUB_HEIGHT + D / 2) / HUB_HEIGHT) ** SHEAR
    vel_bottom = ((HUB_HEIGHT - D / 2) / HUB_HEIGHT) ** SHEAR
    gamma_top = sind(yaw_i) * cosd(yaw_i) * (math.pi / 8) * D * vel_top * uinf * ct_i
    gamma_bottom = (
        -1.0 * sind(yaw_i) * cosd(yaw_i) * (math.pi / 8) * D * vel_bottom * uinf * ct_i
    )
    gamma_wake_rotation = 0.25 * 2 * math.pi * D * (a_i - a_i**2) * rotor_speed_i / TSR

    lm = KAPPA * z / (1 + KAPPA * z / mixing_length)
    nu = lm**2 * jnp.abs(dudz_initial)
    delta_x = x - x_i
    decay = eps**2 / (4 * nu * delta_x / uinf + eps**2)
    y_locs = (y - y_i) + NUM_EPS

    two_pi = 2 * math.pi

    z_top = z - (HUB_HEIGHT + D / 2) + NUM_EPS
    r_top = y_locs**2 + z_top**2
    core_top = 1.0 - jnp.exp(-r_top / eps**2)
    v1 = gamma_top * z_top / (two_pi * r_top) * core_top * decay
    w1 = -1.0 * gamma_top * y_locs / (two_pi * r_top) * core_top * decay

    z_bottom = z - (HUB_HEIGHT - D / 2) + NUM_EPS
    r_bottom = y_locs**2 + z_bottom**2
    core_bottom = 1.0 - jnp.exp(-r_bottom / eps**2)
    v2 = gamma_bottom * z_bottom / (two_pi * r_bottom) * core_bottom * decay
    w2 = -1.0 * gamma_bottom * y_locs / (two_pi * r_bottom) * core_bottom * decay

    z_core = z - HUB_HEIGHT + NUM_EPS
    r_core = y_locs**2 + z_core**2
    core_core = 1.0 - jnp.exp(-r_core / eps**2)
    v5 = gamma_wake_rotation * z_core / (two_pi * r_core) * core_core * decay
    w5 = -1.0 * gamma_wake_rotation * y_locs / (two_pi * r_core) * core_core * decay

    z_top_m = z + (HUB_HEIGHT + D / 2) + NUM_EPS
    r_top_m = y_locs**2 + z_top_m**2
    core_top_m = 1.0 - jnp.exp(-r_top_m / eps**2)
    v3 = -1.0 * gamma_top * z_top_m / (two_pi * r_top_m) * core_top_m * decay
    w3 = gamma_top * y_locs / (two_pi * r_top_m) * core_top_m * decay

    z_bottom_m = z + (HUB_HEIGHT - D / 2) + NUM_EPS
    r_bottom_m = y_locs**2 + z_bottom_m**2
    core_bottom_m = 1.0 - jnp.exp(-r_bottom_m / eps**2)
    v4 = (
        -1.0 * gamma_bottom * z_bottom_m / (two_pi * r_bottom_m) * core_bottom_m * decay
    )
    w4 = gamma_bottom * y_locs / (two_pi * r_bottom_m) * core_bottom_m * decay

    z_core_m = z + HUB_HEIGHT + NUM_EPS
    r_core_m = y_locs**2 + z_core_m**2
    core_core_m = 1.0 - jnp.exp(-r_core_m / eps**2)
    v6 = (
        -1.0
        * gamma_wake_rotation
        * z_core_m
        / (two_pi * r_core_m)
        * core_core_m
        * decay
    )
    w6 = gamma_wake_rotation * y_locs / (two_pi * r_core_m) * core_core_m * decay

    v = v1 + v2 + v3 + v4 + v5 + v6
    w = w1 + w2 + w3 + w4 + w5 + w6
    gate = delta_x > 0.0 if self_exclude else delta_x >= 0.0
    v = jnp.where(gate, v, 0.0)
    w = jnp.where(gate, w, 0.0)
    w = jnp.where(w >= 0.0, w, 0.0)
    return v, w
