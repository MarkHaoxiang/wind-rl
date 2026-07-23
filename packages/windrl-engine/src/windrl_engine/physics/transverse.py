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
    freestream_velocity: Scalar,
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
    tip_speed_ratio = turbine.tsr
    mixing_length = D / 8
    eps = EPS_GAIN * D

    vel_top = ((HUB_HEIGHT + D / 2) / HUB_HEIGHT) ** SHEAR
    vel_bottom = ((HUB_HEIGHT - D / 2) / HUB_HEIGHT) ** SHEAR
    gamma_top = (
        sind(yaw_i)
        * cosd(yaw_i)
        * (math.pi / 8)
        * D
        * vel_top
        * freestream_velocity
        * ct_i
    )
    gamma_bottom = (
        -1.0
        * sind(yaw_i)
        * cosd(yaw_i)
        * (math.pi / 8)
        * D
        * vel_bottom
        * freestream_velocity
        * ct_i
    )
    gamma_wake_rotation = (
        0.25 * 2 * math.pi * D * (a_i - a_i**2) * rotor_speed_i / tip_speed_ratio
    )

    l_m = KAPPA * z / (1 + KAPPA * z / mixing_length)
    nu = l_m**2 * jnp.abs(dudz_initial)
    delta_x = x - x_i
    decay = eps**2 / (4 * nu * delta_x / freestream_velocity + eps**2)
    y_locs = (y - y_i) + NUM_EPS

    two_pi = 2 * math.pi

    z_top = z - (HUB_HEIGHT + D / 2) + NUM_EPS
    r_top = y_locs**2 + z_top**2
    core_top = 1.0 - jnp.exp(-r_top / eps**2)
    v_top = gamma_top * z_top / (two_pi * r_top) * core_top * decay
    w_top = -1.0 * gamma_top * y_locs / (two_pi * r_top) * core_top * decay

    z_bottom = z - (HUB_HEIGHT - D / 2) + NUM_EPS
    r_bottom = y_locs**2 + z_bottom**2
    core_bottom = 1.0 - jnp.exp(-r_bottom / eps**2)
    v_bottom = gamma_bottom * z_bottom / (two_pi * r_bottom) * core_bottom * decay
    w_bottom = -1.0 * gamma_bottom * y_locs / (two_pi * r_bottom) * core_bottom * decay

    z_core = z - HUB_HEIGHT + NUM_EPS
    r_core = y_locs**2 + z_core**2
    core_core = 1.0 - jnp.exp(-r_core / eps**2)
    v_core = gamma_wake_rotation * z_core / (two_pi * r_core) * core_core * decay
    w_core = -1.0 * gamma_wake_rotation * y_locs / (two_pi * r_core) * core_core * decay

    z_top_mirror = z + (HUB_HEIGHT + D / 2) + NUM_EPS
    r_top_mirror = y_locs**2 + z_top_mirror**2
    core_top_mirror = 1.0 - jnp.exp(-r_top_mirror / eps**2)
    v_top_mirror = (
        -1.0
        * gamma_top
        * z_top_mirror
        / (two_pi * r_top_mirror)
        * core_top_mirror
        * decay
    )
    w_top_mirror = (
        gamma_top * y_locs / (two_pi * r_top_mirror) * core_top_mirror * decay
    )

    z_bottom_mirror = z + (HUB_HEIGHT - D / 2) + NUM_EPS
    r_bottom_mirror = y_locs**2 + z_bottom_mirror**2
    core_bottom_mirror = 1.0 - jnp.exp(-r_bottom_mirror / eps**2)
    v_bottom_mirror = (
        -1.0
        * gamma_bottom
        * z_bottom_mirror
        / (two_pi * r_bottom_mirror)
        * core_bottom_mirror
        * decay
    )
    w_bottom_mirror = (
        gamma_bottom * y_locs / (two_pi * r_bottom_mirror) * core_bottom_mirror * decay
    )

    z_core_mirror = z + HUB_HEIGHT + NUM_EPS
    r_core_mirror = y_locs**2 + z_core_mirror**2
    core_core_mirror = 1.0 - jnp.exp(-r_core_mirror / eps**2)
    v_core_mirror = (
        -1.0
        * gamma_wake_rotation
        * z_core_mirror
        / (two_pi * r_core_mirror)
        * core_core_mirror
        * decay
    )
    w_core_mirror = (
        gamma_wake_rotation
        * y_locs
        / (two_pi * r_core_mirror)
        * core_core_mirror
        * decay
    )

    v = v_top + v_bottom + v_top_mirror + v_bottom_mirror + v_core + v_core_mirror
    w = w_top + w_bottom + w_top_mirror + w_bottom_mirror + w_core + w_core_mirror
    gate = delta_x > 0.0 if self_exclude else delta_x >= 0.0
    v = jnp.where(gate, v, 0.0)
    w = jnp.where(gate, w, 0.0)
    w = jnp.where(w >= 0.0, w, 0.0)
    return v, w
