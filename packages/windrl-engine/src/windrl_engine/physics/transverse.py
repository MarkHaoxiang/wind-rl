import math
from typing import Final

import jax.numpy as jnp

from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec
from windrl_engine.physics.deflection import EPS_GAIN, NUM_EPS
from windrl_engine.physics.flow import SHEAR
from windrl_engine.physics.frame import RotorField, Scalar, cosd, sind

KAPPA: Final = 0.41
TWO_PI: Final = 2 * math.pi


def _vortex(
    sign: float,
    circulation: Scalar,
    y_locs: RotorField,
    z_offset: RotorField,
    eps: Scalar | float,
    decay: RotorField,
) -> tuple[RotorField, RotorField]:
    r_squared = y_locs**2 + z_offset**2
    core = 1.0 - jnp.exp(-r_squared / eps**2)
    v = sign * circulation * z_offset / (TWO_PI * r_squared) * core * decay
    w = -sign * circulation * y_locs / (TWO_PI * r_squared) * core * decay
    return v, w


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
    """Spanwise/vertical velocities (v, w) from turbine `i`'s 3 real + 3 mirror vortices."""
    D = turbine.rotor_diameter
    hub_height = turbine.hub_height
    tip_speed_ratio = turbine.tsr
    mixing_length = D / 8
    eps = EPS_GAIN * D

    vel_top = ((hub_height + D / 2) / hub_height) ** SHEAR
    vel_bottom = ((hub_height - D / 2) / hub_height) ** SHEAR
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

    # V1/W1 top
    v_top, w_top = _vortex(
        1.0, gamma_top, y_locs, z - (hub_height + D / 2) + NUM_EPS, eps, decay
    )
    # V2/W2 bottom
    v_bottom, w_bottom = _vortex(
        1.0, gamma_bottom, y_locs, z - (hub_height - D / 2) + NUM_EPS, eps, decay
    )
    # V5/W5 wake rotation
    v_core, w_core = _vortex(
        1.0, gamma_wake_rotation, y_locs, z - hub_height + NUM_EPS, eps, decay
    )
    # V3/W3 top mirror
    v_top_mirror, w_top_mirror = _vortex(
        -1.0, gamma_top, y_locs, z + (hub_height + D / 2) + NUM_EPS, eps, decay
    )
    # V4/W4 bottom mirror
    v_bottom_mirror, w_bottom_mirror = _vortex(
        -1.0, gamma_bottom, y_locs, z + (hub_height - D / 2) + NUM_EPS, eps, decay
    )
    # V6/W6 wake-rotation mirror
    v_core_mirror, w_core_mirror = _vortex(
        -1.0, gamma_wake_rotation, y_locs, z + hub_height + NUM_EPS, eps, decay
    )

    v = v_top + v_bottom + v_top_mirror + v_bottom_mirror + v_core + v_core_mirror
    w = w_top + w_bottom + w_top_mirror + w_bottom_mirror + w_core + w_core_mirror
    # self_exclude drops turbine `i`'s own rotor plane deterministically; FLORIS's
    # `>= 0` instead lets the floating-point rounding of `x_i` decide.
    gate = delta_x > 0.0 if self_exclude else delta_x >= 0.0
    v = jnp.where(gate, v, 0.0)
    w = jnp.where(gate, w, 0.0)
    w = jnp.where(w >= 0.0, w, 0.0)
    return v, w
