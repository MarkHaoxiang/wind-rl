from typing import Final

import jax.numpy as jnp

from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec
from windrl_engine.physics.flow import AMBIENT_TI
from windrl_engine.physics.frame import GRID, RotorField, RotorPlane, Scalar
from windrl_engine.physics.thrust import cubic_mean

CRESPO_INITIAL: Final = 0.1
CRESPO_CONSTANT: Final = 0.5
CRESPO_AI: Final = 0.8
CRESPO_DOWNSTREAM: Final = -0.32

# Literals of floris 4.6.6's sequential_solver, none exposed in default_inputs.yaml.
# The gain is 2 there and 1 in full_flow_sequential_solver; we mirror the former.
GCH_GAIN: Final = 2.0
LATERAL_GATE_DIAMETERS: Final = 2.0
DOWNSTREAM_INFLUENCE_DIAMETERS: Final = 15.0
AREA_OVERLAP_THRESHOLD: Final = 0.05


def crespo_hernandez(
    x: RotorField, x_i: Scalar, a_i: Scalar, *, turbine: TurbineSpec = DEFAULT_TURBINE
) -> RotorField:
    """Crespo-Hernandez wake-added turbulence intensity from turbine `i`."""
    D = turbine.rotor_diameter
    delta_x = x - x_i
    upstream_mask = delta_x <= 0.1
    downstream_mask = delta_x > -0.1
    delta_x = delta_x * downstream_mask + upstream_mask
    ti: RotorField = (
        CRESPO_CONSTANT
        * a_i**CRESPO_AI
        * AMBIENT_TI**CRESPO_INITIAL
        * (delta_x / D) ** CRESPO_DOWNSTREAM
    )
    return jnp.where(downstream_mask, ti, 0.0)


def yaw_added_mixing(
    u_i: RotorPlane,
    ti_i: RotorPlane,
    v_i: RotorPlane,
    w_i: RotorPlane,
    turb_v_i: RotorPlane,
    turb_w_i: RotorPlane,
) -> Scalar:
    """Turbulence-intensity increment from yaw-induced spanwise/vertical mixing."""
    ti_ambient = ti_i[0, 0]
    rotor_speed = cubic_mean(u_i)
    k = (rotor_speed * ti_ambient) ** 2 / (2 / 3)
    u_term = jnp.sqrt(2 * k)
    v_term = jnp.mean(v_i + turb_v_i)
    w_term = jnp.mean(w_i + turb_w_i)
    k_total = 0.5 * (u_term**2 + v_term**2 + w_term**2)
    ti_total = jnp.sqrt((2 / 3) * k_total) / rotor_speed
    return ti_total - ti_ambient


def wake_added_turbulence(
    ti: RotorField,
    deficit: RotorField,
    u_initial: RotorField,
    wake_added_ti: RotorField,
    x: RotorField,
    y: RotorField,
    x_i: Scalar,
    y_i: Scalar,
    *,
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> RotorField:
    """Fold area-overlap-weighted wake TI into the per-turbine field via elementwise max."""
    D = turbine.rotor_diameter
    area_overlap = jnp.sum(
        deficit * u_initial > AREA_OVERLAP_THRESHOLD, axis=(-2, -1)
    ) / (GRID * GRID)
    ti_added = (
        area_overlap[..., None, None]
        * jnp.nan_to_num(wake_added_ti, posinf=0.0)
        * (x > x_i)
        * (jnp.abs(y_i - y) < LATERAL_GATE_DIAMETERS * D)
        * (x <= DOWNSTREAM_INFLUENCE_DIAMETERS * D + x_i)
    )
    return jnp.maximum(jnp.sqrt(ti_added**2 + AMBIENT_TI**2), ti)
