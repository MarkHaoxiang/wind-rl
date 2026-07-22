from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.deficit import deficit_field
from windrl_engine.physics.deflection import deflection_field, wake_added_yaw
from windrl_engine.physics.flow import AMBIENT_TI, initial_flow
from windrl_engine.physics.frame import (
    RotorField,
    rotate_to_wind_frame,
    rotor_grid,
    upstream_order,
)
from windrl_engine.physics.thrust import axial_induction, cubic_mean, effective_ct
from windrl_engine.physics.transverse import transverse_velocity
from windrl_engine.physics.turbulence import (
    GCH_GAIN,
    crespo_hernandez,
    wake_added_turbulence,
    yaw_added_mixing,
)


class FlowSolution(NamedTuple):
    u: Float[Array, "turbines grid grid"]  # m/s, original turbine order
    v: Float[Array, "turbines grid grid"]
    w: Float[Array, "turbines grid grid"]
    turbulence_intensity: Float[Array, "turbines"]


def rotor_plane_x(xc: Float[Array, "turbines"]) -> Float[Array, "turbines"]:
    """Rotor-plane streamwise x reproducing FLORIS's np.mean to 1 ULP (spec §5.3)."""
    # x is constant over the rotor plane, so FLORIS's x_i = np.mean of nine identical
    # values, which rounds to x or x+ulp; that one-ulp choice decides the transverse
    # source-plane `delta_x >= 0` gate (spec §5.3). np.mean's pairwise sum is
    # round(9x), so `e1` is the exact residual 9*x - round(9x): the two subtractions
    # are Sterbenz-exact (8*x is exact), hence FMA-invariant and identical under jit
    # and vmap. np.mean rounds up (dropping the self plane) exactly when e1 < -4.5*ulp.
    p = 9.0 * xc
    e1 = (8.0 * xc - p) + xc
    ulp = jnp.nextafter(xc, jnp.inf) - xc
    return xc + jnp.where(e1 < -4.5 * ulp, ulp, 0.0)


class _Carry(NamedTuple):
    wake_field: RotorField
    v: RotorField
    w: RotorField
    turbulence_intensity: RotorField


def solve_farm(
    layout: FarmLayout, wind: WindCondition, yaw: Float[Array, "turbines"]
) -> FlowSolution:
    """Steady-state GCH wake solve; fields in original turbine order (spec §5)."""
    x_rot, y_rot = rotate_to_wind_frame(layout.x, layout.y, wind.direction)
    x_grid, y_grid, z_grid = rotor_grid(x_rot, y_rot)
    sorted_idx, unsorted_idx = upstream_order(x_rot)

    xs = x_grid[sorted_idx]
    ys = y_grid[sorted_idx]
    zs = z_grid[sorted_idx]
    yaw_s = yaw[sorted_idx]
    u_initial, dudz_initial = initial_flow(zs, wind.speed)

    x_i_all = rotor_plane_x(xs[:, 0, 0])
    y_i_all = jnp.mean(ys, axis=(1, 2))
    uinf = jnp.mean(u_initial)
    n = xs.shape[0]

    def body(i: Array, carry: _Carry) -> _Carry:
        u_s = u_initial - carry.wake_field
        x_i = x_i_all[i]
        y_i = y_i_all[i]
        u_i = u_s[i]
        v_i = carry.v[i]
        w_i = carry.w[i]
        ti_i = carry.turbulence_intensity[i]
        yaw_i = yaw_s[i]

        rotor_speed = cubic_mean(u_i)
        ct_i = effective_ct(rotor_speed, yaw_i)
        a_i = axial_induction(ct_i, yaw_i)

        added = wake_added_yaw(v_i, ys[i] - y_i, zs[i], uinf, rotor_speed, ct_i, a_i)
        effective_yaw = yaw_i + added

        deflection = deflection_field(xs, u_initial, x_i, effective_yaw, ti_i, ct_i)
        v_wake, w_wake = transverse_velocity(
            xs, ys, zs, dudz_initial, uinf, rotor_speed, x_i, y_i, yaw_i, ct_i, a_i
        )

        i_mixing = yaw_added_mixing(u_i, ti_i, v_i, w_i, v_wake[i], w_wake[i])
        # FLORIS mutates turbulence_intensity_i in place, so the deficit sees the
        # yaw-mixing-updated TI while the deflection above kept the old TI (spec §5.3).
        ti = carry.turbulence_intensity.at[i].add(GCH_GAIN * i_mixing)

        deficit = deficit_field(
            xs, ys, zs, u_initial, deflection, x_i, y_i, ct_i, yaw_i, ti[i]
        )
        wake_field = jnp.hypot(carry.wake_field, deficit * u_initial)

        wake_added_ti = crespo_hernandez(xs, x_i, a_i)
        ti = wake_added_turbulence(
            ti, deficit, u_initial, wake_added_ti, xs, ys, x_i, y_i
        )

        return _Carry(wake_field, carry.v + v_wake, carry.w + w_wake, ti)

    init = _Carry(
        jnp.zeros_like(u_initial),
        jnp.zeros_like(u_initial),
        jnp.zeros_like(u_initial),
        jnp.full_like(u_initial, AMBIENT_TI),
    )
    final = jax.lax.fori_loop(0, n, body, init)

    u_sorted = u_initial - final.wake_field
    ti_sorted = jnp.mean(final.turbulence_intensity, axis=(1, 2))
    return FlowSolution(
        u=u_sorted[unsorted_idx],
        v=final.v[unsorted_idx],
        w=final.w[unsorted_idx],
        turbulence_intensity=ti_sorted[unsorted_idx],
    )
