from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec
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


def rotor_plane_x(x: Float[Array, "turbines"]) -> Float[Array, "turbines"]:
    """Rotor-plane streamwise x reproducing FLORIS's np.mean to 1 ULP."""
    # x is constant over the rotor plane, so FLORIS's x_i = np.mean of nine identical
    # values, which rounds to x or x+ulp; that one-ulp choice decides the transverse
    # source-plane `delta_x >= 0` gate. np.mean's pairwise sum is round(9x), so `e1`
    # is the exact residual 9*x - round(9x): the two subtractions are Sterbenz-exact
    # (8*x is exact), hence FMA-invariant and identical under jit and vmap. np.mean
    # rounds up (dropping the self plane) exactly when e1 < -4.5*ulp.
    p = 9.0 * x
    e1 = (8.0 * x - p) + x
    ulp = jnp.nextafter(x, jnp.inf) - x
    return x + jnp.where(e1 < -4.5 * ulp, ulp, 0.0)


class _Carry(NamedTuple):
    wake_field: RotorField
    v: RotorField
    w: RotorField
    turbulence_intensity: RotorField


def solve_farm(
    layout: FarmLayout,
    wind: WindCondition,
    yaw: Float[Array, "turbines"],
    *,
    fidelity: str = "floris",
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> FlowSolution:
    """Steady-state GCH wake solve; fields in original turbine order.

    ``fidelity="corrected"`` (static; a separate jit specialization) drops two FLORIS
    reference quirks: the rotor-plane self-interaction becomes a deterministic strict
    self-exclusion (independent of ``x_i`` float rounding), and the yaw-added-recovery
    TI update is applied before *both* the deflection and deficit calls (the reference
    lets deflection see the stale TI). ``turbine`` selects the NREL-5MW library.
    """
    corrected = fidelity == "corrected"
    x_rot, y_rot = rotate_to_wind_frame(layout.x, layout.y, wind.direction)
    x_grid, y_grid, z_grid = rotor_grid(x_rot, y_rot, turbine=turbine)
    sorted_idx, unsorted_idx = upstream_order(x_rot)

    x_sorted = x_grid[sorted_idx]
    y_sorted = y_grid[sorted_idx]
    z_sorted = z_grid[sorted_idx]
    yaw_sorted = yaw[sorted_idx]
    u_initial, dudz_initial = initial_flow(z_sorted, wind.speed, turbine=turbine)

    # corrected: x_i is exactly the turbine's own rotor-plane x (no mean-rounding
    # trick), so the strict `delta_x > 0` transverse gate excludes only its own plane.
    x_i_all = x_sorted[:, 0, 0] if corrected else rotor_plane_x(x_sorted[:, 0, 0])
    y_i_all = jnp.mean(y_sorted, axis=(1, 2))
    freestream_velocity = jnp.mean(u_initial)
    num_turbines = x_sorted.shape[0]

    def body(i: Array, carry: _Carry) -> _Carry:
        u_current = u_initial - carry.wake_field
        x_i = x_i_all[i]
        y_i = y_i_all[i]
        u_i = u_current[i]
        v_i = carry.v[i]
        w_i = carry.w[i]
        ti_i = carry.turbulence_intensity[i]
        yaw_i = yaw_sorted[i]

        rotor_speed = cubic_mean(u_i)
        ct_i = effective_ct(rotor_speed, yaw_i, turbine=turbine)
        a_i = axial_induction(ct_i, yaw_i)

        added_yaw = wake_added_yaw(
            v_i,
            y_sorted[i] - y_i,
            z_sorted[i],
            freestream_velocity,
            rotor_speed,
            ct_i,
            a_i,
            turbine=turbine,
        )
        effective_yaw = yaw_i + added_yaw

        v_wake, w_wake = transverse_velocity(
            x_sorted,
            y_sorted,
            z_sorted,
            dudz_initial,
            freestream_velocity,
            rotor_speed,
            x_i,
            y_i,
            yaw_i,
            ct_i,
            a_i,
            turbine=turbine,
            self_exclude=corrected,
        )

        yaw_mixing_ti_increment = yaw_added_mixing(
            u_i, ti_i, v_i, w_i, v_wake[i], w_wake[i]
        )
        # FLORIS mutates turbulence_intensity_i in place: the deficit always sees the
        # yaw-mixing-updated TI; the reference lets the deflection keep the stale TI,
        # while `corrected` feeds both the updated value.
        ti = carry.turbulence_intensity.at[i].add(GCH_GAIN * yaw_mixing_ti_increment)
        ti_deflection = ti[i] if corrected else ti_i

        deflection = deflection_field(
            x_sorted,
            u_initial,
            x_i,
            effective_yaw,
            ti_deflection,
            ct_i,
            turbine=turbine,
        )
        deficit = deficit_field(
            x_sorted,
            y_sorted,
            z_sorted,
            u_initial,
            deflection,
            x_i,
            y_i,
            ct_i,
            yaw_i,
            ti[i],
            turbine=turbine,
        )
        wake_field = jnp.hypot(carry.wake_field, deficit * u_initial)

        wake_added_ti = crespo_hernandez(x_sorted, x_i, a_i, turbine=turbine)
        ti = wake_added_turbulence(
            ti,
            deficit,
            u_initial,
            wake_added_ti,
            x_sorted,
            y_sorted,
            x_i,
            y_i,
            turbine=turbine,
        )

        return _Carry(wake_field, carry.v + v_wake, carry.w + w_wake, ti)

    init = _Carry(
        jnp.zeros_like(u_initial),
        jnp.zeros_like(u_initial),
        jnp.zeros_like(u_initial),
        jnp.full_like(u_initial, AMBIENT_TI),
    )
    final = jax.lax.fori_loop(0, num_turbines, body, init)

    u_sorted = u_initial - final.wake_field
    ti_sorted = jnp.mean(final.turbulence_intensity, axis=(1, 2))
    return FlowSolution(
        u=u_sorted[unsorted_idx],
        v=final.v[unsorted_idx],
        w=final.w[unsorted_idx],
        turbulence_intensity=ti_sorted[unsorted_idx],
    )
