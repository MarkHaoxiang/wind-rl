import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.deficit import sosfs_combine
from windrl_engine.physics.deflection import wake_added_yaw
from windrl_engine.physics.flow import initial_flow
from windrl_engine.physics.solver import (
    Fidelity,
    rotor_plane_x,
    solve_farm,
    wake_contribution,
    wind_frame,
)
from windrl_engine.physics.thrust import axial_induction, cubic_mean, effective_ct

QueryPlane = Float[Array, "res_a res_b"]


def solve_query_points(
    layout: FarmLayout,
    wind: WindCondition,
    yaw: Float[Array, "turbines"],
    x_query: QueryPlane,
    y_query: QueryPlane,
    z_query: QueryPlane,
    *,
    fidelity: Fidelity = "floris",
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> QueryPlane:
    """Streamwise u (m/s) at arbitrary points of the wind-aligned frame.

    ``x_query``/``y_query``/``z_query`` are wind-frame meters (z above ground).
    The farm is solved first and each turbine then re-casts its wake onto the
    query points from its *converged* transverse velocity and rotor-mean
    turbulence intensity, so the field is a post-hoc rendering of the solution
    rather than a second solve.
    """
    corrected = fidelity == "corrected"
    solution = solve_farm(layout, wind, yaw, fidelity=fidelity, turbine=turbine)
    frame = wind_frame(layout, wind, yaw, turbine)

    u_turbine = solution.u[frame.sorted_idx]
    v_turbine = solution.v[frame.sorted_idx]
    ti_sorted = solution.turbulence_intensity[frame.sorted_idx]

    x_i_all = frame.x[:, 0, 0] if corrected else rotor_plane_x(frame.x[:, 0, 0])
    u_initial_query, _ = initial_flow(z_query, wind.speed, turbine=turbine)
    num_turbines = frame.x.shape[0]

    def body(i: Array, wake_field: QueryPlane) -> QueryPlane:
        y_i = frame.y_i[i]
        yaw_i = frame.yaw[i]
        ti_i = ti_sorted[i]

        rotor_speed = cubic_mean(u_turbine[i])
        ct_i = effective_ct(rotor_speed, yaw_i, turbine=turbine)
        a_i = axial_induction(ct_i, yaw_i)

        effective_yaw = yaw_i + wake_added_yaw(
            v_turbine[i],
            frame.y[i] - y_i,
            frame.z[i],
            frame.freestream_velocity,
            rotor_speed,
            ct_i,
            a_i,
            turbine=turbine,
        )
        deficit = wake_contribution(
            x_query,
            y_query,
            z_query,
            u_initial_query,
            x_i_all[i],
            y_i,
            effective_yaw,
            yaw_i,
            ct_i,
            ti_i,
            ti_i,
            turbine,
        )
        return sosfs_combine(wake_field, deficit * u_initial_query)

    wake_field: QueryPlane = jax.lax.fori_loop(
        0, num_turbines, body, jnp.zeros_like(x_query)
    )
    u: QueryPlane = u_initial_query - wake_field
    return u
