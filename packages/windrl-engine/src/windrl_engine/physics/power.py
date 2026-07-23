import jax.numpy as jnp
from jaxtyping import Array, Float

from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec, power_lookup
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.flow import AIR_DENSITY
from windrl_engine.physics.frame import cosd
from windrl_engine.physics.solver import FlowSolution


def turbine_powers(
    u: Float[Array, "turbines grid grid"],
    yaw: Float[Array, "turbines"],
    *,
    turbine: TurbineSpec = DEFAULT_TURBINE,
) -> Float[Array, "turbines"]:
    """Per-turbine electrical power (W) from the cubic-mean rotor velocity."""
    rotor_speed = jnp.cbrt(jnp.mean(u**3, axis=(1, 2)))
    u_eff = (AIR_DENSITY / turbine.ref_density) ** (1 / 3) * rotor_speed
    u_eff = u_eff * cosd(yaw) ** (turbine.pP / 3)
    return power_lookup(turbine, u_eff)


def load_proxies(solution: FlowSolution) -> Float[Array, "turbines 4"]:
    """Per-turbine [TI, std(u), std(v), std(w)] over the rotor plane."""
    return jnp.stack(
        [
            solution.turbulence_intensity,
            jnp.std(solution.u, axis=(1, 2)),
            jnp.std(solution.v, axis=(1, 2)),
            jnp.std(solution.w, axis=(1, 2)),
        ],
        axis=1,
    )


def local_wind(
    solution: FlowSolution, wind: WindCondition
) -> tuple[Float[Array, "turbines"], Float[Array, "turbines"]]:
    """Per-turbine local speed ∛(mean u³) and inflow direction (deg)."""
    speed = jnp.cbrt(jnp.mean(solution.u**3, axis=(1, 2)))
    direction = jnp.mean(
        wind.direction - jnp.rad2deg(jnp.arctan2(solution.v, solution.u)), axis=(1, 2)
    )
    return speed, direction
