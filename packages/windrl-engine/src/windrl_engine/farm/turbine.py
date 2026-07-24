from typing import Final, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float

from windrl_engine.farm.floris_tables import load_nrel5mw_v4

D: Final = 126.0  # nominal NREL-5MW rotor diameter for synthetic row-layout spacing
HUB_HEIGHT: Final = 90.0  # nominal NREL-5MW hub height for synthetic query grids

TABLE_SIZE: Final = 54
TurbineTable = Float[Array, f"table={TABLE_SIZE}"]

_V4 = load_nrel5mw_v4()
_V4_WIND_SPEED: TurbineTable = jnp.asarray(_V4["wind_speed"])
_V4_POWER_KW: TurbineTable = jnp.asarray(_V4["power_kw"])
_V4_THRUST: TurbineTable = jnp.asarray(_V4["thrust"])


class TurbineSpec(NamedTuple):
    """Turbine geometry and power/thrust tables as a PyTree (single shared spec).

    Electrical power is one interpolation of the absolute-kW ``power_table``
    scaled by ``power_scale`` (1e3 -> W). ``ct_fill_low``/``ct_fill_high`` are the
    fill values applied to ``C_t`` outside the wind-speed table before the clip.
    """

    rotor_diameter: float
    hub_height: float
    pP: float
    tsr: float
    generator_efficiency: float
    ref_density: float
    wind_speed_table: TurbineTable
    thrust_table: TurbineTable
    power_table: TurbineTable
    power_scale: float
    ct_fill_low: float
    ct_fill_high: float


def nrel5mw_v4() -> TurbineSpec:
    """NREL-5MW as shipped in FLORIS 4.6.6 (cosine-loss, absolute-kW power table)."""
    return TurbineSpec(
        rotor_diameter=float(_V4["rotor_diameter"]),
        hub_height=float(_V4["hub_height"]),
        pP=float(_V4["pP"]),  # cosine_loss_exponent_yaw
        tsr=float(_V4["tsr"]),
        generator_efficiency=float(_V4["generator_efficiency"]),  # baked into the table
        ref_density=float(_V4["ref_density"]),
        wind_speed_table=_V4_WIND_SPEED,
        thrust_table=_V4_THRUST,
        power_table=_V4_POWER_KW,
        power_scale=1e3,
        ct_fill_low=0.0001,
        ct_fill_high=0.0001,
    )


DEFAULT_TURBINE: Final = nrel5mw_v4()


def ct_lookup(
    spec: TurbineSpec, wind_speed: Float[Array, "*shape"]
) -> Float[Array, "*shape"]:
    """Thrust coefficient C_t for ``spec``; linear interp, fill, then clip to (1e-4, 0.9999)."""
    ws = spec.wind_speed_table
    ct = jnp.interp(wind_speed, ws, spec.thrust_table)
    ct = jnp.where(wind_speed < ws[0], spec.ct_fill_low, ct)
    ct = jnp.where(wind_speed > ws[-1], spec.ct_fill_high, ct)
    return jnp.clip(ct, 0.0001, 0.9999)


def power_lookup(
    spec: TurbineSpec, u_eff: Float[Array, "*shape"]
) -> Float[Array, "*shape"]:
    """Electrical power (W) for ``spec`` from the effective rotor velocity; fill 0.0."""
    ws = spec.wind_speed_table
    power = jnp.interp(u_eff, ws, spec.power_table)
    power = jnp.where(u_eff < ws[0], 0.0, power)
    power = jnp.where(u_eff > ws[-1], 0.0, power)
    return power * spec.power_scale
