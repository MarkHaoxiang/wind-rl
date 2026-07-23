import math
from typing import Final, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float

D: Final = 126.0
HUB_HEIGHT: Final = 90.0
pP: Final = 1.88
TSR: Final = 8.0
GENERATOR_EFFICIENCY: Final = 1.0
REF_DENSITY: Final = 1.225
ROTOR_AREA: Final = math.pi * 63.0**2

TABLE_SIZE: Final = 50
TurbineTable = Float[Array, f"table={TABLE_SIZE}"]

# Verbatim from FLORIS 3.5 turbine_library/nrel_5MW.yaml (power_thrust_table).
# fmt: off
WIND_SPEED: TurbineTable = jnp.asarray([
    0.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
    6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0,
    11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0,
    16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0,
    21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.01, 25.02,
])
THRUST: TurbineTable = jnp.asarray([
    0.0, 0.0, 0.0, 0.99, 0.99, 0.97373036, 0.92826162, 0.89210543, 0.86100905, 0.835423,
    0.81237673, 0.79225789, 0.77584769, 0.7629228, 0.76156073, 0.76261984, 0.76169723, 0.75232027, 0.74026851, 0.72987175,
    0.70701647, 0.54054532, 0.45509459, 0.39343381, 0.34250785, 0.30487242, 0.27164979, 0.24361964, 0.21973831, 0.19918151,
    0.18131868, 0.16537679, 0.15103727, 0.13998636, 0.1289037, 0.11970413, 0.11087113, 0.10339901, 0.09617888, 0.09009926,
    0.08395078, 0.0791188, 0.07448356, 0.07050731, 0.06684119, 0.06345518, 0.06032267, 0.05741999, 0.05472609, 0.0,
])
POWER: TurbineTable = jnp.asarray([
    0.0, 0.0, 0.0, 0.178085, 0.289075, 0.349022, 0.384728, 0.406059, 0.420228, 0.428823,
    0.433873, 0.436223, 0.436845, 0.436575, 0.436511, 0.436561, 0.436517, 0.435903, 0.434673, 0.433230,
    0.430466, 0.378869, 0.335199, 0.297991, 0.266092, 0.238588, 0.214748, 0.193981, 0.175808, 0.159835,
    0.145741, 0.133256, 0.122157, 0.112257, 0.103399, 0.095449, 0.088294, 0.081836, 0.075993, 0.070692,
    0.065875, 0.061484, 0.057476, 0.053809, 0.050447, 0.047358, 0.044518, 0.041900, 0.039483, 0.0,
])
# fmt: on

# power_interp lookup table: 0.5·rotor_area·Cp(ws)·gen_eff·ws³ (W).
_INNER_POWER: TurbineTable = (
    0.5 * ROTOR_AREA * POWER * GENERATOR_EFFICIENCY * WIND_SPEED**3
)


def ct_interp(wind_speed: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """Thrust coefficient C_t; linear table interp, fill (0.0001, 0.9999) then clip."""
    ct = jnp.interp(wind_speed, WIND_SPEED, THRUST)
    ct = jnp.where(wind_speed < WIND_SPEED[0], 0.0001, ct)
    ct = jnp.where(wind_speed > WIND_SPEED[-1], 0.9999, ct)
    return jnp.clip(ct, 0.0001, 0.9999)


def cp_interp(wind_speed: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """Power coefficient C_p; linear table interp, fill (0.0, 1.0)."""
    cp = jnp.interp(wind_speed, WIND_SPEED, POWER)
    cp = jnp.where(wind_speed < WIND_SPEED[0], 0.0, cp)
    return jnp.where(wind_speed > WIND_SPEED[-1], 1.0, cp)


def power_interp(wind_speed: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """Inner rotor power (W) before density scaling; linear table interp, fill 0.0."""
    power = jnp.interp(wind_speed, WIND_SPEED, _INNER_POWER)
    power = jnp.where(wind_speed < WIND_SPEED[0], 0.0, power)
    return jnp.where(wind_speed > WIND_SPEED[-1], 0.0, power)


# --- v4 nrel_5MW (FLORIS 4.6.6 cosine-loss operation model) -------------------
# Verbatim from FLORIS 4.6.6 turbine_library/nrel_5MW.yaml (power_thrust_table).
# The `power` table is absolute output in kW; the `cosine-loss` model interpolates
# it directly (·1e3 -> W) rather than the v3 Cp -> inner-power path.
# fmt: off
_V4_WIND_SPEED = jnp.asarray([
    0.0, 2.9, 3.0, 4.0, 5.0, 6.0, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9,
    8.0, 9.0, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1,
    11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 25.1, 50.0,
])
_V4_POWER_KW = jnp.asarray([
    0.0, 0.0, 40.518011517569214, 177.67162506419703, 403.900880943964,
    737.5889584824021, 1187.1774030611875, 1239.245945375778, 1292.5184293723503,
    1347.3213147477102, 1403.2573725578948, 1460.7011898730707, 1519.6419125979983,
    1580.174365096404, 1642.1103166918167, 1705.758292831, 1771.1659528893977,
    2518.553107505315, 3448.381605840943, 3552.140809000129, 3657.9545431794127,
    3765.121299313842, 3873.928844315059, 3984.4800226955504, 4096.582833096852,
    4210.721306623712, 4326.154305853405, 4443.395565353604, 4562.497934188341,
    4683.419890251577, 4806.164748311019, 4929.931918769215, 5000.0, 5000.0, 5000.0,
    5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0,
    5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 0.0, 0.0,
])
_V4_THRUST = jnp.asarray([
    0.0, 0.0, 1.132034888, 0.999470963, 0.917697381, 0.860849503, 0.815371198,
    0.811614904, 0.807939328, 0.80443352, 0.800993851, 0.79768116, 0.794529244,
    0.791495834, 0.788560434, 0.787217182, 0.787127977, 0.785839257, 0.783812219,
    0.783568108, 0.783328285, 0.781194418, 0.777292539, 0.773464375, 0.769690236,
    0.766001924, 0.762348072, 0.758760824, 0.755242872, 0.751792927, 0.748434131,
    0.745113997, 0.717806682, 0.672204789, 0.63831272, 0.610176496, 0.585456847,
    0.563222111, 0.542912273, 0.399312061, 0.310517829, 0.248633226, 0.203543725,
    0.169616419, 0.143478955, 0.122938861, 0.106515296, 0.093026095, 0.081648606,
    0.072197368, 0.064388275, 0.057782745, 0.0, 0.0,
])
# fmt: on


class TurbineSpec(NamedTuple):
    """Turbine geometry and power/thrust tables as a PyTree (single shared spec).

    Power evaluation is one interpolation of ``power_table`` scaled by ``power_scale``
    (kept as an explicit post-multiply to preserve v3's exact float arithmetic order):
    v3 tabulates the inner power ``0.5·A·Cp·gen_eff·ws³`` scaled by ``ref_density``, v4
    the absolute kW output scaled by ``1e3``. ``ct_fill_low``/``ct_fill_high`` are the
    scipy ``interp1d`` fill values applied outside the table before the ``C_t`` clip.
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


def nrel5mw_v3() -> TurbineSpec:
    """NREL-5MW as shipped in FLORIS 3.5 (verbatim; the default, zero numeric change)."""
    return TurbineSpec(
        rotor_diameter=D,
        hub_height=HUB_HEIGHT,
        pP=pP,
        tsr=TSR,
        generator_efficiency=GENERATOR_EFFICIENCY,
        ref_density=REF_DENSITY,
        wind_speed_table=WIND_SPEED,
        thrust_table=THRUST,
        power_table=_INNER_POWER,
        power_scale=REF_DENSITY,
        ct_fill_low=0.0001,
        ct_fill_high=0.9999,
    )


def nrel5mw_v4() -> TurbineSpec:
    """NREL-5MW as shipped in FLORIS 4.6.6 (cosine-loss, absolute-kW power table)."""
    return TurbineSpec(
        rotor_diameter=125.88,
        hub_height=HUB_HEIGHT,
        pP=1.88,  # cosine_loss_exponent_yaw
        tsr=TSR,
        generator_efficiency=0.944,  # baked into the absolute power table
        ref_density=REF_DENSITY,
        wind_speed_table=_V4_WIND_SPEED,
        thrust_table=_V4_THRUST,
        power_table=_V4_POWER_KW,
        power_scale=1e3,
        ct_fill_low=0.0001,
        ct_fill_high=0.0001,
    )


DEFAULT_TURBINE: Final = nrel5mw_v3()


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
