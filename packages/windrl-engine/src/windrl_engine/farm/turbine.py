"""NREL-5MW turbine spec, read from the FLORIS-packaged ``nrel_5MW.yaml`` at import."""

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, TypeAlias

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]
from jaxtyping import Array, Float

TABLE_SIZE: Final = 54

# FLORIS cosine-loss operation-model constants: code, not turbine data.
POWER_SCALE: Final = 1e3  # the yaml power table is absolute kW
CT_FILL: Final = 0.0001  # two-sided fill outside the wind-speed table
CT_MIN: Final = 0.0001
CT_MAX: Final = 0.9999

if TYPE_CHECKING:
    # jaxtyping needs the bare `np.ndarray` class, off which mypy reads numpy's
    # `Any`-defaulted type parameters -- rejected by `disallow_any_explicit`.
    TurbineTable: TypeAlias = npt.NDArray[np.float64]
else:
    TurbineTable = Float[np.ndarray, f"table={TABLE_SIZE}"]


class TurbineSpec(NamedTuple):
    """Turbine geometry and power/thrust tables as a PyTree (single shared spec).

    Every field is a value ``nrel_5MW.yaml`` supplies; the operation model's own
    constants live beside the lookups below.

    Tables stay numpy float64 so their JAX dtype resolves at trace time under
    whatever ``jax_enable_x64`` is then active, instead of being frozen by the
    config that happened to hold when this module was first imported.
    """

    rotor_diameter: float
    hub_height: float
    pP: float
    tsr: float
    ref_density: float
    wind_speed_table: TurbineTable
    thrust_table: TurbineTable
    power_table: TurbineTable


def _floris_turbine_library() -> Path:
    # find_spec locates the installed package without executing `floris/__init__`,
    # which drags in scipy + shapely just to read one yaml (771ms -> 48ms import).
    floris = importlib.util.find_spec("floris")
    if floris is None or floris.submodule_search_locations is None:
        raise ImportError("floris is required for its packaged turbine library")
    return Path(floris.submodule_search_locations[0]) / "turbine_library"


def _load_nrel5mw_v4() -> TurbineSpec:
    path = _floris_turbine_library() / "nrel_5MW.yaml"
    turbine = yaml.safe_load(path.read_text())
    table = turbine["power_thrust_table"]
    return TurbineSpec(
        rotor_diameter=float(turbine["rotor_diameter"]),
        hub_height=float(turbine["hub_height"]),
        pP=float(table["cosine_loss_exponent_yaw"]),
        tsr=float(turbine["TSR"]),
        ref_density=float(table["ref_air_density"]),
        wind_speed_table=np.asarray(table["wind_speed"], dtype=np.float64),
        thrust_table=np.asarray(table["thrust_coefficient"], dtype=np.float64),
        power_table=np.asarray(table["power"], dtype=np.float64),
    )


DEFAULT_TURBINE: Final = _load_nrel5mw_v4()


def nrel5mw_v4() -> TurbineSpec:
    """NREL-5MW as shipped in FLORIS 4.6.6 (cosine-loss, absolute-kW power table)."""
    return DEFAULT_TURBINE


def ct_lookup(
    spec: TurbineSpec, wind_speed: Float[Array, "*shape"]
) -> Float[Array, "*shape"]:
    """Thrust coefficient C_t for ``spec``; linear interp, fill, then clip to (1e-4, 0.9999)."""
    ws = spec.wind_speed_table
    ct = jnp.interp(wind_speed, ws, spec.thrust_table)
    ct = jnp.where(wind_speed < ws[0], CT_FILL, ct)
    ct = jnp.where(wind_speed > ws[-1], CT_FILL, ct)
    return jnp.clip(ct, CT_MIN, CT_MAX)


def power_lookup(
    spec: TurbineSpec, u_eff: Float[Array, "*shape"]
) -> Float[Array, "*shape"]:
    """Electrical power (W) for ``spec`` from the effective rotor velocity; fill 0.0."""
    ws = spec.wind_speed_table
    power = jnp.interp(u_eff, ws, spec.power_table)
    power = jnp.where(u_eff < ws[0], 0.0, power)
    power = jnp.where(u_eff > ws[-1], 0.0, power)
    return power * POWER_SCALE
