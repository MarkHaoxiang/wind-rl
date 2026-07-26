from typing import Final

from windrl_engine.farm.turbine import DEFAULT_TURBINE, TurbineSpec
from windrl_engine.physics.frame import QueryField, Scalar

SHEAR: Final = 0.12
# A scenario input, not a model parameter: FLORIS takes turbulence_intensities per
# run, so this is our choice of operating point rather than an upstream default.
AMBIENT_TI: Final = 0.06
AIR_DENSITY: Final = 1.225


def initial_flow(
    z_grid: QueryField, speed: Scalar, *, turbine: TurbineSpec = DEFAULT_TURBINE
) -> tuple[QueryField, QueryField]:
    """Sheared inflow u = speed*(z/HH)^shear and its vertical gradient du/dz."""
    hub = turbine.hub_height
    wind_profile = (z_grid / hub) ** SHEAR
    dwind_profile = SHEAR * (1.0 / hub) ** SHEAR * z_grid ** (SHEAR - 1.0)
    return speed * wind_profile, speed * dwind_profile
