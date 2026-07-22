from typing import Final

from windrl_engine.farm.turbine import HUB_HEIGHT
from windrl_engine.physics.frame import RotorField, Scalar

SHEAR: Final = 0.12
AMBIENT_TI: Final = 0.06
AIR_DENSITY: Final = 1.225


def initial_flow(z_grid: RotorField, speed: Scalar) -> tuple[RotorField, RotorField]:
    """Sheared inflow u = speed*(z/HH)^shear and its vertical gradient du/dz."""
    wind_profile = (z_grid / HUB_HEIGHT) ** SHEAR
    dwind_profile = SHEAR * (1.0 / HUB_HEIGHT) ** SHEAR * z_grid ** (SHEAR - 1.0)
    return speed * wind_profile, speed * dwind_profile
