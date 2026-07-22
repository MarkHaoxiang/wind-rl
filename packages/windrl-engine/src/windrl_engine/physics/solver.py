from typing import NamedTuple

from jaxtyping import Array, Float

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.wind import WindCondition


class FlowSolution(NamedTuple):
    u: Float[Array, "turbines grid grid"]  # m/s, original turbine order
    v: Float[Array, "turbines grid grid"]
    w: Float[Array, "turbines grid grid"]
    turbulence_intensity: Float[Array, "turbines"]


def solve_farm(
    layout: FarmLayout, wind: WindCondition, yaw: Float[Array, "turbines"]
) -> FlowSolution:
    raise NotImplementedError
