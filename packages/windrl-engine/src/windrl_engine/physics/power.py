from jaxtyping import Array, Float

from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.solver import FlowSolution


def turbine_powers(
    u: Float[Array, "turbines grid grid"], yaw: Float[Array, "turbines"]
) -> Float[Array, "turbines"]:
    raise NotImplementedError


def load_proxies(solution: FlowSolution) -> Float[Array, "turbines 4"]:
    raise NotImplementedError


def local_wind(
    solution: FlowSolution, wind: WindCondition
) -> tuple[Float[Array, "turbines"], Float[Array, "turbines"]]:
    raise NotImplementedError
