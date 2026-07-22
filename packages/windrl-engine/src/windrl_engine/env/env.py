from typing import NamedTuple

from jaxtyping import Array, Bool, Float, Key

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.state import FarmState
from windrl_engine.farm.wind import WindCondition


class Observation(NamedTuple):
    yaw: Float[Array, "turbines"]
    freewind: Float[Array, "2"]  # [speed, direction]
    wind_speed: Float[Array, "turbines"]  # local ∛(mean u³)
    wind_direction: Float[Array, "turbines"]


def reset(
    layout: FarmLayout,
    key: Key[Array, ""],
    wind: WindCondition | None = None,
) -> tuple[FarmState, Observation]:
    raise NotImplementedError


def step(
    layout: FarmLayout,
    state: FarmState,
    action: Float[Array, "turbines"],
    *,
    yaw_step: float,
    load_coef: float,
    horizon: int,
) -> tuple[FarmState, Observation, Float[Array, ""], Bool[Array, ""]]:
    raise NotImplementedError
