from collections.abc import Callable
from typing import Annotated, Literal

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field

from windrl_engine.env.actions import ControlMode, Fidelity
from windrl_engine.farm.layout import (
    FarmLayout,
    ablaincourt,
    horns_rev2,
    turb3_row1,
)
from windrl_engine.farm.turbine import TurbineSpec, nrel5mw_v4

LayoutName = Literal["turb3_row1", "ablaincourt", "horns_rev2"]
TurbineName = Literal["nrel5mw_v4"]

_LAYOUT_BUILDERS: dict[LayoutName, Callable[[], FarmLayout]] = {
    "turb3_row1": turb3_row1,
    "ablaincourt": ablaincourt,
    "horns_rev2": horns_rev2,
}

_TURBINE_BUILDERS: dict[TurbineName, Callable[[], TurbineSpec]] = {
    "nrel5mw_v4": nrel5mw_v4,
}


class WindFarmEnvConfig(BaseModel):
    """User-facing construction surface for :class:`BatchedWindFarmEnv`.

    ``layout`` is a named reference or explicit ``(x, y)`` coordinates (meters,
    world frame). Validated once at env construction and converted to static
    jit args + arrays; pydantic objects never enter jitted code. It is the
    shared default only: per-env co-design layouts arrive at ``reset``, not here.
    """

    model_config = ConfigDict(extra="forbid")

    # min_length: a 0-turbine farm builds and traces fine, then dies inside the
    # wake solve on a zero-size reduction, far from the config that caused it.
    layout: LayoutName | Annotated[list[tuple[float, float]], Field(min_length=1)] = (
        "turb3_row1"
    )
    yaw_step: float = Field(default=5.0, gt=0.0)
    control_mode: ControlMode = "continuous"
    fidelity: Fidelity = "floris"
    turbine: TurbineName = "nrel5mw_v4"
    horizon: int = Field(default=500, ge=1)
    load_coef: float = Field(default=0.1, ge=0.0)
    n_envs: int = Field(default=1, ge=1)

    def build_layout(self) -> FarmLayout:
        if isinstance(self.layout, str):
            return _LAYOUT_BUILDERS[self.layout]()
        xs = [c[0] for c in self.layout]
        ys = [c[1] for c in self.layout]
        return FarmLayout(x=jnp.asarray(xs), y=jnp.asarray(ys))

    def build_turbine(self) -> TurbineSpec:
        return _TURBINE_BUILDERS[self.turbine]()
