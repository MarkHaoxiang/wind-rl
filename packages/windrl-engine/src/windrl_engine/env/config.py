from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field

from windrl_engine.env.actions import ControlMode
from windrl_engine.farm.layout import (
    FarmLayout,
    ablaincourt,
    horns_rev2,
    turb3_row1,
)

LayoutName = Literal["turb3_row1", "ablaincourt", "horns_rev2"]

_LAYOUT_BUILDERS: dict[LayoutName, Callable[[], FarmLayout]] = {
    "turb3_row1": turb3_row1,
    "ablaincourt": ablaincourt,
    "horns_rev2": horns_rev2,
}


class WindFarmEnvConfig(BaseModel):
    """User-facing construction surface for :class:`BatchedWindFarmEnv`.

    ``layout`` is a named reference or explicit ``(x, y)`` coordinates (meters,
    world frame). Validated once at env construction and converted to static
    jit args + arrays; pydantic objects never enter jitted code.
    """

    model_config = ConfigDict(extra="forbid")

    layout: LayoutName | list[tuple[float, float]] = "turb3_row1"
    yaw_step: float = Field(default=5.0, gt=0.0)
    control_mode: ControlMode = "continuous"
    horizon: int = Field(default=500, ge=1)
    load_coef: float = Field(default=0.1, ge=0.0)
    n_envs: int = Field(default=1, ge=1)

    def build_layout(self) -> FarmLayout:
        if isinstance(self.layout, str):
            return _LAYOUT_BUILDERS[self.layout]()
        xs = [c[0] for c in self.layout]
        ys = [c[1] for c in self.layout]
        return FarmLayout(x=jnp.asarray(xs), y=jnp.asarray(ys))
