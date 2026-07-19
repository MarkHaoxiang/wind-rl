"""The :class:`Designer` protocol and its discriminated-union config."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from wind_rl.config import Config

#: Reset-policy output key holding the sampled ``(N, 2)`` layout; the contract
#: between a designer's ``to_td_module`` and the env wrapper that consumes it.
LAYOUT_WEIGHTS_KEY = ("environment_design", "layout_weights")


@runtime_checkable
class Designer(Protocol):
    """Produces turbine layouts for the co-design env.

    ``to_td_module`` returns the reset policy consumed by
    :class:`~wind_rl.env.wrapper.WfcrlCoDesignWrapper`: on every env reset it
    writes one ``(n_turbines, 2)`` float32 tensor of absolute xy map
    coordinates (metres) to ``("environment_design", "layout_weights")``, which
    the wrapper reads and forwards as the reset ``xcoords``/``ycoords`` that
    rebuild the farm.
    """

    def generate_layout_batch(self, batch_size: int) -> NDArray[np.float64]: ...
    def update(self, sampling_td: TensorDict) -> None: ...
    def to_td_module(self) -> TensorDictModule: ...
    def get_logs(self) -> dict[str, float]: ...


class RandomDesignerConfig(Config):
    kind: Literal["random"] = "random"
    seed: int | None = None


class FixedDesignerConfig(Config):
    kind: Literal["fixed"] = "fixed"
    seed: int | None = None


class ManualDesignerConfig(Config):
    kind: Literal["manual"] = "manual"
    farm: str


class FlowMapDesignerConfig(Config):
    kind: Literal["flow_map"] = "flow_map"
    checkpoint: Path
    sampling_steps: int = Field(default=4, ge=1, le=8)


DesignerConfig = Annotated[
    RandomDesignerConfig
    | FixedDesignerConfig
    | ManualDesignerConfig
    | FlowMapDesignerConfig,
    Field(discriminator="kind"),
]
