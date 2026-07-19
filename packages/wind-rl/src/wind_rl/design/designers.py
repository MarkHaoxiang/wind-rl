"""Static/manual designers and the :func:`create_designer` factory."""

from __future__ import annotations

from typing import assert_never, override

import numpy as np
import torch
from numpy.typing import NDArray
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from wind_rl.design.base import (
    LAYOUT_WEIGHTS_KEY,
    Designer,
    DesignerConfig,
    FixedDesignerConfig,
    ManualDesignerConfig,
    RandomDesignerConfig,
)
from wind_rl.design.geometry import sample_feasible_layout
from wind_rl.scenario import ScenarioConfig, real_farm_layout


class _LayoutSource(nn.Module):
    def __init__(self, designer: Designer) -> None:
        super().__init__()
        self._designer = designer

    @override
    def forward(self) -> torch.Tensor:
        layout = self._designer.generate_layout_batch(1)[0]
        return torch.as_tensor(layout, dtype=torch.float32)


def _live_layout_module(designer: Designer) -> TensorDictModule:
    return TensorDictModule(
        _LayoutSource(designer), in_keys=[], out_keys=[LAYOUT_WEIGHTS_KEY]
    )


class RandomDesigner:
    def __init__(self, scenario: ScenarioConfig, seed: int | None = None) -> None:
        self._scenario = scenario
        self._rng = np.random.default_rng(seed)

    def generate_layout_batch(self, batch_size: int) -> NDArray[np.float64]:
        return np.stack(
            [
                sample_feasible_layout(self._scenario, self._rng)
                for _ in range(batch_size)
            ]
        )

    def update(self, sampling_td: TensorDict) -> None:
        return None

    def to_td_module(self) -> TensorDictModule:
        return _live_layout_module(self)

    def get_logs(self) -> dict[str, float]:
        return {}


class FixedDesigner:
    def __init__(self, layout: NDArray[np.float64]) -> None:
        self._layout = np.asarray(layout, dtype=np.float64)

    def generate_layout_batch(self, batch_size: int) -> NDArray[np.float64]:
        return np.broadcast_to(self._layout, (batch_size, *self._layout.shape)).copy()

    def update(self, sampling_td: TensorDict) -> None:
        return None

    def to_td_module(self) -> TensorDictModule:
        return _live_layout_module(self)

    def get_logs(self) -> dict[str, float]:
        return {}


class ManualDesigner(FixedDesigner):
    def __init__(self, farm: str) -> None:
        # Real farm layouts are feasible by construction and use their own
        # coordinate frame (often outside a scenario's map bounds), so they skip
        # the scenario feasibility check that the random/fixed designers enforce.
        super().__init__(real_farm_layout(farm))


def create_designer(cfg: DesignerConfig, scenario: ScenarioConfig) -> Designer:
    match cfg:
        case RandomDesignerConfig():
            return RandomDesigner(scenario, seed=cfg.seed)
        case FixedDesignerConfig():
            rng = np.random.default_rng(cfg.seed)
            return FixedDesigner(sample_feasible_layout(scenario, rng))
        case ManualDesignerConfig():
            return ManualDesigner(cfg.farm)
        case _:
            assert_never(cfg)
