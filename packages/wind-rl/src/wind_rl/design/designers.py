"""Static/manual designers and the :func:`create_designer` factory."""

from __future__ import annotations

from pathlib import Path
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
    FlowMapDesignerConfig,
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


class FlowMapDesigner:
    """Samples layouts from a trained flow-map prior, few-step, then projects to
    feasibility (hard SLSQP). No critic guidance yet -- unconditional prior only."""

    def __init__(
        self,
        scenario: ScenarioConfig,
        checkpoint: Path,
        sampling_steps: int,
        device: str = "cpu",
    ) -> None:
        # Imported lazily: wind_rl.generative imports wind_rl.design.geometry, so a
        # module-level import here would cycle through the design package __init__.
        from wind_rl.generative.flowmap import load_flowmap

        self._scenario = scenario
        self._model = load_flowmap(str(checkpoint), device=device)
        if self._model.arch.n_turbines != scenario.n_turbines:
            raise ValueError(
                f"checkpoint prior has {self._model.arch.n_turbines} turbines, "
                f"scenario {scenario.name!r} has {scenario.n_turbines}"
            )
        self._steps = sampling_steps
        self._total_nfe = 0

    def generate_layout_batch(self, batch_size: int) -> NDArray[np.float64]:
        from wind_rl.generative.constraints import project_slsqp
        from wind_rl.generative.flowmap import sample_layouts

        raw = sample_layouts(self._model, batch_size, self._steps)
        # One velocity evaluation per Euler step, independent of batch size.
        self._total_nfe += self._steps
        return project_slsqp(raw, self._scenario)

    def update(self, sampling_td: TensorDict) -> None:
        return None

    def to_td_module(self) -> TensorDictModule:
        return _live_layout_module(self)

    def get_logs(self) -> dict[str, float]:
        return {
            "nfe_per_batch": float(self._steps),
            "total_nfe": float(self._total_nfe),
        }


def create_designer(cfg: DesignerConfig, scenario: ScenarioConfig) -> Designer:
    match cfg:
        case RandomDesignerConfig():
            return RandomDesigner(scenario, seed=cfg.seed)
        case FixedDesignerConfig():
            rng = np.random.default_rng(cfg.seed)
            return FixedDesigner(sample_feasible_layout(scenario, rng))
        case ManualDesignerConfig():
            return ManualDesigner(cfg.farm)
        case FlowMapDesignerConfig():
            return FlowMapDesigner(scenario, cfg.checkpoint, cfg.sampling_steps)
        case _:
            assert_never(cfg)
