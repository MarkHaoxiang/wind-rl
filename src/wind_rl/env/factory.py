"""The single env-construction entry point: :func:`make_env`.

Assembles the full co-design environment pipeline on FLORIS:

    FLORIS case (from scenario / fixed layout)
      -> DesignableWindFarmEnv (yaw control only)
      -> aec_to_parallel
      -> WfcrlCoDesignWrapper
      -> TransformedEnv(Compose(RewardNormalisation, RewardSum, RemoveEmptySpecs))

The signature keeps ``simulator`` open for a future FastFarm backend, but only
FLORIS is implemented. ``layout`` and ``reset_policy`` are the injection points
for the designer/buffer machinery that arrives in a later task.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pettingzoo.utils.conversions import aec_to_parallel
from tensordict.nn import TensorDictModule
from torchrl.envs import Compose, RemoveEmptySpecs, RewardSum, TransformedEnv

from wind_rl.env.transforms import RewardNormalisation
from wind_rl.env.windfarm import GROUP_NAME, build_designable_windfarm
from wind_rl.env.wrapper import WfcrlCoDesignWrapper
from wind_rl.scenario import ScenarioConfig

_REWARD_KEY = (GROUP_NAME, "reward")
_EPISODE_REWARD_KEY = (GROUP_NAME, "episode_reward")


def default_layout(scenario: ScenarioConfig) -> NDArray[np.float64]:
    """A deterministic grid layout inside the scenario's map, for bootstrapping.

    Used as the initial farm when no explicit ``layout`` is given. Real layouts
    come from a designer at reset time (T4+); this only needs to be a valid,
    in-bounds starting point.
    """
    n = scenario.n_turbines
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_x = scenario.map_x_length / cols
    cell_y = scenario.map_y_length / rows
    coords = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        r, c = divmod(i, cols)
        coords[i] = (c * cell_x + cell_x / 2.0, r * cell_y + cell_y / 2.0)
    return coords


def make_env(
    mode: Literal["train", "eval", "reference"],
    scenario: ScenarioConfig,
    layout: NDArray[np.float64] | None = None,
    reset_policy: TensorDictModule | None = None,
    simulator: Literal["floris"] = "floris",
    device: str | None = None,
) -> TransformedEnv:
    """Build the co-design :class:`TransformedEnv` for ``scenario``.

    Parameters
    ----------
    mode:
        Accepted for interface stability; behaviour is currently identical
        across ``"train"``/``"eval"``/``"reference"``.
    layout:
        ``(N, 2)`` initial turbine coordinates. Defaults to
        :func:`default_layout`.
    simulator:
        Only ``"floris"`` is implemented.
    """
    if simulator != "floris":
        raise NotImplementedError(f"Unsupported simulator {simulator!r}")

    _ = mode  # reserved; no behavioural branch yet

    coords = default_layout(scenario) if layout is None else np.asarray(layout)
    if coords.shape != (scenario.n_turbines, 2):
        raise ValueError(
            f"layout must have shape ({scenario.n_turbines}, 2), got {coords.shape}"
        )

    aec_env = build_designable_windfarm(
        scenario=scenario,
        xcoords=coords[:, 0],
        ycoords=coords[:, 1],
    )
    parallel_env = aec_to_parallel(aec_env)
    wrapped = WfcrlCoDesignWrapper(
        env=parallel_env,
        reset_policy=reset_policy,
        device=device,
    )

    return TransformedEnv(
        wrapped,
        Compose(
            RewardNormalisation(in_keys=[_REWARD_KEY]),
            RewardSum(in_keys=[_REWARD_KEY], out_keys=[_EPISODE_REWARD_KEY]),
            RemoveEmptySpecs(),
        ),
    )
