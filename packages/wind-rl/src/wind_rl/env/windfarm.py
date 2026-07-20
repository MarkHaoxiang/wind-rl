"""A designable FLORIS/wfcrl multi-agent wind-farm environment.

:class:`DesignableWindFarmEnv` extends wfcrl's :class:`MAWindFarmEnv` so that the
turbine *layout* (xy coordinates) is exposed both in the global ``state()`` and
in every agent's local observation, and so that a ``reset`` can rebuild the
underlying :class:`WindFarmMDP` (and FLORIS interface) with a brand new layout.
This is the environment side of the co-design loop: the designer proposes a
layout, the environment is rebuilt around it, and the MAPPO policy controls the
turbines within it.

Ported from DiCoDe's ``DesignableMAWindFarmEnv`` (torchrl 0.9 era). The port keeps
DiCoDe's MDP-rebuild-on-reset mechanism and its FLORIS ``__simul__`` case-file
cleanup, and adds a numpy>=2 compatible ``_join_actions`` override (see below).
"""

from __future__ import annotations

import copy
import math
import os
import shutil
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray
from wfcrl.environments.data_cases import FlorisCase
from wfcrl.environments.registration import get_default_control, validate_case
from wfcrl.interface import FlorisInterface
from wfcrl.mdp import WindFarmMDP
from wfcrl.multiagent_env import MAWindFarmEnv
from wfcrl.rewards import DoNothingReward

from wind_rl.scenario import ScenarioConfig

if TYPE_CHECKING:
    from floris.tools import (
        FlorisInterface as FlorisSimulator,  # type: ignore[import-untyped]
    )


class ResetOptions(TypedDict, total=False):
    """``reset(options=...)`` payload honoured by :class:`DesignableWindFarmEnv`."""

    xcoords: ArrayLike
    ycoords: ArrayLike
    wind_direction: float
    wind_speed: float


class DesignableWindFarmEnv(MAWindFarmEnv):  # type: ignore[misc]
    """A ``MAWindFarmEnv`` whose turbine layout is observable and re-designable.

    Compared to the base wfcrl environment this class:

    * adds a ``"layout"`` entry (shape ``(num_turbines, 2)``) to the global
      ``state_space``/``state()`` and a per-agent ``"layout"`` entry (shape
      ``(2,)``) to every agent's observation space;
    * accepts ``reset(options={"xcoords": ..., "ycoords": ...})``, rebuilding the
      underlying :class:`WindFarmMDP` (and, for FLORIS, its ``__simul__`` case
      files) so that a genuinely different farm is simulated;
    * overrides ``_join_actions`` for numpy>=2 compatibility.
    """

    # Attributes owned by the (untyped) wfcrl base, reassigned on redesign.
    mdp: WindFarmMDP
    farm_case: FlorisCase

    def __init__(
        self,
        interface: type[FlorisInterface],
        farm_case: FlorisCase,
        controls: dict[str, tuple[float, float, float]],
        *,
        continuous_control: bool = True,
        start_iter: int = 0,
        max_num_steps: int = 500,
        load_coef: float = 0.1,
        scenario: ScenarioConfig | None = None,
        render_mode: Literal["rgb_array"] | None = None,
    ) -> None:
        # DiCoDe offsets the horizon by 1 to work around an off-by-one in the
        # wfcrl truncation logic; keep the same behaviour.
        super().__init__(
            interface=interface,
            farm_case=farm_case,
            controls=controls,
            continuous_control=continuous_control,
            reward_shaper=DoNothingReward(),
            start_iter=start_iter,
            max_num_steps=max_num_steps + 1,
            load_coef=load_coef,
        )
        self.interface_cls = interface
        self.start_iter = start_iter
        self.scenario = scenario
        self.render_mode = render_mode

        self.state_space["layout"] = spaces.Box(
            low=0.0, high=np.inf, shape=(self.num_turbines, 2), dtype=np.float32
        )

    @property
    def floris(self) -> FlorisSimulator:
        """The live ``floris.tools.FlorisInterface`` behind the wfcrl interface.

        The visualiser uses it to compute the wake-resolved hub-height flow plane.
        """
        return self.mdp.interface.fi

    def reset(
        self,
        seed: int | None = None,
        options: ResetOptions | None = None,
    ) -> None:
        """Reset the environment, optionally rebuilding the MDP with new coords.

        If ``options`` contains ``"xcoords"`` and/or ``"ycoords"`` the underlying
        :class:`WindFarmMDP` is rebuilt around the new layout. For FLORIS the
        stale ``__simul__`` case directory of the previous interface is deleted
        first, so those directories do not accumulate unboundedly across resets.
        """
        if options is not None and ("xcoords" in options or "ycoords" in options):
            new_farm_case = copy.copy(self.mdp.farm_case)
            if "xcoords" in options:
                new_farm_case.xcoords = _as_coord_list(options["xcoords"])
            if "ycoords" in options:
                new_farm_case.ycoords = _as_coord_list(options["ycoords"])
            if len(new_farm_case.xcoords) != len(new_farm_case.ycoords):
                raise ValueError("xcoords and ycoords must have the same length")

            self._cleanup_floris_files()

            self.mdp = WindFarmMDP(
                interface=self.interface_cls,
                farm_case=new_farm_case,
                controls=self.controls,
                continuous_control=self.continuous_control,
                start_iter=self.start_iter,
                horizon=self.start_iter + self.max_num_steps,
            )
            self.farm_case = new_farm_case

        super().reset(seed, options)

    def _cleanup_floris_files(self) -> None:
        """Delete the previous FLORIS ``__simul__`` case directory, if any."""
        interface = self.mdp.interface
        if not isinstance(interface, FlorisInterface):
            return
        simul_file = getattr(interface, "simul_file", None)
        if simul_file is None or not os.path.exists(simul_file):
            return
        os.remove(simul_file)
        shutil.rmtree(os.path.dirname(simul_file), ignore_errors=True)

    # NDArray[Any]: wfcrl's own state keys are float64, ours ("layout") is
    # float32 -- the merged dict is genuinely dtype-heterogeneous, so no single
    # scalar type is honest here.
    def state(self) -> dict[str, NDArray[Any]]:  # type: ignore[explicit-any]
        """Global state, augmented with the turbine ``"layout"``."""
        base_state = super().state()
        state: dict[str, NDArray[Any]] = OrderedDict(base_state)  # type: ignore[explicit-any]
        state["layout"] = np.stack(
            [self.mdp.farm_case.xcoords, self.mdp.farm_case.ycoords], axis=-1
        ).astype(np.float32)
        return state

    def _join_actions(
        self, agent_actions: Mapping[str, Mapping[str, ArrayLike]]
    ) -> dict[str, NDArray[np.float32]]:
        """Assemble per-agent actions into a joint action array.

        Overrides the wfcrl implementation, which assigns a shape-``(1,)`` array
        into a scalar slot (``joint[control][j] = action[control][:]``). numpy>=2
        rejects that as "setting an array element with a sequence", so we flatten
        each agent's control value to a scalar explicitly.
        """
        joint_action: dict[str, NDArray[np.float32]] = {
            control: np.zeros(self.num_turbines, dtype=np.float32)
            for control in self.mdp.controls
        }
        for j, action in enumerate(agent_actions.values()):
            for control, value in action.items():
                joint_action[control][j] = np.asarray(value).reshape(-1)[0]
        return joint_action

    def _build_agent_spaces(self) -> None:
        super()._build_agent_spaces()
        for agent in self.possible_agents:
            self._obs_spaces[agent]["layout"] = spaces.Box(
                low=0.0, high=np.inf, shape=(2,), dtype=np.float32
            )


def _as_coord_list(coords: ArrayLike) -> list[float]:
    """Coerce coordinates to a plain ``list[float]``.

    FLORIS serialises the case to YAML on rebuild and cannot represent numpy
    scalar types, so we must hand it native Python floats.
    """
    return np.asarray(coords, dtype=float).reshape(-1).tolist()


def build_designable_windfarm(
    scenario: ScenarioConfig,
    xcoords: ArrayLike,
    ycoords: ArrayLike,
    *,
    render: bool = False,
) -> DesignableWindFarmEnv:
    """Construct a yaw-controlled :class:`DesignableWindFarmEnv` from a scenario."""
    case = FlorisCase(
        num_turbines=scenario.n_turbines,
        xcoords=_as_coord_list(xcoords),
        ycoords=_as_coord_list(ycoords),
        dt=60,
        buffer_window=1,
        t_init=0,
    )
    validate_case(scenario.name, case)
    return DesignableWindFarmEnv(
        interface=FlorisInterface,
        farm_case=case,
        controls=get_default_control(["yaw"]),
        start_iter=math.ceil(case.t_init / case.dt),
        max_num_steps=scenario.max_steps,
        scenario=scenario,
        render_mode="rgb_array" if render else None,
    )
