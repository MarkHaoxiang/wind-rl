"""Wind farm scenario configuration.

Fixed metre-scale layouts (real farms, reference rows) live in
``windrl_engine.farm.layout``; ``ScenarioConfig`` carries only the map geometry
and control parameters shared across the layout-feasibility and generative code.
"""

from __future__ import annotations

from pydantic import Field

from wind_rl.config import Config


class ScenarioConfig(Config):
    name: str
    n_turbines: int = Field(ge=1)
    max_steps: int = Field(gt=0)
    map_x_length: float = Field(gt=0)
    map_y_length: float = Field(gt=0)
    min_distance_between_turbines: float = Field(gt=0)
    # When set, every reset uses this fixed wind (degrees, m/s) instead of
    # per-episode random sampling -- a deterministic, high-headroom regime for
    # evaluation and smoke tests. The override is applied as one (direction,
    # speed) pair keyed on fixed_wind_direction, so fixed_wind_speed only takes
    # effect once fixed_wind_direction is set -- speed alone cannot be fixed.
    fixed_wind_direction: float | None = None
    fixed_wind_speed: float = Field(default=8.0, gt=0)
    # Weight on the fatigue-load penalty in the per-step reward
    # (power/u_inf^3 - load_coef * mean|load|). The FLORIS/WFCRL default is 0.1;
    # the NeurIPS-2024 MAPPO benchmark trains with 1.0.
    load_coef: float = Field(default=0.1, ge=0)
