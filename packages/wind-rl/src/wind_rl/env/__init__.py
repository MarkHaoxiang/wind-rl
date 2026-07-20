"""Co-design wind-farm environment pipeline (FLORIS, torchrl 0.11)."""

from __future__ import annotations

from wind_rl.env.factory import default_layout, make_env
from wind_rl.env.render import RenderConfig, render_farm
from wind_rl.env.transforms import RewardNormalisation
from wind_rl.env.windfarm import DesignableWindFarmEnv, build_designable_windfarm
from wind_rl.env.wrapper import WfcrlCoDesignWrapper
from wind_rl.static import ENV_NAME, GROUP_NAME

__all__ = [
    "ENV_NAME",
    "GROUP_NAME",
    "DesignableWindFarmEnv",
    "RenderConfig",
    "RewardNormalisation",
    "WfcrlCoDesignWrapper",
    "build_designable_windfarm",
    "default_layout",
    "make_env",
    "render_farm",
]
